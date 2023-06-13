import os
import torch
from torch.nn import CrossEntropyLoss
from transformers import OpenAIGPTLMHeadModel
from transformers import get_linear_schedule_with_warmup
import pytorch_lightning as pl
import logging
logger = logging.getLogger(__name__)

class OpenAIGPTLMHeadModel(OpenAIGPTLMHeadModel):
    '''
        We need to override in order to take attention_mask into account when doing batch inference.
        This allows for batch inference
    '''
    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

class BlindGST(pl.LightningModule):
    def __init__(self, model_name_or_path: str, num_special_tokens: int, pad_token_id: int, eos_token_id: int, **kwargs):
        super().__init__()
        assert model_name_or_path and num_special_tokens and pad_token_id and eos_token_id
        self.save_hyperparameters()
        self.model_name_or_path = model_name_or_path
        self.model = OpenAIGPTLMHeadModel.from_pretrained(self.model_name_or_path)
        self.model.resize_token_embeddings(num_special_tokens + self.model.config.vocab_size)
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

    @staticmethod
    def add_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BlindGST")
        parser.add_argument("--model_name_or_path", type=str, default='openai-gpt')
        parser.add_argument("--weight_decay", type=float, default=0., help="Regularization parameter during training")
        parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
        parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer")
        parser.add_argument("--warmup_steps", type=int, default=0, help="Number of steps for linear warmup")
        parser.add_argument("--gradient_clip_val", type=float, default=1.0, help="Maximum norm of gradients")
        return parent_parser

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
    
    def _loss(self, output, labels):
        # We calculate loss outside because we dont want transformer to shift labels and trim input last token
        # We calculate loss only for what is after <START> and untill (including) <END>
        logits = output.logits
        loss_fct = CrossEntropyLoss() # Ignores positions where == -100
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return loss

    def training_step(self, batch, batch_idx):
        output = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            return_dict=True
        )
        loss = self._loss(output, batch['labels'])

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar = True, logger=True)
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        output = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['labels'], 
            return_dict=True
        )
        self.log("val_loss", output.loss, on_step=True, on_epoch=True, prog_bar = True, logger=True)
        return {'loss':output.loss}

    def predict_step(self, batch, batch_idx):
        # https://huggingface.co/blog/how-to-generate
        # Generate with padding -> https://github.com/huggingface/transformers/pull/7552#issue-497255933
        preds = self.generate(
            inputs=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            num_return_sequences=1,
            do_sample=True,
            # max_length=128,
            max_length=256,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            num_beams=1
        )
        return preds

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}

        return [optimizer], [scheduler]
    
    def save_preds(self, preds, tokenizer, evaluation_data_split):
        root_dir = self.hparams.default_root_dir
        path = f"{root_dir}/{evaluation_data_split}_preds.txt"
        if not os.path.exists(root_dir): os.makedirs(root_dir)

        logger.info(f"Will output inference predictions to {path}")
        filtered_texts = [] 
        for batch_preds in preds:
            # We only want what is generated after <START> so we:
            #   - Remove <START> token and everythin before
            #   - Remove <END> token
            gen_texts = tokenizer.batch_decode(batch_preds, skip_special_tokens=False)
            for t in gen_texts:
                filtered_texts += [t.split("<START>")[-1].split("<END>")[0].strip()] 

        with open(path, 'w') as f: f.write('\n'.join(filtered_texts)+'\n')


if __name__ == "__main__":
    from data_modules.blind_gst import BlindGSTDM

    dm = BlindGSTDM(
        tokenizer_name_or_path='openai-gpt',
        batch_size=32,
        preprocess_kind='word_embeddings',
        dataset_name="jigsaw",
        lexicon_name="abusive",
        similarity_strategy="top-1"
    )
    dm.prepare_data()
    dm.setup(stage="fit")

    model = BlindGST(
        model_name_or_path='openai-gpt',
        num_special_tokens=len(dm.special_tokens),
        pad_token_id=dm.tokenizer.pad_token_id,
        eos_token_id=dm.tokenizer.eos_token_id,
    )

    batch = dm.datasets['train'][:5]

    output = model(batch['input_ids'], attention_mask=batch['attention_mask'])
    loss = model._loss(output, batch['labels'])

    print(loss)