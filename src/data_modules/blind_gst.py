import os
from typing import Optional
from datasets.blind_gst import BlindGSTDataset
import pytorch_lightning as pl
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

class BlindGSTDM(pl.LightningDataModule):
    '''
        BlindGST model's datamodule to feed data into BlindGST
    '''

    def __init__(
            self, 
            tokenizer_name_or_path: str, 
            batch_size: int, 
            preprocess_kind: str,
            evaluation_data_split: str,
            dataset_name: str,
            lexicon_name: str = None,
            similarity_strategy: str = None,
            keep_common_words: bool = None,
            **kwargs
        ):
        super().__init__()
        assert preprocess_kind and dataset_name
        self.save_hyperparameters()
        self.preprocess_kind = preprocess_kind
        self.dataset_name = dataset_name
        self.lexicon_name = lexicon_name if lexicon_name else None
        self.similarity_strategy = similarity_strategy if similarity_strategy else None
        self.keep_common_words = keep_common_words if keep_common_words else None
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.batch_size = batch_size
        self.evaluation_data_split = evaluation_data_split
        self.special_tokens = ['<TOX>', '<NEU>','<CON_START>','<START>','<END>', '<PAD>']
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name_or_path, 
            use_fast=True, 
            additional_special_tokens=self.special_tokens,
            pad_token='<PAD>',
            eos_token='<END>',
            bos_token='<START>'
        )
        self.datasets = {}

    @staticmethod
    def add_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("BlidGSTDM")
        parser.add_argument("--tokenizer_name_or_path", type=str, default='openai-gpt')
        parser.add_argument("--batch_size", type=int, default=4) # I made this up bc seq_len is 2x yelp's data
        parser.add_argument("--max_seq_len", type=int, default=256) # Calculated it. Real max_len=266. But use round number -> 256
        parser.add_argument("--preprocess_kind",
                            type=str, 
                            default="word_embeddings", 
                            choices=["bert_best_head", "classification", "word_embeddings"],
                            help="Kind of preprocessing of data:\t1. bert_best_head\t2. classification\t3. word_embeddings")
        parser.add_argument("--lexicon_name", 
                                type=str, 
                                default="abusive", 
                                choices=["toxic", "ngram-refined", "abusive", "hate", "jigsaw", "paradetox"], 
                                help="1. toxic\t2. ngram-refined\t3. abusive\t4. hate\t5. paradetox\t6. jigsaw")
        parser.add_argument("--similarity_strategy", 
                                type=str, 
                                choices=["top-1", "top-3", "top-5", "top-7", "top-9"], 
                                help="1. top-1\t2. top-3\t3. top-5.\t4. top-7\t5.top-9")
        parser.add_argument("--dataset_name", 
                                type=str, 
                                default="paradetox", 
                                choices=["jigsaw", "paradetox"], 
                                help="1. paradetox\t2. jigsaw")
        parser.add_argument("--keep_common_words", 
                                action="store_true",
                                help="If present it loads the data from the keep_common_words folder, ELSE it loads from remove_common_words folder.")
        return parent_parser

    def prepare_data(self):
        AutoTokenizer.from_pretrained(self.tokenizer_name_or_path, use_fast=True, additional_special_tokens=self.special_tokens)
        # Memory optimization:
        #   Dont load all split datasets into memory, only load in setup() the required datasets for the stage we are in
        # for split in ['train', 'dev', 'test']:
        #     BlindGSTDataset(
        #         split=split, 
        #         tokenizer=self.tokenizer, 
        #         preprocess_kind=self.preprocess_kind, 
        #         dataset_name=self.dataset_name, 
        #         lexicon_name=self.lexicon_name, 
        #         similarity_strategy=self.similarity_strategy,
        #         keep_common_words=self.keep_common_words
        #     )

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            for split in ['train', 'dev']:
                self.datasets[split] = BlindGSTDataset(
                    split=split, 
                    tokenizer=self.tokenizer, 
                    preprocess_kind=self.preprocess_kind, 
                    dataset_name=self.dataset_name, 
                    lexicon_name=self.lexicon_name, 
                    similarity_strategy=self.similarity_strategy,
                    keep_common_words=self.keep_common_words
                )
        if stage == "predict":
            if self.evaluation_data_split == "dev":
                self.datasets['test'] = BlindGSTDataset(
                    split='dev', 
                    evaluate=True,
                    tokenizer=self.tokenizer, 
                    preprocess_kind=self.preprocess_kind, 
                    dataset_name=self.dataset_name, 
                    lexicon_name=self.lexicon_name, 
                    similarity_strategy=self.similarity_strategy,
                    keep_common_words=self.keep_common_words
                )
            
            else:
                self.datasets['test'] = BlindGSTDataset(
                    split='test', 
                    tokenizer=self.tokenizer, 
                    preprocess_kind=self.preprocess_kind, 
                    dataset_name=self.dataset_name, 
                    lexicon_name=self.lexicon_name, 
                    similarity_strategy=self.similarity_strategy,
                    keep_common_words=self.keep_common_words
                )
    

    def predict_dataloader(self):
        return DataLoader(self.datasets['test'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=False, pin_memory=True)
    
    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=True, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.datasets['dev'], batch_size = self.batch_size, num_workers=os.cpu_count(), shuffle=False, pin_memory=True)


if __name__ == "__main__":
    dm = BlindGSTDM(
        tokenizer_name_or_path='openai-gpt',
        batch_size=32,
        preprocess_kind='bert_best_head',
        evaluation_data_split="dev"
    )

    dm.prepare_data()
    dm.setup(stage="predict")

    loader = dm.val_dataloader()
    for batch in loader:
        print(batch)
        break