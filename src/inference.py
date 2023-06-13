import torch
import logging
import pytorch_lightning as pl
import data_modules
import models

logger = logging.getLogger(__name__)


def main(args):

    # Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    dict_args = vars(args)

    # pick datamodule
    dm = getattr(data_modules, args.datamodule_name).load_from_checkpoint(args.checkpoint_path, evaluation_data_split=args.evaluation_data_split, dataset_name=args.dataset_name)
    # pick model
    model = getattr(models, args.model_name).load_from_checkpoint(args.checkpoint_path)

    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    trainer = pl.Trainer.from_argparse_args(
        args,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        # fast_dev_run=True,
        # deterministic=True,
        # limit_train_batches=10,
        # limit_val_batches=10,
        # limit_test_batches=1,
        # limit_predict_batches=1,
        # profiler="advanced",
    )
    preds = trainer.predict(model=model, datamodule=dm)

    model.save_preds(preds=preds, tokenizer=dm.tokenizer, evaluation_data_split=dm.evaluation_data_split)

if __name__ == "__main__":
    from argparse import  ArgumentParser

    # Resolve handler is key to be able to overrid trainer args with model/datamodule specific args
    parser = ArgumentParser(conflict_handler='resolve')
    parser = pl.Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument("--model_name", type=str, default="BlindGST", 
                            choices=["BlindGST", "SentimentRoBERTa"], 
                            help="1. BlindGST\t2. SentimentRoBERTa")
    parser.add_argument("--model_name", type=str, default="BlindGST", 
                            choices=["BlindGST", "SentimentRoBERTa"], 
                            help="1. BlindGST\t2. SentimentRoBERTa")
    parser.add_argument("--datamodule_name", type=str, default="BlindGSTDM", 
                        choices=["BlindGSTDM"], 
                        help="1. BlindGSTDM")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint from trainer.fit()")
    parser.add_argument("--evaluation_data_split", type=str, default="test", required=True, help="Choose which split of data use to predict on, either test or dev (validation)")

    # THIS LINE IS KEY TO PULL THE MODEL NAME
    temp_args, _ = parser.parse_known_args()

    # let the model add what it wants
    model = getattr(models, temp_args.model_name)
    parser = model.add_specific_args(parser)

    # let the datamodule add what it wants
    dm = getattr(data_modules, temp_args.datamodule_name)
    parser = dm.add_specific_args(parser)

    args = parser.parse_args()

    # train
    main(args)