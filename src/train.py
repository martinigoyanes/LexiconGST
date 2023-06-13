import torch
import pytorch_lightning as pl
import logging
from pytorch_lightning.profilers import SimpleProfiler, AdvancedProfiler
import data_modules
import models
from checkpoint import HfModelCheckpoint

logger = logging.getLogger(__name__)

# TODO: Use all data
    # Just a reminder to use all data when training for real

def main(args):

    # Setup Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    dict_args = vars(args)

    # pick datamodule
    dm = getattr(data_modules, args.datamodule_name)(**dict_args)
    # pick model
    if args.model_name == "BlindGST":
        model = models.BlindGST(
            num_special_tokens=len(dm.special_tokens),
            pad_token_id=dm.tokenizer.pad_token_id, 
            eos_token_id=dm.tokenizer.eos_token_id, 
            **dict_args
        )
    else:
        model = getattr(models, args.model_name)(**dict_args)


    # https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    checkpoint_callback = HfModelCheckpoint()
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        # fast_dev_run=True,
        # deterministic=True,
        # limit_train_batches=10,
        # limit_val_batches=2,
        # max_epochs=1,
        # profiler="advanced",
    )
    trainer.fit(model, datamodule=dm)

    # If model has _save_results means that is prepared to run trainer.test
    if callable(getattr(model, "_save_results", None)):
        logger.info(f"Will test model {args.model_name}")
        trainer.test(model, datamodule=dm)
    else:
        logger.info(f"Will NOT test model {args.model_name}")



if __name__ == "__main__":
    from argparse import  ArgumentParser

    # Resolve handler is key to be able to overrid trainer args with model/datamodule specific args
    parser = ArgumentParser(conflict_handler='resolve')
    parser = pl.Trainer.add_argparse_args(parser)

    # figure out which model to use
    parser.add_argument("--model_name", type=str, default="BlindGST", 
                            choices=["BlindGST", "SentimentRoBERTa"], 
                            help="1. BlindGST\t2. SentimentRoBERTa")
    parser.add_argument("--datamodule_name", type=str, default="BlindGSTDM", 
                        choices=["BlindGSTDM"], 
                        help="1. BlindGSTDM")
    parser.add_argument("--default_root_dir", type=str, default=".", help="Directory to store run logs and ckpts")


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