from pytorch_lightning.callbacks import ModelCheckpoint
from fsspec.core import url_to_fs
import pytorch_lightning as pl
import logging 

logger = logging.getLogger(__name__)


class HfModelCheckpoint(ModelCheckpoint):
	def _save_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
		super()._save_checkpoint(trainer, filepath)
		model = trainer.lightning_module.model
		tokenizer = trainer.datamodule.tokenizer
		if getattr(model, "save_pretrained", None) and getattr(tokenizer, "save_pretrained", None):
			dir_name = "/".join(filepath.split('/')[:-1])
			logger.info("#"*10)
			logger.info(f"\tSaving model and tokenizer in {dir_name}...")
			logger.info("#"*10)
			model_path = f"{dir_name}/model"
			tokenizer_path = f"{dir_name}/tokenizer"
			if trainer.is_global_zero:
				model.save_pretrained(model_path)
				tokenizer.save_pretrained(tokenizer_path)
	
	# https://github.com/Lightning-AI/lightning/pull/16067
	def _remove_checkpoint(self, trainer: "pl.Trainer", filepath: str) -> None:
		super()._remove_checkpoint(trainer, filepath)
		hf_save_dir = filepath+".dir"
		model = trainer.lightning_module.model
		tokenizer = trainer.datamodule.tokenizer
		if getattr(model, "save_pretrained", None) and getattr(tokenizer, "save_pretrained", None):
			dir_name = "/".join(filepath.split('/')[:-1])
			logger.info("#"*10)
			logger.info(f"\tDeleting model and tokenizer in {dir_name}...")
			logger.info("#"*10)
			model_path = f"{dir_name}/model"
			tokenizer_path = f"{dir_name}/tokenizer"
			if trainer.is_global_zero:
				fs, _ = url_to_fs(model_path)
				if fs.exists(model_path):
					fs.rm(model_path, recursive=True)

				fs, _ = url_to_fs(tokenizer_path)
				if fs.exists(tokenizer_path):
					fs.rm(tokenizer_path, recursive=True)
