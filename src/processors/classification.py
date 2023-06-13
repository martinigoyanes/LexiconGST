import torch
import logging
logger = logging.getLogger(__name__)

class ClassificationProcessor():
	'''
		Processes data that have been processed by BERT to remove detected style tokens.
		The data has the form:
		<STYLE_LABEL> <CON_START> x1 x2 ... xN <START> y1 y2 ... yN <END>
	'''
	def _prepare_data(
		self,
		cache_file: str, 
		data_file_0: str,
		data_file_1: str, 
		create_labels: bool = False
	):
		with open(data_file_0, 'r') as f:
				texts_0 = [line.strip() for line in f.readlines()]
		with open(data_file_1, 'r') as f:
				texts_1 = [line.strip() for line in f.readlines()]

		logger.info(f"Tokenizing {self.split} data from {data_file_0} and {data_file_1} ...")
		texts = texts_0 + texts_1
		self.data = self.tokenizer(texts, return_tensors='pt', padding=True)

		if create_labels:
			logger.info(f"Masking {self.split} labels from {data_file_0} and {data_file_1} ...")
			self.data['labels'] = self._create_labels(num_samples_0=len(texts_0), num_samples_1=len(texts_1))
		else:
			self.data['labels'] = []

		logger.info(f"Saving {self.split} tokenization to cache [{cache_file}] ...")
		with open(cache_file, 'wb') as f:
			torch.save(obj=self.data, f=f)

	
	def _create_labels(
		self,
		num_samples_0: int, 
		num_samples_1: int, 
	):
		labels_0 = torch.zeros(size=(num_samples_0,), dtype= torch.long)
		labels_1 = torch.full(size=(num_samples_1,), fill_value=1, dtype=torch.long)
		labels = torch.cat((labels_0, labels_1))
		return labels