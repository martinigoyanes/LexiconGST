import os
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from datasets import DATA_DIR
import logging
logger = logging.getLogger(__name__)

class BlindGSTDataset(Dataset):
    '''
        Dataset to tokenize and load data for BlindGST model
        Processes data that have been processed to remove detected style tokens.
        The data files have the form:
        <STYLE_LABEL> <CON_START> x1 x2 ... xN <START> y1 y2 ... yN <END>
    '''
    def __init__(
            self, 
            split: str, 
            tokenizer: AutoTokenizer, 
            dataset_name: str,
            preprocess_kind: str,
            lexicon_name: str,
            similarity_strategy: str,
            keep_common_words: bool = False,
            evaluate: bool = False,
        ):
        super().__init__()
        self.split = split
        self.evaluate = evaluate
        self.tokenizer = tokenizer
        if preprocess_kind == "word_embeddings":
            if keep_common_words:
                self.data_dir = f"{DATA_DIR}/{dataset_name}/{preprocess_kind}/keep_common_words/{lexicon_name}/{similarity_strategy}"
            else:
                self.data_dir = f"{DATA_DIR}/{dataset_name}/{preprocess_kind}/remove_common_words/{lexicon_name}/{similarity_strategy}"
        else:
            self.data_dir = f"{DATA_DIR}/{dataset_name}/{preprocess_kind}/"
        self.cache_dir = f'{self.data_dir}/.cache'
        self._setup()

    def _setup(self):

        if self.split == "train":
            cache_file = f"{self.cache_dir}/train.tokenized.pt"
            data_file = f"{self.data_dir}/train"
        
        if self.split == 'dev':
            if self.evaluate:
                cache_file = f"{self.cache_dir}/dev.toxic.in.tokenized.pt"
                data_file = f"{self.data_dir}/dev.toxic.in"
                self.tokenizer.padding_side = "left"
            else:
                cache_file = f"{self.cache_dir}/dev.tokenized.pt"
                data_file = f"{self.data_dir}/dev"

        if self.split == 'test':
            cache_file = f"{self.cache_dir}/test.toxic.in.tokenized.pt"
            data_file = f"{self.data_dir}/test.toxic.in"
            # https://github.com/huggingface/transformers/pull/7552#issue-497255933
            self.tokenizer.padding_side = "left"

        os.makedirs(self.cache_dir, exist_ok=True)
        if not os.path.exists(cache_file): 
            self._prepare_data(
                cache_file=cache_file, 
                data_file=data_file, 
                create_labels=(self.split in ['train', 'dev'] and not self.evaluate)
            )

        if getattr(self, 'data', None) == None:
            logger.info(f"Loading {self.split} tokenization from cache [{cache_file}] ...")
            self.data = torch.load(cache_file)
    
    def _prepare_data(
        self,
        cache_file: str, 
        data_file: str, 
        create_labels: bool = False
    ):

        with open(data_file, 'r') as f:
                texts = [line.strip() for line in f.readlines()]

        logger.info(f"Tokenizing {self.split} data from {data_file}...")
        self.data = self.tokenizer(texts, return_tensors='pt', padding=True)

        if create_labels:
            logger.info(f"Masking {self.split} labels from {data_file}...")
            self.data['labels'] = self._create_labels()         
        else:
            self.data['labels'] = []

        logger.info(f"Saving {self.split} tokenization to cache [{cache_file}] ...")
        with open(cache_file, 'wb') as f:
            torch.save(obj=self.data, f=f)
    
    def _create_labels(self):
        def _get_index(tensor, value):
            return (tensor == value).nonzero()[0]

        size = self.data.input_ids.size()
        labels = torch.full(size=size, fill_value=-100)

        for i, ids in enumerate(self.data.input_ids):
            idx_start = _get_index(ids, self.tokenizer.bos_token_id) 
            idx_end = _get_index(ids, self.tokenizer.eos_token_id) 
            # Input: <STYLE> <CON_START> bla bla bla <START> ble ble ble ble ble <END>
            # Labels: [-100 ... -100] ble ble ble ble ble <END> [-100 ... -100]
            labels[i, idx_start:idx_end] = ids[idx_start+1:idx_end+1] # Dont include <START> but include <END>

        return labels

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, index):
        return {
            'input_ids': self.data['input_ids'][index], 
            'attention_mask': self.data['attention_mask'][index],
            'labels': self.data['labels'][index] if len(self.data['labels']) > 0 else []
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    special_tokens = ['<NEU>', '<TOX>','<CON_START>','<START>','<END>', '<PAD>']
    tokenizer = AutoTokenizer.from_pretrained(
        'openai-gpt', 
        use_fast=True, 
        additional_special_tokens=special_tokens,
        pad_token='<PAD>',
        eos_token='<END>',
        bos_token='<START>'
    )
    dataset = BlindGSTDataset(
        split="dev", 
        tokenizer=tokenizer, 
        dataset_name="paradetox",
        preprocess_kind="word_embeddings",
        lexicon_name="hate",
        similarity_strategy="top-3"
    )
    print(dataset.__getitem__(0))