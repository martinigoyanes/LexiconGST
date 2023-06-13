from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
import json
import os
from datasets import DATA_DIR
import logging 
import time
from multiprocessing import Process, Manager

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)


def read_toxic_words(filename):
    with open(filename, 'r', encoding="ISO-8859-1") as f:
        toxic_words = [line.strip() for line in f.readlines()]
    return toxic_words

def toxic_words_to_word_set(words):
    return set(words)

def read_corpus(file_list):
    corpus = []
    for file in file_list:
        with open(file, 'r') as f:
            corpus += [line.strip() for line in f.readlines()]
    return corpus

def corpus_to_word_set(corpus):
    corpus_words_set = set()
    for sentence in corpus:
        corpus_words_set.update(sentence.split(" "))
    return corpus_words_set

def get_embedding(embed_table, token_ids):
    # We define embeddings of a words as teh average of the embeddings from the tokens that compose the word
    return torch.mean(embed_table[token_ids], dim=0)

def get_toxic_word_embed_list(embed_table, toxic_words_set, tokenizer):
    toxic_word_ids = tokenizer.batch_encode_plus(toxic_words_set, add_special_tokens=False)

    n_words, embed_dim = len(toxic_words_set), embed_table.size(1)
    toxic_word_embed_list = torch.zeros(size=(n_words, embed_dim), dtype=torch.float32)
    for j,word_ids in enumerate(toxic_word_ids['input_ids']):
        toxic_word_embed = get_embedding(embed_table=embed_table, token_ids=word_ids)
        toxic_word_embed_list[j] = torch.FloatTensor(toxic_word_embed)

    return toxic_word_embed_list
    
def sort_similarities(similarities):
    # Sorts from most similar to toxic words to least similar	
    return {k: v for k, v in sorted(similarities.items(), key=lambda item: item[1], reverse=True)}

def async_compute_similarities(similarities, strategy, embed_table, toxic_words_embed_list, corpus_words_set, sim_fn, tokenizer):
    pid = str(os.getpid())[-2:]
    logger.info(f"{pid}:\tREADY")

    for i,corpus_word in enumerate(corpus_words_set):
        # Thread/processs skips if this similarity has been computed by some other thread
        if similarities.get(corpus_word, None) is None:
            similarities[corpus_word] = -100
            corpus_word_ids = tokenizer.encode(corpus_word, add_special_tokens=False)
            corpus_word_embed = get_embedding(embed_table=embed_table, token_ids=corpus_word_ids)
            if all(corpus_word_embed.isnan()):
                corpus_toxic_sim = torch.zeros(size=(1,), dtype=torch.float32)
            else:
                corpus_toxic_sim = sim_fn(corpus_word_embed, toxic_words_embed_list)

            sorted_corpus_toxic_sim, indices = torch.sort(corpus_toxic_sim, descending=True)
            if strategy == "top-1":
                sorted_corpus_toxic_sim = sorted_corpus_toxic_sim[:1]
            if strategy == "top-3":
                sorted_corpus_toxic_sim = sorted_corpus_toxic_sim[:3]
            if strategy == "top-5":
                sorted_corpus_toxic_sim = sorted_corpus_toxic_sim[:5]
            if strategy == "top-7":
                sorted_corpus_toxic_sim = sorted_corpus_toxic_sim[:7]
            if strategy == "top-9":
                sorted_corpus_toxic_sim = sorted_corpus_toxic_sim[:9]

            similarities[corpus_word] = torch.mean(sorted_corpus_toxic_sim).item()

            logger.info(f"{pid}:\tCalculated {(i+1)*toxic_words_embed_list.size(0)} similarities")


if __name__ == "__main__":
    # 0. Toxic dictionary
    # 1. Get embeddings from model.model.roberta.embeddings.word_embeddings.weight[idx]
    # 2. Load corpus
    # 3. Store embeddings from corpus and embeddings from toxic dictionary
    # 3. Compute similarity between word_embedding_i in corpus and word_embedding_j in toxic dictionary 
    # 4. Store matrix

    from argparse import  ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--lexicon_name", 
                            type=str, 
                            default="abusive", 
                            choices=["toxic", "ngram-refined", "abusive", "hate", "jigsaw", "paradetox"], 
                            help="1. toxic\t2. ngram-refined\t3. abusive\t4. hate\t5. paradetox\t6. jigsaw")
    parser.add_argument("--similarity_strategy", 
                            type=str, 
                            default="top-3", 
                            choices=["top-1", "top-3", "top-5", "top-7", "top-9"], 
                            help="1. top-1\t2. top-3\t3. top-5.\t4. top-7\t5.top-9")
    parser.add_argument("--dataset_name", 
                            type=str, 
                            default="paradetox", 
                            choices=["jigsaw", "paradetox"], 
                            help="1. paradetox\t2. jigsaw")
    args = parser.parse_args()

    dataset_dir = f"{DATA_DIR}/{args.dataset_name}"
    out_dir = f"{dataset_dir}/word_embeddings/similarities"
    out_file = f"{out_dir}/{args.lexicon_name}_{args.similarity_strategy}.pt"
    lexicon_f = f"{DATA_DIR}/lexicons/{args.lexicon_name}.txt"
    corpus_files = [
        f"{dataset_dir}/dev.neutral",
        f"{dataset_dir}/dev.toxic",
        f"{dataset_dir}/test.neutral",
        f"{dataset_dir}/test.toxic",
        f"{dataset_dir}/train.neutral",
        f"{dataset_dir}/train.toxic",
    ]
    logger.info(f"Loading corpus files: {str(corpus_files)}")
    logger.info(f"Loading toxic words from lexicon:\t{lexicon_f}")
    logger.info(f"Saving similarity matrix to:\t{out_file}")

    toxic_words = read_toxic_words(lexicon_f)
    logger.info(f"Len toxic_words = {len(toxic_words)} words/phrases")
    toxic_words_set = toxic_words_to_word_set(toxic_words)
    logger.info(f"Len toxic_words = {len(toxic_words_set)} unique words/phrases")

    corpus = read_corpus(corpus_files)
    logger.info(f"Len corpus = {len(corpus)} sentences")
    corpus_words_set = corpus_to_word_set(corpus)
    logger.info(f"Len corpus = {len(corpus_words_set)} unique words")

    if not os.path.exists(out_file):
        os.makedirs(out_dir, exist_ok=True)

        tokenizer = RobertaTokenizer.from_pretrained('s-nlp/roberta_toxicity_classifier')
        model = RobertaForSequenceClassification.from_pretrained('s-nlp/roberta_toxicity_classifier')

        embed_table = model.roberta.embeddings.word_embeddings.weight
        sim_fn = torch.nn.CosineSimilarity(dim=1)

        toxic_word_embed_list = get_toxic_word_embed_list(
            embed_table=embed_table, 
            toxic_words_set=toxic_words_set, 
            tokenizer=tokenizer
        )

        with Manager() as manager:
            torch.set_num_threads(1)
            similarities = manager.dict()
            now = time.time()
            processes = [
                Process(
                    target=async_compute_similarities, 
                    args=(similarities, args.similarity_strategy, embed_table, toxic_word_embed_list, corpus_words_set, sim_fn, tokenizer)
                )
                for _ in range(os.cpu_count())
            ]
            
            for p in processes:  p.start()
            for p in processes:  p.join()

            logger.info(f"Took: {time.time() - now} s")

            torch.save(obj=dict(similarities), f=out_file)

            logger.info('#'*50)
            logger.info('\tSorted Similarities')
            logger.info('#'*50)
            logger.info(json.dumps(sort_similarities(similarities), indent=4))
    else:
        logger.info(f"{out_file} already exists")
