import torch
import os
from datasets import DATA_DIR
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def preprocess_split(split, similarities, style, threshold, common_words):
    assert isinstance(common_words, list)

    def remove_toxic_word_in_sentence(sentence):
        # logger.info("#"*10)
        out = f"<{style}> <CON_START>"
        for w in sentence.split(" "):
            try:
                if similarities[w] < threshold:
                    out += f" {w}"
                if similarities[w] >= threshold and w in common_words:
                    # If words is above threshold but is a common word we keep it
                    # logger.info(f"Common word:\t{w} - sim:\t{similarities[w]} - threshold:\t{threshold}")
                    out += f" {w}"
                if similarities[w] >= threshold:
                    continue
            except KeyError:
                out += f" {w}"
        out += f" <START> {sentence} <END>"
        # logger.info(f"Processed sentence:\t{out}")
        return out

    preprocessed_sentences = []
    for sentence in split:
        preprocessed_sentences += [remove_toxic_word_in_sentence(sentence=sentence)]

    return preprocessed_sentences



if __name__ == "__main__":

    # 1. Go through corpus and discard tokens if similarity > threshold (define a threshold)
    # 2. Store results as: <STYLE> <CON_START> x1 x2 ... xN <START> y1 y2 ... yN <END>

    from argparse import  ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--lexicon_name", 
                            type=str, 
                            default="abusive", 
                            choices=["toxic", "ngram-refined", "abusive", "hate", "paradetox", "jigsaw"], 
                            required=True,
                            help="1. toxic\t2. ngram-refined\t3. abusive\t4. hate\t5. paradetox\t6. jigsaw")
    parser.add_argument("--similarity_strategy", 
                            type=str, 
                            default="top-1", 
                            required=True,
                            choices=["top-1", "top-3", "top-5", "top-7", "top-9"], 
                            help="1. top-1\t2. top-3\t3. top-5.\t4. top-7\t5.top-9")
    parser.add_argument("--dataset_name", 
                            type=str, 
                            default="jigsaw", 
                            choices=["jigsaw", "paradetox"], 
                            required=True,
                            help="1. paradetox\t2. jigsaw")
    parser.add_argument("--keep_common_words", 
                            action="store_true",
                            help="If present it does NOT remove words if they are common even if they are above $threshold.")
    parser.add_argument("--threshold", type=float, required=True)

    args = parser.parse_args()

    dataset_dir = f"{DATA_DIR}/{args.dataset_name}"

    logger.info(f"Keeping_common_words:\t{args.keep_common_words}")
    logger.info(f"Threshold:\t{args.threshold}")

    common_words = []
    if args.keep_common_words:
        out_dir = f"{dataset_dir}/word_embeddings/keep_common_words/{args.lexicon_name}/{args.similarity_strategy}"
        common_words_f = f"{DATA_DIR}/lexicons/common_words.txt"
        with open(common_words_f, "r") as f:
            common_words = f.read().split("\n")
    else:
        out_dir = f"{dataset_dir}/word_embeddings/remove_common_words/{args.lexicon_name}/{args.similarity_strategy}"
    logger.info(f"Output directory:\t{out_dir}")

    corpus_files = [
        f"{dataset_dir}/dev.toxic",
        f"{dataset_dir}/test.toxic",
        f"{dataset_dir}/train.toxic",
    ]
    style_labels = {
        'TOXIC': 'TOX',
    }
    similarity_f = f'{dataset_dir}/word_embeddings/similarities/{args.lexicon_name}_{args.similarity_strategy}.pt'

    similarities = torch.load(similarity_f)
    logger.info(f"Loading similarity matrix from:\t{similarity_f}")
    logger.info(f"Loading corpus files:\t{corpus_files}")


    for split_f in corpus_files:
        style = split_f.split(".")[1]
        label = style_labels[style.upper()]
        
        out_file = f"{out_dir}/{split_f.split('/')[-1]}"

        logger.info(f"Processing file {out_file}")
        with open(split_f, 'r') as f:
            split = [line.strip() for line in f.readlines()]
        sentences = preprocess_split(split=split, similarities=similarities, style=label, threshold=args.threshold, common_words=common_words)

        os.makedirs(out_dir, exist_ok=True)
        with open(out_file, 'w') as f:
            f.write("\n".join(sentences))	
