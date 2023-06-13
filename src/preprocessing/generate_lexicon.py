from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json

from datasets import DATA_DIR
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

def sort_coef(coef):
    return {k: v for k, v in sorted(coef.items(), key=lambda item: item[1], reverse=True)}

def read_corpus(filepath):
    logger.info(f"Loading corpus file:\t{filepath}")
    corpus = []
    with open(filepath, 'r') as f:
        corpus += [line.strip() for line in f.readlines()]
    return corpus

if __name__ == "__main__":

    from argparse import  ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--dataset_name", 
                            type=str, 
                            default="paradetox", 
                            choices=["jigsaw", "paradetox"], 
                            help="1. paradetox\t2. jigsaw")
    parser.add_argument("--do_plot", 
                            action="store_true",
                            help="If true it plots distribution of coefficients inside the upper and lower bounds")
    parser.add_argument("--debug", 
                            action="store_true",
                            help="If true it outputs the COMPLETE word-coefficient dictionary instead of the filtered lexicon.")
    parser.add_argument("--upper_bound", 
                            type=float, 
                            help="Upper bound used when plotting distribution")
    parser.add_argument("--lower_bound", 
                            type=float, 
                            help="Lower bound used when plotting distribution")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    logger.info(f"Selected dataset:\t{dataset_name}")
    if dataset_name == "jigsaw":
        threshold = 0.8
        logger.info(f"Using threshold:\t{threshold}")
    if dataset_name == "paradetox":
        threshold = 0.7
        logger.info(f"Using threshold:\t{threshold}")
    
    tox_train_filepath = f"{DATA_DIR}/{dataset_name}/train.toxic"
    neu_train_filepath = f"{DATA_DIR}/{dataset_name}/train.neutral"
    tox_dev_filepath = f"{DATA_DIR}/{dataset_name}/dev.toxic"
    neu_dev_filepath = f"{DATA_DIR}/{dataset_name}/dev.neutral"
    
    out_filepath = f"{DATA_DIR}/lexicons/{dataset_name}.txt"

    tox_train = read_corpus(tox_train_filepath)
    neu_train = read_corpus(neu_train_filepath)
    tox_dev = read_corpus(tox_dev_filepath)
    neu_dev = read_corpus(neu_dev_filepath)

    pipe = Pipeline([
        ('vectorizer', CountVectorizer(stop_words='english', lowercase=True)), 
        ('clf', LogisticRegression(max_iter=1000))
    ], verbose=True)

    X = tox_train + tox_dev + neu_train + neu_dev 
    y = [1] * (len(tox_train) + len(tox_dev)) + [0] * (len(neu_train) + len(neu_dev))
    pipe.fit(X, y)

    # Get coefs
    coefs = pipe[1].coef_[0]
    word2coef = {w: coefs[idx] for w, idx in pipe[0].vocabulary_.items()}
    word2coef = sort_coef(word2coef)

    if args.debug:
        logger.info(f"Writting COMPLETE word->coef dictionary to:\t{out_filepath}")
        with open(out_filepath, "w") as f:
            f.write(json.dumps(word2coef, indent=4))
    else:
        filtered_words = [word for word, coef in word2coef.items() if coef > threshold]
        logger.info(f"Number of words in lexicon:\t{len(filtered_words)} words")
        logger.info(f"Writting lexicon to:\t{out_filepath}")
        with open(out_filepath, "w") as f:
            f.write("\n".join(filtered_words))


    if args.do_plot:
        upper_bound = args.upper_bound if args.upper_bound else max(coefs)
        lower_bound = args.upper_bound if args.upper_bound else min(coefs)
        filtered_coefs = [c for c in coefs if lower_bound < c < upper_bound]
        sns.histplot(filtered_coefs, stat='frequency', element='bars', bins=8)
        plt.title("Proportion Distribution of Lexicon")
        plt.xlabel("Coefficient Value")
        plt.show()