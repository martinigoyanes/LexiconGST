import os
import csv
import logging 
from collections import defaultdict

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)

# ROOT_PATH = "/Midgard/home/martinig/thesis-src"
ROOT_PATH = "/home/martin/Documents/Education/Master/thesis/project/thesis-src"

MODEL_NAME = "BlindGST"
DATA_SPLIT = "dev"

def nested_dict():
    return defaultdict(nested_dict)

if __name__ == "__main__":

    from argparse import  ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--root_path", 
                            type=str, 
                            default=ROOT_PATH)
    parser.add_argument("--model_name", 
                            type=str, 
                            default=MODEL_NAME)
    parser.add_argument("--data_split", 
                            type=str,
                            help="dev or test", 
                            default=DATA_SPLIT)

    args = parser.parse_args()

    jobs_path = f"{args.root_path}/jobs"
    out_path = f"{args.root_path}/results/{args.data_split}"
    model_name = args.model_name

    results_path = f"{jobs_path}/{model_name}"
    logger.info(f"Collecting results from:\t{results_path}")
    logger.info(f"Outputing results into:\t{out_path}")
    
    results = nested_dict()

    # Read results from all dev_results.md or (test)_results.md files for all jobs run
    for path, dirs, files in os.walk(results_path):
        for filename in files:
            if filename == f"{args.data_split}_results.md":
                path_folders = path.split("/")
                filepath = os.path.join(path, filename) 
                with open(filepath, "r") as f:
                    lines = f.readlines()
                    result_values = lines[-1].strip().split("|")

                if path_folders[-1] == "bert_best_head":
                    dataset = path_folders[-2]
                    submodel = "bert_best_head"
                    outfile_path = f"{out_path}/{dataset}/bert_best_head.csv"
                    results[dataset][submodel] = {
                        "ACC": result_values[0], 
                        "SIM": result_values[1],
                        "FL": result_values[2],
                        "J": result_values[3],
                        "BLEU": result_values[4]
                    }
                    logger.info(f"Read results from:\t{dataset}/{submodel}")

                if "top-" in path_folders[-1]:
                    sim_strategy = path_folders[-1]
                    lexicon = path_folders[-2]
                    common_words_dir = path_folders[-3]
                    dataset = path_folders[-5]
                    submodel = "word_embeddings"
                    results[dataset][submodel][common_words_dir][lexicon][sim_strategy] = {
                        "ACC": result_values[0], 
                        "SIM": result_values[1],
                        "FL": result_values[2],
                        "J": result_values[3],
                        "BLEU": result_values[4]
                    }

                    logger.info(f"Read results from:\t{dataset}/{submodel}/{common_words_dir}/{lexicon}/{sim_strategy}")

    # Aggregate results into csv files
    for dataset in results.keys():
        dataset_dir = f"{out_path}/{dataset}" 
        os.makedirs(dataset_dir, exist_ok=True)
        for submodel in results[dataset].keys():
            if submodel == "word_embeddings":
                # jigsaw/word_embeddings/remove_common_words/abusive/top-1/results.md
                submodel_dir = f"{dataset_dir}/{submodel}" 
                os.makedirs(submodel_dir, exist_ok=True)
           
                for common_words_model in results[dataset][submodel].keys():
                    out_filepath = f"{dataset_dir}/{submodel}/{common_words_model}.csv"
                    
                    logger.info(f"Writting:\t{out_filepath}")
                    with open(out_filepath, "w") as f:
                        writer = csv.writer(f)
                        writer.writerow(["Lexicon", "K", "ACC", "SIM", "FL", "J", "BLEU"])
                        
                        for lexicon in sorted(results[dataset][submodel][common_words_model].keys()):
                            for sim_strategy in sorted(results[dataset][submodel][common_words_model][lexicon].keys()):
                                k = sim_strategy[-1] # top-9, top-7, ...
                                metrics = results[dataset][submodel][common_words_model][lexicon][sim_strategy].values()
                                writer.writerow([lexicon] + [k] + list(metrics))

            if submodel == "bert_best_head":
                # jigsaw/bert_best_head/results.md
                out_filepath = f"{dataset_dir}/{submodel}.csv"
                logger.info(f"Writting:\t{out_filepath}")
                with open(out_filepath, "w") as f:
                    writer = csv.writer(f)
                    writer.writerow(["ACC", "SIM", "FL", "J", "BLEU"])
                    writer.writerow(results[dataset][submodel].values())
            