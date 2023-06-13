import os
import torch
import logging 
from datasets import DATA_DIR

logger = logging.getLogger(__name__)

def generate_evaluation_input(filepath, out_dir, tgt_style):
    '''
    Takes in test file containing <STYLE> <CON_START> .... <START> ... <END>
    and generates a test file with only the input to the model: <DESIRED_STYLE> <CON_START> ... <START> 
    '''
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    # get source style and data split
    split, src_style = os.path.split(filepath)[-1].split(".")
    ref_in_f = f'{out_dir}/{split}.{src_style}.in'

    if os.path.exists(ref_in_f):
        print(f"Files already exist:\n\t-{ref_in_f}")

    for line in lines:
        in_text = []

        tokens = line.split()
        start_token_idx = tokens.index('<START>')

        in_text = tokens[:start_token_idx+1] # Include <START>	
        in_text[0] = f"<{tgt_style}>"

        with open(ref_in_f, 'a') as f:
            in_text = " ".join(in_text) + "\n"
            f.write(in_text)


if __name__ == "__main__":
    from argparse import  ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--filepath", type=str, required=True)
    parser.add_argument("--tgt_style", type=str, required=True)
    args = parser.parse_args()

    generate_evaluation_input(filepath=args.filepath, out_dir=args.out_dir, tgt_style=args.tgt_style)