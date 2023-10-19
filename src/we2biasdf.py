
import argparse
import logging

import numpy as np
from scipy import sparse

from utils.vocab import load_vocab
from utils.bias import we_bias_df

logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


def main():

    parser = argparse.ArgumentParser()

    required = parser.add_argument_group('required named arguments')
    required.add_argument('--vocab', type=str, required=True)
    required.add_argument('--matrix', type=str, required=True)
    required.add_argument('--a', type=str, required=True)
    required.add_argument('--b', type=str, required=True)
    required.add_argument('--out', type=str, required=True)

    optional = parser.add_argument_group('optional named arguments')
    optional.add_argument('--only_positive', action='store_true')

    logging.info("START")
    args = parser.parse_args()

    logging.info("Loading input data...")
    str2idx, idx2str, str2count = load_vocab(args.vocab)
    if args.matrix.endswith('.npz'):
        embed_matrix = sparse.load_npz(args.matrix)
    else:
        embed_matrix = np.load(args.matrix)

    logging.info("Getting words lists...")
    words_a = [
        line.strip().lower() for line in open(f'words_lists/{args.a}.txt','r')]
    words_b = [
        line.strip().lower() for line in open(f'words_lists/{args.b}.txt','r')]
    words_target = [
        w for w, freq in str2count.items() if w not in words_a + words_b]

    logging.info("Computing WE bias wrt each target word...")
    df_bias = we_bias_df(
        embed_matrix, words_target, words_a, words_b, str2idx, str2count, 
        only_positive=args.only_positive)

    logging.info("Saving results in csv...")
    df_bias.to_csv(args.out, index=False)

    logging.info("DONE")


if __name__ == "__main__":
    main()
