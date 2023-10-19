
import argparse
import logging
import pickle
from pathlib import Path
from tqdm import tqdm


logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)


def main():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("corpus_file", help="path to corpus file")
    parser.add_argument("word_a", help="word A to count")
    parser.add_argument("word_b", help="word B to count")
    
    args = parser.parse_args()

    logging.info("Running")
    counts = counts_dict(args.corpus_file, args.word_a, args.word_b)

    logging.info("Saving as pickle")
    corpus_name = Path(args.corpus_file).stem
    outfile = f"data/working/{corpus_name}_counts_{args.word_a}-{args.word_b}.pkl"
    with open(outfile, 'wb') as f:
        pickle.dump(counts, f, protocol=pickle.HIGHEST_PROTOCOL)


def counts_dict(corpus_file: str, word_a: str, word_b: str) -> dict:
    """
    Read each doc from corpus.txt file and count appearances of word_a, word_b
    Returns:
        dict {i_doc: (count_a, count_b)}
    """
    
    logging.info("Computing total number of docs")
    with open(corpus_file) as f:
        n_docs = sum(1 for _ in f)
    
    logging.info("Computing counts for each doc")
    i = 0
    counts = dict()
    pbar = tqdm(total=n_docs)
    with open(corpus_file, "r", encoding="utf-8") as f:
        while True:
            tokens = f.readline().strip().split()
            a_count, b_count = tokens.count(word_a), tokens.count(word_b)
            counts[i] = (a_count, b_count)
            i += 1
            pbar.update(1)
            if not tokens:
                break
        pbar.close()
    return counts


if __name__ == "__main__":
    main()
