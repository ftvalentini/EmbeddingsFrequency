
import numpy as np
import argparse
import datetime

from tqdm import tqdm


def write_to_file(words, out_file):
    """Write list of words to file
    """
    while words:
        w = words.pop()
        out_file.write(w + " ")


def main(corpus_file, out_file, seed):
    
    print("Counting total number of sentences...")
    n_lines = sum(1 for line in open(corpus_file, 'r', encoding="utf-8"))
    # checkpoints to read/shuffle/write words
    partes = 10
    cutpoints = np.linspace(0, n_lines, num=partes+1, dtype=int, endpoint=True)
    cutpoints = cutpoints[1:]
    np.random.seed(seed)
    k = 0
    with open(corpus_file, encoding="utf-8") as file_in,\
         open(out_file, mode="w", encoding="utf-8") as file_out:
        for corte in tqdm(cutpoints):
            print(f"\nReading words...")
            words = list()
            for line in file_in:
                words_line = line.split()
                words.extend(words_line + ['\n']) # adds one newline per sentence
                k += 1
                if k == corte:
                    print(f"Shuffling {len(words)} words...")
                    np.random.shuffle(words)
                    print(f"Writing {len(words)} words...")
                    write_to_file(words, file_out)
                    # go to next checkpoint
                    break


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('corpusfile', type=str)
    parser.add_argument('outfile', type=str)
    parser.add_argument('seed', type=int)

    args = parser.parse_args()

    print("START", datetime.datetime.now())
    main(args.corpusfile, args.outfile, args.seed)
    print("--- END ---", datetime.datetime.now())
