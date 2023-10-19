
import argparse
import logging
from pathlib import Path

import numpy as np
from gensim.models.fasttext import FastText

from utils.vocab import load_vocab


# we do this to prevent gensim from printing too much
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
logging.getLogger('gensim').setLevel(logging.WARNING)


def main():
    """Train, save model and save embeddings matrix
    """
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required named arguments')
    required.add_argument('--corpus', type=str, required=True)
    required.add_argument('--vocab', type=str, required=True)
    optional = parser.add_argument_group('optional named arguments')
    optional.add_argument('--size', type=int, required=False, default=300)
    optional.add_argument('--window', type=int, required=False, default=10)
    optional.add_argument('--count', type=int, required=False, default=100)
    optional.add_argument('--sg', type=int, required=False, default=1)
    optional.add_argument('--ns', type=int, required=False, default=5)
    optional.add_argument('--ns_exponent', type=float, required=False, default=0.75)
    optional.add_argument('--seed', type=int, required=False, default=1)
    optional.add_argument(
        '--outdir', type=str, required=False, nargs='?', const='', default="")
    args = parser.parse_args()
    kwargs_model = {
        'size': args.size, 'window': args.window, 'min_count': args.count,
        'ns': args.ns, 'ns_exponent': args.ns_exponent, 'sg': args.sg, 
        'seed': args.seed}

    logging.info("START")

    logging.info("Loading vocab")
    str2idx, idx2str, str2count = load_vocab(args.vocab)

    # model file name
    kw = kwargs_model
    corpus_name = Path(args.corpus).stem
    basename = \
        f"ft-{corpus_name}-V{kw['min_count']}-W{kw['window']}-D{kw['size']}-" + \
        f"SG{kw['sg']}-S{kw['seed']}-NS{kw['ns']}-NSE{kw['ns_exponent']}"
    model_file = str(Path(args.outdir) / "data/working" / f"{basename}.model")

    if Path(model_file).is_file():
        logging.info(f"Model file {model_file} exists. Skipping training.")
        model = FastText.load(model_file)
    else:
        logging.info("Training vectors with params:")
        print(*kwargs_model.items(), sep="\n")
        model = train_fasttext(args.corpus, **kwargs_model)
        logging.info("Saving model...")
        model.save(model_file)

    logging.info("Testing vocabulary...")
    str2count_model = {w: model.wv.get_vecattr(w, "count") for w in model.wv.key_to_index}
    assert str2count_model == str2count, \
        "gensim vocab is different from input vocab file"

    logging.info("Converting vectors to array...")
    w_matrix, wc_matrix  = gensim_to_arrays(model, str2idx)

    logging.info("Saving arrays...")
    w_file = str(Path(args.outdir) / "data/working" / f"{basename}.npy")
    wc_file = str(Path(args.outdir) / "data/working" / f"{basename}-WC0.npy")
    np.save(w_file, w_matrix)
    np.save(wc_file, wc_matrix)

    logging.info("DONE")


class Corpus:
    """
    Helper iterator that yields documents/sentences (doc: list of str)
    Needed so that gensim can read docs from disk on the fly
    """
    def __init__(self, corpus_file):
        """
        corpus_file: a txt with one document per line and tokens separated
        by whitespace
        """
        self.corpus_file = corpus_file
    def __iter__(self):
        for line in open(self.corpus_file, encoding="utf-8"):
            # returns each doc as a list of tokens
            yield line.split()


def train_fasttext(corpus_file, size, window, ns, ns_exponent, min_count, seed, sg):
    """
    Returns fasttext gensim trained model
    Params:
        - min_count: min word frequency
        - sg: 1 if skipgram -- 0 if cbow
        - ns: number of negative samples
        - ns_exponent: exponent for negative sampling ("alpha")
    """
    # create generator of lines
    sentences = Corpus(corpus_file)
    # train
    model = FastText(
        sentences=sentences, vector_size=size, window=window, negative=ns, 
        ns_exponent=ns_exponent, min_count=min_count, seed=seed, sg=sg, workers=9)
    return model


def gensim_to_arrays(model, str2idx):
    """
    Convert gensim vectors to np.arrays DIMx(V+1), using column indices
    of str2idx of vocab_file produced by GloVe

    Returns:
        - word vectors matrix (w): np.array of shape (D, V+1)
        - context vectors matrix + W (c+w): np.array of shape (D, V+1)
    """
    word_vectors = model.wv
    context_vectors = model.syn1neg # embeddings are in rows
    D = model.wv.vector_size
    V = len(str2idx)
    W = np.full((D, V+1), np.nan)
    C = np.full((D, V+1), np.nan)
    for w, i in str2idx.items():
        W[:,i] = word_vectors[w]
        C[:,i] = context_vectors[i-1] # assumes idx starts at 1
    return W, W+C


if __name__ == "__main__":
    main()
