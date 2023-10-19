import numpy as np
import argparse
import logging

from pathlib import Path
from gensim.models.word2vec import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from utils.vocab import load_vocab


# we do this to prevent gensim from printing too much
logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)
logging.getLogger('gensim').setLevel(logging.WARNING)


def main():
    """Train w2v, save w2v model and save embeddings matrix
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
    optional.add_argument('--save_epochs', action='store_true')
    args = parser.parse_args()
    kwargs_w2v = {
        'size': args.size, 'window': args.window, 'min_count': args.count,
        'ns': args.ns, 'ns_exponent': args.ns_exponent, 'sg': args.sg, 
        'seed': args.seed, 'save_epochs': args.save_epochs}

    logging.info("START")

    logging.info("Loading vocab")
    str2idx, idx2str, str2count = load_vocab(args.vocab)

    # model file name
    kw = kwargs_w2v
    corpus_name = Path(args.corpus).stem
    basename = \
        f"w2v-{corpus_name}-V{kw['min_count']}-W{kw['window']}-D{kw['size']}-" + \
        f"SG{kw['sg']}-S{kw['seed']}-NS{kw['ns']}-NSE{kw['ns_exponent']}"
    model_file = str(Path(args.outdir) / "data/working" / f"{basename}.model")
    kwargs_w2v["model_file"] = model_file

    if Path(model_file).is_file():
        logging.info(f"Model file {model_file} exists. Skipping training.")
        model = Word2Vec.load(model_file)
    else:
        logging.info("Training vectors with params:")
        print(*kwargs_w2v.items(), sep="\n")
        model = train_w2v(args.corpus, **kwargs_w2v)
        logging.info("Saving model...")
        model.save(model_file)

    logging.info("Testing vocabulary...")
    str2count_w2v = {w: model.wv.get_vecattr(w, "count") for w in model.wv.key_to_index}
    assert str2count_w2v == str2count, \
        "gensim Word2Vec vocab is different from input vocab file"

    logging.info("Converting vectors to arrays...")
    w_matrix, wc_matrix  = w2v_to_arrays(model, str2idx)

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


class EpochSaver(CallbackAny2Vec):
    """Callback to save model after each epoch."""

    def __init__(self, path_prefix):
        self.path_prefix = path_prefix
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(f"Epoch #{self.epoch} start")
        if self.epoch == 0:
            output_path = f"{self.path_prefix}-epoch{self.epoch}.model"
            model.save(output_path)
            print(f"Epoch #{self.epoch} model saved")

    def on_epoch_end(self, model):
        self.epoch += 1
        output_path = f"{self.path_prefix}-epoch{self.epoch}.model"
        model.save(output_path)
        print(f"Epoch #{self.epoch} model saved")


def train_w2v(
    corpus_file, size, window, ns, ns_exponent, min_count, seed, sg, 
    save_epochs, model_file
):
    """
    Returns w2v gensim trained model
    Params:
        - min_count: min word frequency
        - sg: 1 if skipgram -- 0 if cbow
        - ns: number of negative samples
        - ns_exponent: exponent for negative sampling ("alpha")
    """
    # create generator of lines
    sentences = Corpus(corpus_file)
    # save model after each epoch?
    save_prefix = str(Path(model_file).with_suffix(''))
    epoch_saver = EpochSaver(save_prefix) if save_epochs else None
    # train word2vec
    model = Word2Vec(
        sentences=sentences, vector_size=size, window=window, negative=ns, 
        ns_exponent=ns_exponent, min_count=min_count, seed=seed, sg=sg, 
        workers=9, callbacks=[epoch_saver])
    return model


def w2v_to_arrays(w2v_model, str2idx):
    """
    Convert w2v vectors to np.arrays DIMx(V+1), using column indices
    of str2idx of vocab_file produced by GloVe

    Returns:
        - word vectors matrix (w): np.array of shape (D, V+1)
        - context vectors matrix + W (c+w): np.array of shape (D, V+1)
    """
    word_vectors = w2v_model.wv
    context_vectors = w2v_model.syn1neg # embeddings are in rows
    D = w2v_model.wv.vector_size
    V = len(str2idx)
    W = np.full((D, V+1), np.nan)
    C = np.full((D, V+1), np.nan)
    for w, i in str2idx.items():
        W[:,i] = word_vectors[w]
        C[:,i] = context_vectors[i-1] # assumes idx starts at 1
    return W, W+C


if __name__ == "__main__":
    main()
