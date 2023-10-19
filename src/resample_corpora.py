"""
Add or remove documents with context word B so that their frequency in the
new corpora is [10**4, 10**5, 10**6]
"""

import pickle
import argparse
import logging
from pathlib import Path
from contextlib import ExitStack
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.vocab import load_vocab


logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_file', type=str)
    parser.add_argument('vocab_file', type=str)
    parser.add_argument('word_a', type=str)
    parser.add_argument('word_b', type=str)
    parser.add_argument('outdir', type=str)
    parser.add_argument('seed', type=int)

    args = parser.parse_args()

    logging.info("Loading vocab")
    str2idx, idx2str, str2count = load_vocab(args.vocab_file)
    
    logging.info("Loading contexts counts")
    corpus_name = Path(args.corpus_file).stem
    counts_file = f"data/working/{corpus_name}_counts_{args.word_a}-{args.word_b}.pkl"
    with open(counts_file, "rb") as f:
        counts = pickle.load(f)

    freq_a = str2count[args.word_a]
    freq_b = str2count[args.word_b]

    # target log10 frequencies
    target_log10_freqs = [4, 5, 6]
    log10_freq_b = np.log10(freq_b)

    # frequencies that require undersampling B
    log10_freqs_undersampling = [f for f in target_log10_freqs if f <= log10_freq_b]
    
    # frequencies that require oversampling B
    log10_freqs_oversampling = [f for f in target_log10_freqs if f > log10_freq_b]

    logging.info("Getting list of target documents to sample from")
    df = pd.DataFrame({'doc': list(counts.keys()), 'counts': list(counts.values())})
    df['freq_a'] = [v[0] for v in df['counts']]
    df['freq_b'] = [v[1] for v in df['counts']]
    # docs that can be resampled 
    df_target_docs = df.query("freq_b > 0").copy()
    # NOTE nos gustaria eliminar docs con B sin A para que la freq de A no se modifique
    # sin embargo esto no permitiria eliminar suficientes docs con B para llegar
    # a frecuencias bajas (10k por ej). Esto pasa porque hay docs que tienen B y A
    # (Hay 150k docs con he y she, donde la freq de he es 182k).
    # Eliminando docs con freqB > freqA no se soluciona esto.
    # Entonces eliminamos cualquier doc con freqB > 0, sabiendo que en ese proceso
    # eliminamos algunos docs con A, pero son relativamente pocos -- la freq de A
    # no se modifica significativamente
    del df

    logging.info("Computing total number of lines")
    with open(args.corpus_file) as f:
        n_docs = sum(1 for _ in f)

    if len(log10_freqs_undersampling) > 0:
        logging.info("Undersampling context B")
        run_undersampling(
            df_target_docs, log10_freqs_undersampling, freq_a, freq_b, 
            args.corpus_file, args.outdir, args.seed, args.word_b, corpus_size=n_docs)

    if len(log10_freqs_oversampling) > 0:
        logging.info("Oversampling context B")
        run_oversampling(
            df_target_docs, log10_freqs_oversampling, freq_a, freq_b, 
            args.corpus_file, args.outdir, args.seed, args.word_b, corpus_size=n_docs)


def run_undersampling(
    df_docs: pd.DataFrame, target_log10_freqs_b: List[int], freq_a: int, freq_b: int, 
    corpus_file: str, outdir: str, seed: int, word_b: str, corpus_size: int
) -> pd.DataFrame:
    """
    """
    # we create a random sequence of docs to drop to reach the desired ratio
    logging.info("Creating sequence of documents to drop")
    df_docs_to_drop = df_docs.sample(frac=1, replace=False, random_state=seed)
    df_docs_to_drop = df_docs_to_drop[['doc', 'freq_a', 'freq_b']]    
    # cumulative counts
    df_docs_to_drop['freq_a_cum'] = np.cumsum(df_docs_to_drop['freq_a'])
    df_docs_to_drop['freq_b_cum'] = np.cumsum(df_docs_to_drop['freq_b'])
    df_docs_to_drop['freq_a_tot'] = freq_a - df_docs_to_drop['freq_a_cum']
    df_docs_to_drop['freq_b_tot'] = freq_b - df_docs_to_drop['freq_b_cum']

    # keep only rows of docs needed to reach the smallest freq
    min_freq_b = 10 ** min(target_log10_freqs_b)
    df_docs_to_drop = df_docs_to_drop.query("freq_b_tot >= @min_freq_b").copy()

    # outfile names
    corpus_name = Path(corpus_file).stem
    outfiles = [f"{corpus_name}_undersampled_{word_b}{f}.txt" for f in target_log10_freqs_b]
    outfiles = [Path(outdir) / f for f in outfiles]

    def write_to_file(
        df_docs: pd.DataFrame, target_freq_b: float, infile: str, outfile: str, 
        corpus_size: int
    ) -> None:
        indices_to_rm = set(df_docs.query("freq_b_tot >= @target_freq_b")['doc'].tolist())
        with open(infile, "r", encoding="utf-8") as f_in:
            with open(outfile, "w", encoding="utf-8") as f_out:
                for i, line in tqdm(enumerate(f_in), total=corpus_size):
                    if i not in indices_to_rm:
                        f_out.write(line)
    
    logging.info("Iterating over outfiles")
    for log10_freq, outfile in zip(target_log10_freqs_b, outfiles):
        freq_ = 10 ** log10_freq
        write_to_file(df_docs_to_drop, freq_, corpus_file, outfile, corpus_size)


def run_oversampling(
    df_docs: pd.DataFrame, target_log10_freqs_b: List[int], freq_a: int, freq_b: int, 
    corpus_file: str, outdir: str, seed: int, word_b: str, corpus_size: int
) -> pd.DataFrame:
    """
    """
    # we create an aribtrarily large sequence of documents to sample from
    #  we will add these documents to the original corpus 
    #  until reaching the desired ratio
    logging.info("Creating sequence of documents to sample from")
    n_sample = 10_000_000
    df_docs_to_sample = df_docs.sample(n_sample, replace=True, random_state=seed)
    df_docs_to_sample = df_docs_to_sample[['doc', 'freq_a', 'freq_b']]
    # cumulative counts and cumulative ratio of A/B
    df_docs_to_sample['freq_a_cum'] = np.cumsum(df_docs_to_sample['freq_a'])
    df_docs_to_sample['freq_b_cum'] = np.cumsum(df_docs_to_sample['freq_b'])
    df_docs_to_sample['freq_a_tot'] = df_docs_to_sample['freq_a_cum'] + freq_a
    df_docs_to_sample['freq_b_tot'] = df_docs_to_sample['freq_b_cum'] + freq_b
    # freq_b_tot is the freq of B if we add the documents in df_docs_to_sample 
    #  up to that line to the original corpus

    # keep only rows of docs needed to reach the largest freq
    max_freq = 10 ** max(target_log10_freqs_b)
    df_docs_to_sample = df_docs_to_sample.query("freq_b_tot <= @max_freq")

    # outfile names
    corpus_name = Path(corpus_file).stem
    outfiles = [f"{corpus_name}_oversampled_{word_b}{f}.txt" for f in target_log10_freqs_b]
    outfiles = [Path(outdir) / f for f in outfiles]

    logging.info("Getting text of lines to write")
    doc_numbers = sorted(df_docs_to_sample['doc'].unique().tolist())
    doc2text = {}
    current_doc_number = doc_numbers.pop(0)
    with open(corpus_file, "r", encoding="utf-8") as f:
        for i, text in tqdm(enumerate(f), total=corpus_size):
            if i == current_doc_number:
                doc2text[current_doc_number] = text
                if len(doc_numbers) == 0:
                    break
                current_doc_number = doc_numbers.pop(0)

    logging.info("Writing oversampled lines to files")
    target_freqs_b = [10 ** f for f in target_log10_freqs_b]
    with ExitStack() as stack:
        files = [stack.enter_context(open(file, "w", encoding="utf-8")) for file in outfiles]
        for freq, f in zip(target_freqs_b, files):
            doc_numbers = df_docs_to_sample.query("freq_b_tot <= @freq")['doc']
            lines_to_write = doc_numbers.map(doc2text)
            for line in lines_to_write.values:
                f.write(line)

    logging.info("Appending whole corpus to each file")
    with ExitStack() as stack:
        files = [stack.enter_context(open(file, "a", encoding="utf-8")) \
                                                        for file in outfiles]
        with open(corpus_file, "r", encoding="utf-8") as f_in:
            for line in tqdm(f_in, total=corpus_size):
                for f_out in files:
                    f_out.write(line)


if __name__ == "__main__":
    main()


