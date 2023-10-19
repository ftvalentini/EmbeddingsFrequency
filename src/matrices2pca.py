
import glob
import re
import logging
from typing import Union
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA

from utils.vocab import load_vocab


logging.basicConfig(
    format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)


INPUT_FILES = [
    "data/working/w2v-wiki2021-V100-W10-D300-SG1-S1-NS5-NSE0.75.npy",
    "data/working/w2v-wiki2021s1-V100-W10-D300-SG1-S1-NS5-NSE0.75.npy",
    "data/working/ft-wiki2021-V100-W10-D300-SG1-S1-NS5-NSE0.75.npy",
    "data/working/ft-wiki2021s1-V100-W10-D300-SG1-S1-NS5-NSE0.75.npy",
    "data/working/glove-wiki2021-V100-W10-D1-D300-R0.05-E100-M2-S1.npy",
    "data/working/glove-wiki2021s1-V100-W10-D1-D300-R0.05-E100-M2-S1.npy",
]
IDX_FILE = None
VOCAB_FILE = "data/working/vocab-wiki2021-V100.txt"
N_COMPONENTS = 2 # PCA dims
SEED = 33

def main():

    parser = ArgumentParser()
    parser.add_argument("--normalize", action="store_true", 
                        help="Normalize vectors before PCA")

    args = parser.parse_args()

    output_file = "results/pca.csv"
    if args.normalize:
        output_file = "results/pca-normalized.csv"

    input_files = []
    for pattern in INPUT_FILES:
        input_files.extend(glob.glob(pattern))
    input_files = sorted(input_files)
    
    logging.info("Input files:")
    print(*input_files, sep="\n")

    logging.info(f"Reading vocab {VOCAB_FILE}")
    str2idx, idx2str, str2count = load_vocab(VOCAB_FILE)

    if IDX_FILE:
        logging.info(f"Reading {IDX_FILE}")
        indices = np.loadtxt(IDX_FILE, dtype=int)
    else:
        indices = stratified_sampling(VOCAB_FILE, n_by_bin=100, random_state=SEED)

    logging.info("Preparing output DF")
    df = create_vocab_df(idx2str, str2count, indices)

    # empty list for variance ratios
    list_ratios = []

    for i, f in enumerate(input_files):

        metric_name = make_colname(f)

        logging.info(f"Reading {f}")
        M = load_matrix(f)

        # Use idx of sampled 10k words
        if not "pmi_eps" in metric_name:
            print(f" Filtering indices")
            M = M[:, indices]

        # max(x, 0) para PPMI
        if "ppmi" in metric_name:
            if sparse.issparse(M):
                M = M.maximum(0)
            else:
                M = np.maximum(M, 0)

        # remove first row if has NULLs (not necessary in sparse data, used in PMI w/epsilon)
        # (TODO drop if using 0 idx with data)
        if not sparse.issparse(M):
            if np.isnan(M[0,:]).all():
                M = M[1:,:]

        # convert to float32 to avoid overflow in float16
        if M.dtype != np.float32:
            M = M.astype(np.float32)
        
        # normalize vectors
        if args.normalize:
            logging.info(f"Normalizing vectors")
            M = M / np.linalg.norm(M, axis=0)

        logging.info(f"Computing principal components of {metric_name}")
        components, ratios = compute_pca(M, N_COMPONENTS, variance_ratios=True)

        # print(f" Adding {metric_name} PCs to DF")
        colnames = [f"{metric_name}-{i}" for i in range(N_COMPONENTS)]
        df[colnames] = components

        # print(f" Adding {metric_name} ratios to DF")
        list_ratios.append([metric_name] + list(ratios))

        # also compute with shift=5 for pmi and ppmi:
        if "pmi" in metric_name: 
            shift = 5
            if sparse.issparse(M):
                M = M.toarray()
            M = np.add(M, -np.log(shift), out=M)
            components, ratios = compute_pca(M, N_COMPONENTS, variance_ratios=True)
            metric_name += "_shift"
            print(f" Adding {metric_name} PCs to DF")
            colnames = [f"{metric_name}-{i}" for i in range(N_COMPONENTS)]
            df[colnames] = components
            list_ratios.append([metric_name] + list(ratios))

    # Dframe for variance ratios
    df_ratios = pd.DataFrame(list_ratios, columns=["vectors"] + list(range(N_COMPONENTS)))

    logging.info("Saving output DFs")
    df.to_csv(output_file, index=False)
    df_ratios.to_csv(output_file.replace(".csv", "_ratios.csv"), index=False)


def compute_pca(
    M: Union[sparse.csr_matrix, sparse.csc_matrix, np.ndarray], n_dim: int,
    variance_ratios: bool = False
) -> np.ndarray:
    """Get PCA dims of a matrix. If input is sparse, it is converted to dense.
    
    NOTE we use the transpose of the matrix because vectors are in columns! 
    """
    if sparse.issparse(M):
        M = M.toarray()
    pca = PCA(n_components=n_dim)
    M_reduced = pca.fit_transform(M.T)
    if variance_ratios:
        return M_reduced, pca.explained_variance_ratio_
    return M_reduced


def stratified_sampling(
    vocab_file: str, n_by_bin: int, random_state=33
    ) -> np.ndarray:
    """Stratified random sampling of words by frequency.
    """
    str2idx, idx2str, str2count = load_vocab(vocab_file)
    df_words = pd.DataFrame(str2count.items(), columns=["word", "freq"])
    df_words["idx"] = df_words["word"].map(str2idx)
    del str2idx, str2count, idx2str
    # adhoc bins for wiki2021
    bins = np.arange(2, 9, .25)
    bins[bins > 6] = 9
    bins = np.unique(bins)
    df_words['bin'] = pd.cut(np.log10(df_words['freq']), bins=bins, right=False)
    n_bins = df_words['bin'].nunique()
    df_words = (
        df_words
            .groupby("bin", group_keys=False)
            .apply(
                lambda x: x.sample(n_by_bin, replace=False, 
                random_state=random_state)
                )
            .sort_values("idx")
            .copy()
    )
    sampled_idx = df_words["idx"].values
    # add most freq idx (1) to the sampled idx
    sampled_idx = np.concatenate([[1], sampled_idx])
    sampled_idx = np.unique(sampled_idx)
    return sampled_idx


def load_matrix(file_path: str) -> Union[sparse.csr_matrix, sparse.csc_matrix, np.ndarray]:
    """
    """
    if file_path.endswith("npz"):
        M = sparse.load_npz(file_path)
    elif file_path.endswith("npy"):
        M = np.load(file_path)
    return M


def create_vocab_df(
    idx2str: dict, str2count: dict, indices: np.ndarray = None
) -> pd.DataFrame:
    """
    Args:
        indices: idx of words to keep
    """
    df = pd.DataFrame(idx2str.items(), columns=["idx", "word"])
    df["freq"] = df["word"].map(str2count)
    if indices is not None:
        df = df.loc[df["idx"].isin(indices)].copy()
    return df


def make_colname(file_path: str) -> str:
    """Parse file path to get a name according to the similarity metric 
    (e.g. pmi_eps, ppmi, glove, etc.)
    """
    basename = Path(file_path).stem
    basename_parts = basename.split("-")
    model_name = basename_parts[0]
    corpus = basename_parts[1]
    if model_name == "w2v":
        # if any element contains NS
        if any(re.search("NS", i) for i in basename_parts):
            if "NS5" not in basename_parts:
                # non default neg sampling
                ns_term = [
                    w for w in basename_parts if re.match(r"NS\d+", w)][0].replace(
                        "NS", "")
                model_name += f"_ns{ns_term}" 
            if "NSE0.75" not in basename_parts:
                # non default neg sampling distribution
                nse_term = [
                    w for w in basename_parts if re.match(r"NSE.+", w)][0].replace(
                        "NSE", "")
                model_name += f"_nse{nse_term}"
    if model_name == "pmi":
        if "S0" in basename_parts:
            model_name = "ppmi" # ppmi (no epsilon smoothing)
        else:
            model_name = "pmi_eps" # epsilon smoothing
        if "C1" not in basename_parts:
            model_name += "_alpha" # alpha smoothing
    colname = f"{corpus}-{model_name}"
    return colname


if __name__ == '__main__':
    main()
