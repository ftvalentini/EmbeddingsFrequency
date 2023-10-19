import numpy as np
import pandas as pd

from scipy import sparse


def sparse_euclidean_norms(M, axis=0) -> np.ndarray:
    """
    Returns norms of sparse matrix by axis
    """
    # normas = splinalg.norm(M, axis=0) # NOTE crashes with OOM
    data_ = M.data.copy() # copy temporarily to restore later
    M.data = np.power(M.data, 2)
    normas = np.sqrt(M.sum(axis=axis))
    normas = np.array(normas)[0]
    # restore original data! NOTE this is IMPORTANT
    M.data = data_ 
    return normas


def words2indices(words_target, words_attr_a, words_attr_b, str2idx):
    """
    Converts words to indices and checks words out of vocab
    Param:
        - 3 lists of words (target + 2 attributes)
    """
    # handling attr words out of vocab
    words_attr = words_attr_a + words_attr_b
    attr_outofvocab = [w for w in words_attr if w not in str2idx]
    attr_invocab = set(words_attr) - set(attr_outofvocab)
    assert len(attr_invocab) > 0, "ALL ATTRIBUTE WORDS ARE OUT OF VOCAB"
    if attr_outofvocab:
        print(f'\n{", ".join(attr_outofvocab)} \nATTRIBUTE WORDS NOT IN VOCAB')
    # target words out of vocab
    target_outofvocab = [w for w in words_target if w not in str2idx]
    if target_outofvocab:
        print(f'\n{", ".join(target_outofvocab)} \nTARGET WORDS NOT IN VOCAB')
    # words indices
    idx_t = [str2idx[w] for w in words_target if w not in target_outofvocab]
    idx_a = [str2idx[w] for w in words_attr_a if w not in attr_outofvocab]
    idx_b = [str2idx[w] for w in words_attr_b if w not in attr_outofvocab]
    return idx_t, idx_a, idx_b


def cosine_similarities(M, idx_target, idx_attr, use_norm=True):
    """
    Cosine similarity between target words and attribute words. Returns array of
    shape (len(idx_target), len(idx_attr)) with similarity in each cell.
    Param:
        - M: d x V+1 matrix where column indices are words idx from str2idx
        - idx_context: indices of context word
        - idx_target: indices of target words
        - use_norm: divides by norm (as usual cosine); if False: dot product
    Notes:
        - It works OK if len(idx_*) == 1
        - if M is sparse, it should be sparse.csc_matrix !!!
    """
    M_t = M[:,idx_target] # matriz de target words
    M_a = M[:,idx_attr] # matriz de attr words
    print("  Computing dot products...")
    res = M_t.T @ M_a # rows: target words // cols: dot with each attr
    if use_norm:
        print("  Computing norms...")
        if sparse.issparse(M):
            del M_t, M_a
            normas = sparse_euclidean_norms(M)
            normas_t = normas[idx_target]
            normas_a = normas[idx_attr]
        else:
            normas_t = np.linalg.norm(M_t, axis=0)
            normas_a = np.linalg.norm(M_a, axis=0)
        print("  Dividing by norm...")
        denominadores = np.outer(normas_t, normas_a)
        res = res / denominadores
    return res


def we_bias_scores(
    M, idx_target, idx_attr_a, idx_attr_b, return_components=False,
    use_norm=True, standardize=False, n_dim=None
):
    """
    Bias between target words and attributes A and B. Returns array of shape
    len(idx_target) with score for each target word.
    Param:
        - M: d x V+1 matrix where column indices are words idx from str2idx
        - idx_target: indices of target word
        - idx_attr_*: indices of attribute words
        - use_norm: uses cosine -- if False uses dot product
        - standardize: normalizes difference of means with std (as in Caliskan
        2017)
        - return_components: return all similarities with A and B
        - n_dim: use first n_dim of each row if not None (used por PMI vec)
    Notes:
        - It works OK if len(idx_*) == 1
    """
    # Use only top n dimension (used por PMI vec)
    if n_dim and ((n_dim+1) < M.shape[0]):
        M = M[:(n_dim+1),:] 
        # NOTE +1 porque idx=0 esta vacio; vocab is in the cols
        # NOTE assumes embeddings are in columns! 
    print(" Computing cosine similarities wrt A...")
    similarities_a = cosine_similarities(
        M, idx_target, idx_attr_a, use_norm=use_norm)
    print(" Computing cosine similarities wrt B...")
    similarities_b = cosine_similarities(
        M, idx_target, idx_attr_b, use_norm=use_norm)
    mean_similarities_a = np.mean(similarities_a, axis=1) # avg accross target A
    mean_similarities_b = np.mean(similarities_b, axis=1) # avg accross target B
    res = mean_similarities_a - mean_similarities_b
    if standardize:
        similarities_all = np.concatenate([similarities_a, similarities_b], axis=1)
        std_similarities_all = np.std(similarities_all, axis=1)
        res /= std_similarities_all
    if return_components:
        return res, similarities_a, similarities_b
    return res


def we_bias_df(
    M, words_target, words_attr_a, words_attr_b, str2idx, str2count,
    only_positive=False, **kwargs_bias
):
    """
    Return DataFrame with bias score A/B for each word in words_target, and the
    freq of each word
    Params:
        - M: word vectors matrix d x V+1
        - words_target, words_attr_a, words_attr_b: lists of words
        - str2idx, str2count: vocab dicts
        - only_positive: negative values as zero (used for PPMI vec)
    """
    # negative as zero
    if only_positive:
        print("Using only positive values!")
        M = M.maximum(0)
    if not sparse.issparse(M):
        if np.isnan(M[0,:]).all():
            M = M[1:,:]
    print("Getting words indices...")
    idx_t, idx_a, idx_b = words2indices(
        words_target, words_attr_a, words_attr_b, str2idx)
    print("Computing bias scores...")
    bias_scores, similarities_a, similarities_b = we_bias_scores(
        M, idx_t, idx_a, idx_b, return_components=True, **kwargs_bias)
    print("Similarites as joined strings...")
    def to_joined_string(x):
        lista = x.round(6).astype(str)
        res = "|".join(lista)
        return res
    similarities_a = pd.DataFrame(similarities_a).apply(to_joined_string, axis=1)
    similarities_b = pd.DataFrame(similarities_b).apply(to_joined_string, axis=1)
    print("Computing normas and NNZ...")
    # NOTE axis=0 assumes embeddings are in columns
    if sparse.issparse(M):
        normas = sparse_euclidean_norms(M)
        normas = normas[idx_t]
        npos = (M > 0).sum(axis=0).A1[idx_t] # contar npositive de cada palabra
        nneg = (M < 0).sum(axis=0).A1[idx_t] # contar nnegative de cada palabra
    else:
        normas = np.linalg.norm(M, axis=0)[idx_t]
        npos = (M > 0).sum(axis=0)[idx_t]
        nneg = (M < 0).sum(axis=0)[idx_t]
    print("Preparing DataFrame...")
    str2idx_target = {w: str2idx[w] for w in words_target}
    str2count_target = {w: str2count[w] for w in str2idx_target}
    df = pd.DataFrame(str2idx_target.items(), columns=['word','idx'])
    df['freq'] = str2count_target.values()
    df['bias_score'] = bias_scores
    df['sims_a'] = similarities_a
    df['sims_b'] = similarities_b
    df['norma'] = normas
    df['npos'] = npos
    df['nneg'] = nneg
    return df
