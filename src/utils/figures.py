
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy import sparse
from scipy.sparse import linalg as splinalg


def str2floats(s):
    """Convert strings separated by '|' to list
    """
    res = [float(i) for i in s.split("|")]
    return res


def scatter_plt(
    data, x_var, y_var, flag_color=None, colors=['#1f77b4','#ff7f0e'], x_log=True
    , n_sample=None, smooth=False, frac=0.1, seed=123):
    """
    Scatter plot (x with log10)
    Param:
        - flag_color: name of binary variable to color points with
        - n_sample: size of sample or None
        - lowess: plot smooth curve or not
        - frac: lowess frac
    """
    if n_sample:
        data = data.sample(n_sample, random_state=seed)
    data_resto = data
    if flag_color:
        data_flag = data[data[flag_color] == 1]
        data_resto = data[data[flag_color] == 0]
    fig, ax = plt.subplots()
    if x_log:
        ax.set_xscale('log')
    plt.scatter(
        x_var, y_var, linewidth=0, c=colors[0], s=8, data=data_resto, label=0)
    if flag_color:
        plt.scatter(
            x_var, y_var, linewidth=0, c=colors[1], s=8, data=data_flag, label=1)
    if smooth:
        x_data = data[x_var]
        if x_log:
            x_data = np.log10(data[x_var])
        smooth_data = lowess(data[y_var], x_data, frac=frac)
        x_smooth = smooth_data[:,0]
        if x_log:
            x_smooth = 10**smooth_data[:,0]
        line = ax.plot(
            x_smooth, smooth_data[:,1], color='black', lw=1.0, ls='--')
    ax.axhline(0, ls='--', color='gray', linewidth=0.5)
    plt.xlabel(x_var)
    plt.ylabel(y_var)
    plt.legend()
    return fig, ax


def boxplots_plt(
    data, x_var, y_var, bins=[1,2,3,4,5,6,9], ax=None, n_labels=False):
    """
    Cut x_var in bins and make one boxplot per bin
    """
    freq_bins = pd.cut(np.log10(data[x_var]), bins=bins)
    box = sns.boxplot(
        x=freq_bins, y=data[y_var], showfliers=False, color="white", ax=ax)
    box.axhline(0, ls='--', color='black', linewidth=0.5)
    if n_labels:
        labels_ypos = box.get_ylim()[1] + .005
        nobs = freq_bins.value_counts().values
        nobs = [str(x) for x in nobs.tolist()]
        nobs = ["n: " + i for i in nobs]
        for i in range(len(nobs)):
            plt.text(
                i, labels_ypos, nobs[i], horizontalalignment='center', size='small'
                , color='black', weight='semibold')


def similarity_by_pairs(
    M, indices, normalize=True, shift=None, use_vectors=True
):
    """
    Returns array with cosine similarity and negative euclidean distance of 
    pairs of indices of an array M
    Params:
        - matrix: np.array of shape dim x V
        - indices: tuple of len=2 with indices
        - normalize: use cosine similarity instead of dot
        - shift: computes x - np.log(shift) prior to computations
        - use_vectors: use similarity between vectors. If False, uses the 
          values at the intersection of the indices.        
    
    NOTE computations assume embeddings are in the columns!
    """
    if not use_vectors:
        similarities = np.array(M[indices[1], indices[0]]).ravel()
        if shift:
            similarities = similarities - np.log(shift)
        return similarities
    M_1 = M[:, indices[0]].astype(np.float32) # float32 to avoid overflow
    M_2 = M[:, indices[1]].astype(np.float32)
    if shift:
        M_1 = M_1.toarray() - np.log(shift)
        M_2 = M_2.toarray() - np.log(shift)
    if sparse.issparse(M_1):
        # dot product de a pares
        similarities = M_1.multiply(M_2).sum(axis=0)
        similarities = np.asarray(similarities).ravel()
        if normalize:
            # cosine
            denominadores = splinalg.norm(M_1, axis=0) * \
                              splinalg.norm(M_2, axis=0)
            similarities /= denominadores
            # # and normalize for euclidean distance
            # M_1 = M_1 / splinalg.norm(M_1, axis=0)
            # M_2 = M_2 / splinalg.norm(M_2, axis=0)
        # euclidean distance
        euc_distances = splinalg.norm(M_1 - M_2, axis=0)
    else:
        # dot product de a pares (same as np.sum(a*b, axis=0))
        similarities = np.einsum('ij,ij->i', M_1.T, M_2.T)
        if normalize:
            # cosine
            denominadores = np.linalg.norm(M_1, axis=0) * np.linalg.norm(M_2, axis=0)
            similarities /= denominadores
            # # and normalize for euclidean distance
            # M_1 = M_1 / np.linalg.norm(M_1, axis=0)
            # M_2 = M_2 / np.linalg.norm(M_2, axis=0)
        # euclidean distance
        euc_distances = np.linalg.norm(M_1 - M_2, axis=0)
    return similarities.ravel(), -euc_distances.ravel()


def sample_idx_pairs(
    df, n_pairs, group_var, col_number_var, group_values, colnumber2str: dict = None
):
    """
    Returns 2 lists A,B such that A[i],B[i] are samples of word pairs from df
    Params:
        - n_pairs: number of pairs to sample
        - group_var: name of variable that defines group
        - col_number_var: name of variable that defines column number
        - group_values: tuple of 2 groups from where to sample indices
        - colnumber2str: dict with column number as key and word as value (used
          to avoid sampling pairs with the same word in CWE)
    NOTE: puede haber pares repetidos (sucede cuando el grupo es chico)
    """
    idx_a = df.loc[df[group_var].isin([group_values[0]])][col_number_var]
    idx_b = df.loc[df[group_var].isin([group_values[1]])][col_number_var]
    idx_a_sample = np.random.choice(idx_a, n_pairs, replace=True)
    idx_b_sample = np.full_like(idx_a_sample, -1) # init with -1
    to_sample = np.full(n_pairs, True) # init mask to sample
    # until there are no equal values in each pair:
    if colnumber2str is None:
        while to_sample.sum() > 0:
            new_values = np.random.choice(idx_b, to_sample.sum(), replace=True)
            idx_b_sample[to_sample] = new_values
            to_sample = (idx_a_sample == idx_b_sample)
    else:
        words_a_sample = np.array([colnumber2str[i] for i in idx_a_sample])
        while to_sample.sum() > 0:
            new_values = np.random.choice(idx_b, to_sample.sum(), replace=True)
            idx_b_sample[to_sample] = new_values
            words_b_sample = np.array([colnumber2str[i] for i in idx_b_sample])
            to_sample = (idx_a_sample == idx_b_sample) | (words_a_sample == words_b_sample)
    return idx_a_sample, idx_b_sample


def create_similarity_df(
    df, M, group_var, col_number_var, n_samples, normalize=True, shift=None,
    use_vectors=True, colnumber2str: dict = None
):
    """
    Returns df with similarities of sampled pairs for each combination
    of group_var values
    Param:
        - df: DataFrame of words and groups
        - M: sparse matrix or arr with shape (d, V)
        - n_samples: number of pairs of each combo to sample
        - normalize: use cosine similarity instead of dot prod
        - shift: computes x - np.log(shift) prior to computations
        - use_vectors: use similarity between vectors. If False, uses the 
          values at the intersection of the indices.

    NOTE we assume embeddings are in columns!
    """
    # cross join of DataFrame (all combos of group_freq)
    df_tmp = df[[group_var]].drop_duplicates().reset_index(drop=True)
    df_tmp['key'] = 0
    df_cross = pd.merge(df_tmp, df_tmp, how="outer", on="key")
    df_cross.drop(columns=['key'], inplace=True)
    df_cross['pairs'] = df_cross.apply(
        lambda d: sample_idx_pairs(
            df, n_samples, group_var, col_number_var, 
            (d[group_var+'_x'], d[group_var+'_y']), colnumber2str=colnumber2str), 
            axis=1)
    out = df_cross.apply(
        lambda d: similarity_by_pairs(
            M, d['pairs'], normalize=normalize, shift=shift, 
            use_vectors=use_vectors), axis=1)
    if use_vectors:
        df_cross[['cosine', 'negative_distance']] = pd.DataFrame(out.tolist())
        for col in ['cosine', 'negative_distance']:
            df_cross['mean_' + col] = df_cross[col].apply(np.mean)
            df_cross['sd_' + col] = df_cross[col].apply(np.std)
    else:
        df_cross['similarities'] = out
        df_cross['mean_similarity'] = df_cross['similarities'].apply(np.mean)
        df_cross['sd_similarity'] = df_cross['similarities'].apply(np.std)
    return df_cross


def make_heatmap(df, group_var, column="mean_cosine", cmap="viridis", **kwargs):
    """
    Make a crosstab and return a heatmap
    Param:
        - df: DataFrame with pairs and similarities
        - M: matrix with word vectors
        - group_var: name of variable that defines group
        - n_samples: number samples of pairs of words
        - normalize: use cosine similarity instead of dot prod
    """
    crosstab = df.pivot(
        index=group_var+'_x', columns=group_var+'_y', values=column)
    crosstab.sort_index(level=0, ascending=False, inplace=True)
    ax = sns.heatmap(crosstab, cmap=cmap, **kwargs)
    ax.set(xlabel=group_var, ylabel=group_var)
    return ax
