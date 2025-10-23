"""
Plotting functions for visualizing the generated dataset.
"""

import anndata as ad
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from cytodatagen.utils.adata import adata_to_df


def embed_tsne(adata: ad.AnnData, cat_obs=True, subsample: int = 10_000, random_state=19, **kwargs) -> pd.DataFrame:
    """Returns the t-SNE embedding of adata.X concatenated with adata.obs"""
    # subsample, to speed up t-SNE
    if len(adata.X) > subsample:
        obs = adata.obs.sample(subsample, random_state=random_state)
        adata = adata[obs.index]
    tsne = TSNE(n_components=2, random_state=random_state, **kwargs)
    x_tsne = tsne.fit_transform(adata.X)
    df = pd.DataFrame(dict(tsne_1=x_tsne[:, 0], tsne_2=x_tsne[:, 1]), index=adata.obs.index)
    if cat_obs:
        df = pd.concat([adata.obs, df], axis=1)
    return df


def plot_tsne(adata: ad.AnnData, label="ct", as_categorical=True, title=None, ax=None) -> pd.DataFrame:
    """Plots the t-SNE embedding of adata colored by label."""
    df = embed_tsne(adata)
    if as_categorical:
        df[label] = df[label].astype("category")
    sns.scatterplot(df, x="tsne_1", y="tsne_2", hue=label, ax=ax)
    if title is not None:
        plt.title(title)
    return df


def plot_tsne_matrix():

    raise NotImplementedError()


def plot_shared_tsne_grid(adata: ad.AnnData, col="sample_id", hue="pop", col_wrap=None, add_legend=True, n_jobs=None, **kwargs):
    """Creates a matrix of t-SNE scatterplots with a shared t-SNE embedding space for all group members."""
    df = embed_tsne(adata, n_jobs=n_jobs)
    g = sns.FacetGrid(df, col=col, col_wrap=col_wrap, hue=hue, **kwargs)
    g.map_dataframe(sns.scatterplot, x="tsne_1", y="tsne_2")
    if add_legend:
        g.add_legend()
    return df


def plot_scatter_grid(adata: ad.AnnData, *, x=None, y=None, col=None, add_legend=True, **kwargs) -> sns.FacetGrid:
    """Creates a scatterplot FacetGrid of variables x,y in adata.X, grouped by adata.obs[col]."""
    df = pd.DataFrame(adata[:, [x, y]], columns=[x, y], index=adata.obs.index)
    df[col] = adata.obs[col]
    g = sns.FacetGrid(df, col=col, **kwargs)
    g.map_dataframe(sns.scatterplot, x=x, y=y)
    if add_legend:
        g.add_legend()
    return g


def plot_marker_dists(adata: ad.AnnData, hue="subject", inner="quart", **kwargs):
    """Violin plot for marker distributions."""
    marker_cols = adata.var_names.to_list()
    df = adata_to_df(adata, with_obs=True)
    df = pd.melt(df, id_vars=hue, value_vars=marker_cols, var_name="variable", value_name="value")
    fig, ax = plt.subplots(figsize=(6, 12))
    sns.violinplot(df, y="variable", x="value", hue=hue, ax=ax, split=True, inner=inner, **kwargs)
    return ax


def plot_ct_dists(adata: ad.AnnData, x="pop_name", hue="subject", stat="percent"):
    """Plots cell type distributions."""
    df = adata.obs
    sns.histplot(df, x=x, hue=hue, stat=stat)


def plot_marker_pairplot(adata: ad.AnnData, markers=None, kind="kde", hue="subject", **kwargs):
    """Plots pairplot for marker distributions."""
    if markers is not None:
        adata = adata[:, markers]
    df = adata.to_df()
    df[hue] = adata.obs[hue]

    sns.pairplot(df, hue=hue, kind=kind, **kwargs)
