"""
Utilities for handling anndata.AnnData objects.
"""

import anndata as ad
import pandas as pd
import numpy as np


def adata_to_df(adata: ad.AnnData, layer=None, with_obs: bool = True) -> pd.DataFrame:
    """Like adata.to_df, but concatenates with obs."""
    df = adata.to_df(layer=layer)
    if with_obs:
        df = pd.concat([df, adata.obs], axis=1)
    return df


def select_obs(adata: ad.AnnData, *args, key="sample_id") -> ad.AnnData:
    """Selects data points with obs[key] in *args."""
    return adata[adata.obs[key].isin(args)]


def select_samples(adata: ad.AnnData, *args) -> ad.AnnData:
    """Selects data points with obs['sample_id'] in *args."""
    return select_obs(adata, *args, "sample_id")


def arsinh_xform(adata: ad.AnnData, cofactor: float = 5.0) -> ad.AnnData:
    adata.X = np.arcsinh(adata.X / cofactor)
    return adata
