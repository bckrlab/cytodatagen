import numpy as np
import flowkit as fk
import json
import logging
import anndata as ad

from pathlib import Path


logger = logging.getLogger(__name__)


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return super().default(o)


def write_fcs(path, adata: ad.AnnData, sample_id: str = "sample_id"):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    for group, group_df in adata.obs.groupby(sample_id):
        group_adata = adata[group_df.index]
        sample = fk.Sample(fcs_path_or_data=group_adata.X, sample_id=str(group), channel_labels=adata.var_names)
        sample_path = path / f"sample_{group}.fcs"
        logger.info("writing %s", sample_path)
        sample.export(sample_path, source="raw")
    # write uns as json
    json_path = path / "uns.json"
    with open(json_path, "w") as json_file:
        logger.info("writing uns to: %s", json_path)
        json.dump(adata.uns, json_file, cls=NumpyJSONEncoder)
    # write obs as csv
    obs_path = path / "obs.csv"
    logger.info("writing obs to: %s")
    adata.obs.to_csv(obs_path)


def write_h5ad(path, adata: ad.AnnData):
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    path = path / "cytodata.h5ad"
    logger.info("writing hda5 file to %s", path)
    adata.write_h5ad(path)
