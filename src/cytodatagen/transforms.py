"""Transform effects on synthetic data."""


import abc
import anndata as ad
import numpy as np
import pandas as pd

from cytodatagen.registry import xform_registry


class Transform(abc.ABC):
    @abc.abstractmethod
    def apply(self, adata: ad.AnnData, rng=None) -> ad.AnnData:
        pass

    def __call__(self, adata: ad.AnnData, rng=None):
        return self.apply(adata, rng)

    def to_dict(self) -> dict:
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, data: dict):
        raise NotImplementedError()


@xform_registry.register_class("sinh")
class SinhTransform(Transform):
    def __init__(self, cofactor: float | np.ndarray = 5.0):
        super().__init__()
        self.cofactor = cofactor

    def apply(self, adata: ad.AnnData, rng=None) -> ad.AnnData:
        adata.X = self.cofactor * np.sinh(adata.X)
        return adata

    def to_dict(self):
        return {"_target_": "sinh", "cofactor": self.cofactor}

    @classmethod
    def from_dict(cls, data):
        return cls(data["cofactor"])


@xform_registry.register_class("exp")
class ExpTransform(Transform):
    def __init__(self):
        super().__init__()

    def apply(self, adata: ad.AnnData, rng=None) -> ad.AnnData:
        adata.X = np.exp(adata.X)
        return adata

    def to_dict(self):
        return {"_target_": "exp"}

    @classmethod
    def from_dict(cls, data):
        return cls()


@xform_registry.register_class("noise")
class NoiseTransform(Transform):
    def __init__(self, snr_db: float = 20):
        super().__init__()
        self.snr_db = snr_db

    def apply(self, adata: ad.AnnData, rng=None) -> ad.AnnData:
        rng = np.random.default_rng(rng)
        signal_scale = adata.X.std(axis=0)
        noise_scale = signal_scale / self.snr
        noise = rng.normal(scale=noise_scale)
        adata.X = adata.X + noise
        adata.layers["noise"] = noise
        return adata

    @property
    def snr(self):
        return np.power(10, self.snr_db / 10.0)

    def to_dict(self):
        return {"_target_": "noise", "snr_db": self.snr_db}

    @classmethod
    def from_dict(cls, data):
        return cls(data["snr_db"])


@xform_registry.register_class("batch")
class BatchTransform(Transform):
    def __init__(self, n_batch: int, scale: float = 1.0):
        super().__init__()
        self.n_batch = n_batch
        self.scale = scale

    def apply(self, adata: ad.AnnData, rng=None) -> ad.AnnData:
        rng = np.random.default_rng(rng)
        # randomly assign subjects to batches
        subject_ids = adata.obs["subject_id"]
        ids = subject_ids.unique()
        batch_ids = rng.permuted(np.arange(len(ids)) % self.n_batch)
        batch_shifts = rng.normal(scale=self.scale, size=self.n_batch)[batch_ids]
        df = pd.DataFrame({"batch_id": batch_ids, "batch_shift": batch_shifts}, index=ids)
        # apply batch shifts
        adata.X += df.loc[subject_ids]["batch_shift"].to_numpy()[:, np.newaxis]
        adata.obs["batch_id"] = df.loc[subject_ids]["batch_id"].to_numpy()
        adata.obs["batch_shift"] = df.loc[subject_ids]["batch_shift"].to_numpy()
        return adata

    def to_dict(self):
        return {"_target_": "batch"}

    @classmethod
    def from_dict(cls, data):
        return cls(data["n_batch"], data["scale"])


@xform_registry.register_class("composition")
class ComposedTransform(Transform):
    def __init__(self, transforms: list[Transform]):
        super().__init__()
        self.transforms = transforms

    def apply(self, adata, rng=None):
        for xform in self.transforms:
            adata = xform(adata, rng=rng)
        return adata

    def to_dict(self):
        xforms = [xform.to_dict() for xform in self.transforms]
        data = {"_target_": "composition", "transforms": xforms}
        return data

    @classmethod
    def from_dict(cls, data):
        transforms = []
        for xform in data["transforms"]:
            key = xform.pop("_target_")
            transform = xform_registry.get(key).from_dict(xform)
            transforms.append(transform)
        return cls(transforms)


class TransformBuilder:

    _xforms = {
        "sinh": SinhTransform,
        "exp": ExpTransform,
        "noise": NoiseTransform,
        "batch": BatchTransform
    }

    def __init__(self, transforms: dict):
        self.transforms = dict() if transforms is None else transforms

    def __call__(self):
        return self.build()

    def build(self):
        xforms = []
        for key, value in self.transforms.items():
            xform_cls = self._xforms[key]
            xform = xform_cls(**value)
            xforms.append(xform)
        xform = ComposedTransform(xforms)
        return xform
