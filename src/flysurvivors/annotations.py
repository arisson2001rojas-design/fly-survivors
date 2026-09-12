"""FlyWire neuron annotations (Schlegel et al. 2024): cell types, sides, soma positions.

Source: ``Supplemental_file1_neuron_annotations.tsv`` from
https://github.com/flyconnectome/flywire_annotations (root ids at version 783).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .connectome import DATA_DIR, Connectome

ANNOTATIONS_FILE = "Supplemental_file1_neuron_annotations.tsv"

COLUMNS = [
    "root_id",
    "pos_x",
    "pos_y",
    "pos_z",
    "soma_x",
    "soma_y",
    "soma_z",
    "super_class",
    "cell_class",
    "cell_type",
    "hemibrain_type",
    "side",
    "top_nt",
]


class Annotations:
    """Annotation table restricted to neurons present in a :class:`Connectome`.

    ``df`` is indexed by model index and has the columns listed in :data:`COLUMNS`.
    """

    def __init__(self, cn: Connectome, data_dir: Path | str = DATA_DIR) -> None:
        path = Path(data_dir) / ANNOTATIONS_FILE
        if not path.exists():
            raise FileNotFoundError(f"{path} missing. Run `python scripts/download_data.py`.")
        df = pd.read_csv(path, sep="\t", usecols=COLUMNS, low_memory=False)
        idx, missing = cn.index_of_existing(df["root_id"].to_numpy())
        keep = ~df["root_id"].isin(missing)
        df = df[keep].copy()
        df.index = pd.Index(idx, name="idx")
        df["cell_type"] = df["cell_type"].fillna("")
        df["side"] = df["side"].fillna("")
        self.df = df
        self.cn = cn

    def select(self, cell_type: str | list[str], side: str | None = None) -> np.ndarray:
        """Model indices of neurons of the given type(s), optionally on one side."""
        types = [cell_type] if isinstance(cell_type, str) else list(cell_type)
        m = self.df["cell_type"].isin(types)
        if side is not None:
            m &= self.df["side"] == side
        return self.df.index[m].to_numpy()

    def groups(self, spec: dict[str, tuple[str | list[str], str | None]]) -> dict[str, np.ndarray]:
        """``{name: (cell_type, side)} -> {name: indices}``, dropping empty groups."""
        out = {}
        for name, (ct, side) in spec.items():
            idx = self.select(ct, side)
            if len(idx):
                out[name] = idx
        return out

    def counts(self, pattern: str) -> pd.Series:
        """Cell-type counts by side for types matching a regex, for exploration."""
        m = self.df["cell_type"].str.match(pattern)
        return self.df[m].groupby(["cell_type", "side"]).size()
