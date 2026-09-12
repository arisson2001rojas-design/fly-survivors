"""Load the FlyWire v783 connectome into flat numpy arrays.

Data files come from Shiu et al. 2024 (https://github.com/philshiu/Drosophila_brain_model):

- ``Completeness_783.csv``: one row per neuron, index = FlyWire root id. Row order
  defines the model index used everywhere in this package.
- ``Connectivity_783.parquet``: one row per (pre, post) pair with the synapse count
  and the sign of the presynaptic neurotransmitter (+1 excitatory, -1 inhibitory).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "flywire_783"
COMPLETENESS_FILE = "Completeness_783.csv"
CONNECTIVITY_FILE = "Connectivity_783.parquet"


@dataclass
class Connectome:
    """Flat edge-list representation of the connectome."""

    ids: np.ndarray  # (N,) int64 FlyWire root ids; position = model index
    pre: np.ndarray  # (E,) int32 model index of the presynaptic neuron
    post: np.ndarray  # (E,) int32 model index of the postsynaptic neuron
    weight: np.ndarray  # (E,) float32 signed synapse count (+ excitatory, - inhibitory)

    def __post_init__(self) -> None:
        self._id2idx: dict[int, int] | None = None

    @property
    def n_neurons(self) -> int:
        return int(len(self.ids))

    @property
    def n_edges(self) -> int:
        return int(len(self.pre))

    def index_of(self, root_ids) -> np.ndarray:
        """Map FlyWire root ids to model indices. Raises KeyError on unknown ids."""
        if self._id2idx is None:
            self._id2idx = {int(r): i for i, r in enumerate(self.ids)}
        out = []
        missing = []
        for r in np.atleast_1d(np.asarray(root_ids, dtype=np.int64)):
            i = self._id2idx.get(int(r))
            if i is None:
                missing.append(int(r))
            else:
                out.append(i)
        if missing:
            raise KeyError(f"{len(missing)} root id(s) not in connectome, e.g. {missing[:3]}")
        return np.asarray(out, dtype=np.int64)

    def index_of_existing(self, root_ids) -> tuple[np.ndarray, list[int]]:
        """Like :meth:`index_of` but skips unknown ids and returns them separately."""
        if self._id2idx is None:
            self._id2idx = {int(r): i for i, r in enumerate(self.ids)}
        found, missing = [], []
        for r in np.atleast_1d(np.asarray(root_ids, dtype=np.int64)):
            i = self._id2idx.get(int(r))
            (found if i is not None else missing).append(i if i is not None else int(r))
        return np.asarray(found, dtype=np.int64), missing

    def summary(self) -> str:
        n_exc = int((self.weight > 0).sum())
        return (
            f"Connectome: {self.n_neurons:,} neurons, {self.n_edges:,} edges "
            f"({n_exc:,} excitatory, {self.n_edges - n_exc:,} inhibitory), "
            f"{int(np.abs(self.weight).sum()):,} synapses"
        )


def load_connectome(
    data_dir: Path | str = DATA_DIR, min_syn: int = 1, cache: bool = True
) -> Connectome:
    """Load the connectome, optionally dropping pairs with fewer than ``min_syn`` synapses.

    The first load parses the parquet (~15 M rows, a few seconds); the result is cached
    as an ``.npz`` next to the source files.
    """
    data_dir = Path(data_dir)
    cache_path = data_dir / f"connectome_minsyn{min_syn}.npz"
    if cache and cache_path.exists():
        z = np.load(cache_path)
        return Connectome(z["ids"], z["pre"], z["post"], z["weight"])

    comp_path = data_dir / COMPLETENESS_FILE
    con_path = data_dir / CONNECTIVITY_FILE
    if not comp_path.exists() or not con_path.exists():
        raise FileNotFoundError(
            f"Connectome files not found in {data_dir}. Run `python scripts/download_data.py`."
        )

    comp = pd.read_csv(comp_path, index_col=0)
    ids = comp.index.to_numpy(dtype=np.int64)

    con = pd.read_parquet(
        con_path,
        columns=[
            "Presynaptic_ID",
            "Postsynaptic_ID",
            "Presynaptic_Index",
            "Postsynaptic_Index",
            "Connectivity",
            "Excitatory",
        ],
    )
    if min_syn > 1:
        con = con[con["Connectivity"] >= min_syn]

    pre = con["Presynaptic_Index"].to_numpy(dtype=np.int32)
    post = con["Postsynaptic_Index"].to_numpy(dtype=np.int32)
    # The parquet carries its own index columns; make sure they agree with the csv order.
    if not (ids[pre] == con["Presynaptic_ID"].to_numpy()).all():
        raise ValueError("Presynaptic_Index does not match Completeness csv order")
    if not (ids[post] == con["Postsynaptic_ID"].to_numpy()).all():
        raise ValueError("Postsynaptic_Index does not match Completeness csv order")

    weight = (con["Connectivity"].to_numpy() * con["Excitatory"].to_numpy()).astype(np.float32)

    cn = Connectome(ids, pre, post, weight)
    if cache:
        np.savez(cache_path, ids=ids, pre=pre, post=post, weight=weight)
    return cn
