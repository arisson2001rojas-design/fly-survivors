"""Assign visual neurons to eye columns and give each column a viewing direction.

The compound eye has ~800 ommatidia per side; every ommatidium projects to one
"column" that runs through lamina and medulla (6 R1-6 photoreceptors, one L1, L2, L3,
Mi1, ... per column). We need, for each photoreceptor, which column it belongs to and
where that column looks.

Right eye: Matsliah et al. 2024 assigned each right-eye Mi1 to a hexagonal lattice
point (p, q). We ship that table (``resources/``, from OpticLobe.jl, MIT) and propagate
it to the other columnar types through the connectome: a neuron takes the column of the
already-assigned partner it shares the most synapses with (L1 <- Mi1, R1-6 <- L1, ...).

Left eye: no public lattice, so left Mi1 somata are mirrored across the midline and
matched one-to-one to right Mi1 somata (assignment problem); then the same propagation.

Viewing direction: +p is antero-dorsal and +q postero-dorsal on the hex lattice, so
posterior = q - p and dorsal = p + q. With ~5.5 deg between neighbouring ommatidia this
gives azimuth (0 = straight ahead, positive = that eye's side, 180 = behind) and
elevation (0 = horizon, positive = up). Good to a few degrees, enough for a game.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .annotations import Annotations
from .connectome import DATA_DIR, Connectome

RESOURCES = Path(__file__).parent / "resources"
INTEROMMATIDIAL_DEG = 5.5
# Order matters: each type is assigned from partners assigned before it.
PROPAGATION_ORDER = [
    "L1", "L5", "R1-6", "L2", "L3", "L4", "R7", "R8",
    "Tm3", "Tm1", "Tm2", "Tm4", "Tm9", "Mi4", "Mi9", "Tm20", "C2", "C3",
    "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
]
# Types whose columns we keep in the table (drive candidates + analysis).
COLUMN_TYPES = [
    "Mi1", "L1", "L2", "L3", "L4", "L5", "R1-6", "R7", "R8",
    "Tm1", "Tm2", "Tm3", "Tm4", "Tm9", "Mi4", "Mi9", "Tm20", "C2", "C3",
    "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d",
]


@dataclass
class EyeColumns:
    """Per-column geometry and the neurons in each column.

    ``columns``: one row per column with ``col``, ``eye`` ('R'/'L'), ``p``, ``q``,
    ``az_deg``, ``el_deg``. ``cells``: one row per assigned neuron with ``idx``,
    ``cell_type``, ``eye``, ``col``.
    """

    columns: pd.DataFrame
    cells: pd.DataFrame

    def indices(self, cell_type: str, eye: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        """(neuron model indices, row number in ``columns``) for one cell type."""
        c = self.cells[self.cells["cell_type"] == cell_type]
        if eye is not None:
            c = c[c["eye"] == eye]
        key = self.columns.reset_index().set_index(["eye", "col"])["index"]
        rows = key.loc[list(zip(c["eye"], c["col"]))].to_numpy()
        return c["idx"].to_numpy(), rows

    def save(self, path: Path) -> None:
        self.columns.to_parquet(path.with_suffix(".columns.parquet"))
        self.cells.to_parquet(path.with_suffix(".cells.parquet"))

    @classmethod
    def load(cls, path: Path) -> "EyeColumns":
        return cls(
            pd.read_parquet(path.with_suffix(".columns.parquet")),
            pd.read_parquet(path.with_suffix(".cells.parquet")),
        )


def load_right_lattice() -> pd.DataFrame:
    """Right-eye Mi1 root ids with their (p, q) lattice coordinates (OpticLobe.jl)."""
    mi1 = pd.read_csv(RESOURCES / "columns_Mi1.csv")["Mi1"].to_numpy(dtype=np.int64)
    grid = pd.read_csv(RESOURCES / "RightEyeGrid.csv")
    grid = grid.rename(columns={grid.columns[0]: "p"}).set_index("p")
    rows = []
    for p, row in grid.iterrows():
        for q, col in row.items():
            if pd.notna(col):
                rows.append((int(col), int(p), int(q)))
    lat = pd.DataFrame(rows, columns=["col", "p", "q"]).sort_values("col")
    assert len(lat) == len(mi1) and lat["col"].iloc[0] == 1 and lat["col"].iloc[-1] == len(mi1)
    lat["root_id"] = mi1[lat["col"].to_numpy() - 1]
    return lat


def _mirror_left_mi1(ann: Annotations, right: pd.DataFrame) -> pd.DataFrame:
    """Match left Mi1 to right lattice columns by mirrored soma position."""
    df = ann.df
    midline = df["soma_x"].median()

    def coords(sub):
        xyz = sub[["soma_x", "soma_y", "soma_z"]].to_numpy(dtype=float).copy()
        fallback = sub[["pos_x", "pos_y", "pos_z"]].to_numpy(dtype=float)
        bad = np.isnan(xyz).any(axis=1)
        xyz[bad] = fallback[bad]
        xyz[:, 2] *= 10.0  # z voxels are 40 nm, x/y 4 nm
        return xyz

    r = df.loc[right["idx"]]
    l_idx = ann.select("Mi1", "left")
    l = df.loc[l_idx]
    rx = coords(r)
    lx = coords(l)
    lx[:, 0] = 2 * midline - lx[:, 0]
    dist = np.linalg.norm(lx[:, None, :] - rx[None, :, :], axis=2)
    li, ri = linear_sum_assignment(dist)
    out = pd.DataFrame(
        {
            "idx": l_idx[li],
            "col": right["col"].to_numpy()[ri],
            "p": right["p"].to_numpy()[ri],
            "q": right["q"].to_numpy()[ri],
        }
    )
    return out


def _propagate(cn: Connectome, ann: Annotations, seed: dict[int, tuple[str, int]]) -> dict:
    """Assign columns to neurons in PROPAGATION_ORDER from their assigned partners.

    ``seed``: {model index: (eye, col)}. Returns the extended dict.
    """
    assigned = dict(seed)
    # Edge list both directions: partner, weight.
    pre, post, w = cn.pre, cn.post, np.abs(cn.weight)
    order_post = np.argsort(post, kind="stable")
    order_pre = np.argsort(pre, kind="stable")
    ptr_post = np.searchsorted(post[order_post], np.arange(cn.n_neurons + 1))
    ptr_pre = np.searchsorted(pre[order_pre], np.arange(cn.n_neurons + 1))

    def partners(i):
        a, b = ptr_post[i], ptr_post[i + 1]
        c, d = ptr_pre[i], ptr_pre[i + 1]
        ids = np.concatenate([pre[order_post[a:b]], post[order_pre[c:d]]])
        ws = np.concatenate([w[order_post[a:b]], w[order_pre[c:d]]])
        return ids, ws

    for ct in PROPAGATION_ORDER:
        for i in ann.select(ct):
            ids, ws = partners(int(i))
            votes: dict[tuple[str, int], float] = {}
            for j, wt in zip(ids, ws):
                key = assigned.get(int(j))
                if key is not None:
                    votes[key] = votes.get(key, 0.0) + float(wt)
            if votes:
                assigned[int(i)] = max(votes, key=votes.get)
    return assigned


def build_eye_columns(cn: Connectome, ann: Annotations) -> EyeColumns:
    right = load_right_lattice()
    right["idx"] = cn.index_of(right["root_id"].to_numpy())
    left = _mirror_left_mi1(ann, right)

    seed = {int(i): ("R", int(c)) for i, c in zip(right["idx"], right["col"])}
    seed.update({int(i): ("L", int(c)) for i, c in zip(left["idx"], left["col"])})
    assigned = _propagate(cn, ann, seed)

    cells = pd.DataFrame(
        [(i, ann.df.at[i, "cell_type"], e, c) for i, (e, c) in assigned.items()],
        columns=["idx", "cell_type", "eye", "col"],
    )
    cells = cells[cells["cell_type"].isin(COLUMN_TYPES)].reset_index(drop=True)

    cols = []
    for eye, lat in (("R", right), ("L", left)):
        for _, r in lat.iterrows():
            cols.append((eye, int(r["col"]), int(r["p"]), int(r["q"])))
    columns = pd.DataFrame(cols, columns=["eye", "col", "p", "q"])
    # Hex lattice -> angles. posterior = q - p, dorsal = p + q (see module docstring).
    h = (columns["q"] - columns["p"]) * np.cos(np.radians(30))
    v = (columns["p"] + columns["q"]) * np.sin(np.radians(30))
    az = 90.0 + (h - h.mean()) * INTEROMMATIDIAL_DEG
    el = (v - v.mean()) * INTEROMMATIDIAL_DEG
    columns["az_deg"] = np.where(columns["eye"] == "R", az, -az)
    columns["el_deg"] = el
    return EyeColumns(columns, cells)


def projection_receptive_fields(
    cn: Connectome,
    ann: Annotations,
    ec: EyeColumns,
    cell_types: tuple[str, ...] = ("LC4", "LPLC2"),
    input_types: tuple[str, ...] = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"),
) -> pd.DataFrame:
    """Receptive-field centre and width of visual projection neurons.

    Each LC / LPLC neuron collects from T4/T5 (and other columnar cells) over a patch of
    the lobula; the synapse-weighted mean of its input columns' viewing directions gives
    the RF centre, their spread its width. Returns one row per neuron: ``idx``,
    ``cell_type``, ``eye``, ``az_deg``, ``el_deg``, ``sigma_deg``, ``n_inputs``.
    """
    key = ec.columns.reset_index().set_index(["eye", "col"])["index"]
    cells = ec.cells[ec.cells["cell_type"].isin(input_types)]
    col_row = dict(zip(cells["idx"], key.loc[list(zip(cells["eye"], cells["col"]))].to_numpy()))
    az = ec.columns["az_deg"].to_numpy()
    el = ec.columns["el_deg"].to_numpy()

    targets = ann.select(list(cell_types))
    m = np.isin(cn.post, targets) & np.isin(cn.pre, cells["idx"].to_numpy())
    pre, post, w = cn.pre[m], cn.post[m], np.abs(cn.weight[m])
    rows = []
    for t in targets:
        sel = post == t
        if not sel.any():
            continue
        r = np.array([col_row[int(p)] for p in pre[sel]])
        ww = w[sel]
        a = np.average(az[r], weights=ww)
        e = np.average(el[r], weights=ww)
        spread = np.sqrt(np.average((az[r] - a) ** 2 + (el[r] - e) ** 2, weights=ww))
        rows.append(
            (int(t), ann.df.at[t, "cell_type"], "R" if ann.df.at[t, "side"] == "right" else "L",
             float(a), float(e), float(max(spread, 5.0)), int(sel.sum()))
        )
    return pd.DataFrame(
        rows, columns=["idx", "cell_type", "eye", "az_deg", "el_deg", "sigma_deg", "n_inputs"]
    )


def load_eye_columns(cn: Connectome, data_dir: Path | str = DATA_DIR, cache: bool = True) -> EyeColumns:
    path = Path(data_dir) / "eye_columns"
    if cache and path.with_suffix(".columns.parquet").exists():
        return EyeColumns.load(path)
    ann = Annotations(cn, data_dir)
    ec = build_eye_columns(cn, ann)
    if cache:
        ec.save(path)
    return ec
