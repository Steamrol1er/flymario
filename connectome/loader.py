"""Load the Shiu et al. 2024 connectivity tables into a sparse weight matrix.

Data layout (verified against the actual downloaded files, 2026-09-26):

``Completeness_783.csv``
    Index column = dense neuron index (0..N-1) used everywhere else.
    Column "Unnamed: 0" (we rename to ``root_id``) = FlyWire root ID (int64).
    Column "Completed" = bool.
    138,639 neurons total in the v783 release.

``Connectivity_783.parquet``
    One row per (pre, post) neuron pair with at least one synapse (already
    aggregated — not one row per synapse). Columns:
      Presynaptic_ID / Postsynaptic_ID   FlyWire root IDs
      Presynaptic_Index / Postsynaptic_Index   dense indices into Completeness
      Connectivity                        synapse count for this pair
      Excitatory                          +1 (ACh-like/excitatory) or -1
                                           (GABA/glutamate-like/inhibitory)
      Excitatory x Connectivity            Excitatory * Connectivity — this
                                           is exactly the signed synapse
                                           count Shiu et al. use as the
                                           weight (see their model.py:
                                           ``syn.w = df_con['Excitatory x
                                           Connectivity'] * w_syn``).
    15,091,983 pre/post pairs in v783.

We reuse their sign convention directly rather than re-deriving it from
neurotransmitter predictions, since it's already been validated against real
experiments in their paper.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclasses.dataclass
class Connectome:
    """Sparse fly connectome, in COO form (framework-agnostic — no torch
    dependency here so this module can be tested / inspected without a
    torch install)."""

    n_neurons: int
    pre_idx: np.ndarray          # int64[n_edges]
    post_idx: np.ndarray         # int64[n_edges]
    signed_weight: np.ndarray    # float32[n_edges] = Excitatory * Connectivity
    root_id: np.ndarray          # int64[n_neurons], index -> FlyWire root ID
    root_id_to_index: dict       # FlyWire root ID -> dense index
    completed: np.ndarray        # bool[n_neurons]

    def subcircuit(self, keep_indices: np.ndarray) -> "Connectome":
        """Return a new Connectome restricted to `keep_indices`, with edges
        re-indexed into the smaller neuron set. Edges where either endpoint
        is dropped are removed."""
        keep_indices = np.asarray(keep_indices, dtype=np.int64)
        old_to_new = np.full(self.n_neurons, -1, dtype=np.int64)
        old_to_new[keep_indices] = np.arange(len(keep_indices))

        new_pre = old_to_new[self.pre_idx]
        new_post = old_to_new[self.post_idx]
        mask = (new_pre >= 0) & (new_post >= 0)

        return Connectome(
            n_neurons=len(keep_indices),
            pre_idx=new_pre[mask],
            post_idx=new_post[mask],
            signed_weight=self.signed_weight[mask],
            root_id=self.root_id[keep_indices],
            root_id_to_index={rid: i for i, rid in enumerate(self.root_id[keep_indices])},
            completed=self.completed[keep_indices],
        )

    def summary(self) -> str:
        n_exc = int((self.signed_weight > 0).sum())
        n_inh = int((self.signed_weight < 0).sum())
        return (
            f"Connectome: {self.n_neurons} neurons, {len(self.pre_idx)} edges "
            f"({n_exc} excitatory, {n_inh} inhibitory), "
            f"{self.completed.sum()}/{self.n_neurons} marked complete"
        )


def load_shiu_connectome(data_dir: Path | str = "data") -> Connectome:
    data_dir = Path(data_dir)
    comp_path = data_dir / "Completeness_783.csv"
    con_path = data_dir / "Connectivity_783.parquet"
    if not comp_path.exists() or not con_path.exists():
        raise FileNotFoundError(
            f"Expected {comp_path} and {con_path}. Run:\n"
            f"  python -m connectome.download --dest {data_dir}"
        )

    comp = pd.read_csv(comp_path)
    comp = comp.rename(columns={comp.columns[0]: "root_id"})
    n_neurons = len(comp)
    root_id = comp["root_id"].to_numpy(dtype=np.int64)
    completed = comp["Completed"].to_numpy(dtype=bool)
    root_id_to_index = {int(rid): i for i, rid in enumerate(root_id)}

    con = pd.read_parquet(
        con_path,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre_idx = con["Presynaptic_Index"].to_numpy(dtype=np.int64)
    post_idx = con["Postsynaptic_Index"].to_numpy(dtype=np.int64)
    signed_weight = con["Excitatory x Connectivity"].to_numpy(dtype=np.float32)

    return Connectome(
        n_neurons=n_neurons,
        pre_idx=pre_idx,
        post_idx=post_idx,
        signed_weight=signed_weight,
        root_id=root_id,
        root_id_to_index=root_id_to_index,
        completed=completed,
    )


def load_toy_connectome(seed: int = 0) -> tuple[Connectome, dict]:
    """A tiny synthetic connectome (no download needed) used for fast unit
    tests of the simulator and learning rule, independent of the real data.

    Layout: 2 "odor" input populations -> shared KC-like layer -> 2 MBON-like
    output neurons (approach / avoid) -> mimics the mushroom body circuit at
    toy scale for the M1 conditioning sanity check.

    Returns (connectome, groups) where groups maps region name -> neuron
    index array.
    """
    rng = np.random.default_rng(seed)
    n_odor, n_kc, n_mbon = 20, 200, 2
    n_neurons = 2 * n_odor + n_kc + n_mbon
    odor_a = np.arange(0, n_odor)
    odor_b = np.arange(n_odor, 2 * n_odor)
    kc = np.arange(2 * n_odor, 2 * n_odor + n_kc)
    mbon = np.arange(2 * n_odor + n_kc, n_neurons)

    pre, post, w = [], [], []

    # sparse random odor -> KC (each KC samples ~6 random PNs, like real MB)
    for k in kc:
        srcs = rng.choice(np.concatenate([odor_a, odor_b]), size=6, replace=False)
        for s in srcs:
            pre.append(s); post.append(k); w.append(1.0)

    # all KCs -> both MBONs, initially equal weight (plasticity acts here)
    for k in kc:
        for m in mbon:
            pre.append(k); post.append(m); w.append(1.0)

    root_id = np.arange(n_neurons, dtype=np.int64)
    return Connectome(
        n_neurons=n_neurons,
        pre_idx=np.array(pre, dtype=np.int64),
        post_idx=np.array(post, dtype=np.int64),
        signed_weight=np.array(w, dtype=np.float32),
        root_id=root_id,
        root_id_to_index={int(r): int(r) for r in root_id},
        completed=np.ones(n_neurons, dtype=bool),
    ), {"odor_a": odor_a, "odor_b": odor_b, "kc": kc, "mbon": mbon}
