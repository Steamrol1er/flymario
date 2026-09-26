"""Extract a fast-to-simulate subcircuit from the full connectome.

The full v783 connectome (138,639 neurons, ~15M pre/post pairs) is too slow
to step through thousands of times per Mario decision across thousands of
training episodes. Per docs/PLAN.md section 3, we keep:

  - a seed set of functionally relevant neurons (mushroom body KCs/MBONs/
    DANs, descending neurons, and whatever projection neurons carry our
    Stage A sensory input; see connectome/cell_types.py)
  - everyone within `hops` synapses of that seed set, in either direction

which typically brings a whole-brain model down to ~10-25k neurons for the
regions we actually need, while leaving the multi-synaptic pathways between
sensory input and motor output intact.
"""
from __future__ import annotations

import numpy as np

from connectome.loader import Connectome


def expand_by_hops(conn: Connectome, seed_indices: np.ndarray, hops: int = 2) -> np.ndarray:
    """Return the sorted array of neuron indices reachable from
    `seed_indices` within `hops` directed edges (either as source or
    target), including the seeds themselves."""
    frontier = set(int(i) for i in seed_indices)
    visited = set(frontier)

    # Build adjacency once as python dict-of-sets for BFS. For 15M edges this
    # is a few seconds and a few hundred MB — fine as a one-off preprocessing
    # step, not something done per-training-step.
    fwd: dict[int, list[int]] = {}
    bwd: dict[int, list[int]] = {}
    for pre, post in zip(conn.pre_idx.tolist(), conn.post_idx.tolist()):
        fwd.setdefault(pre, []).append(post)
        bwd.setdefault(post, []).append(pre)

    for _ in range(hops):
        next_frontier = set()
        for n in frontier:
            next_frontier.update(fwd.get(n, ()))
            next_frontier.update(bwd.get(n, ()))
        next_frontier -= visited
        visited |= next_frontier
        frontier = next_frontier
        if not frontier:
            break

    return np.array(sorted(visited), dtype=np.int64)


def build_subcircuit(conn: Connectome, groups: dict[str, list[int]], hops: int = 2) -> Connectome:
    """Convenience wrapper: seed = union of all indices in `groups`
    (typically kenyon_cell, mbon, pam_dan, ppl1_dan, dn_* from
    connectome.cell_types.resolve_groups), expand by `hops`, and return the
    resulting sub-Connectome (re-indexed, see Connectome.subcircuit)."""
    seed = np.unique(np.concatenate([np.asarray(v, dtype=np.int64) for v in groups.values() if len(v)]))
    keep = expand_by_hops(conn, seed, hops=hops)
    return conn.subcircuit(keep)
