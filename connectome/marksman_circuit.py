"""Extract the real LC10a -> AOTU -> DNa visual-pursuit subcircuit (3,963
neurons) from our own FlyWire v783 data.

This reproduces webergithub/fruitfly-lab's "Fly Marksman" circuit -- a real,
untrained visual pursuit pathway that plays ViZDoom's defend_the_center
scenario by nothing but its own wiring (see docs/PONG.md's research
section: real wiring scored 10.2 kills/episode vs. 0.7 for a
degree-preserving scramble). We don't run their code; we rebuild the
circuit ourselves from the primary data, using our own already-validated
simulator (sim/lif.py, whose parameters already match theirs exactly --
both trace back to Shiu et al. 2024).

The neuron list itself -- which 3,963 of FlyWire's 138,639 neurons belong
to this pathway, and their cell-type labels (LC10a, AOTU019, DNa02, ...)
-- comes from `data/marksman_reference/fruitfly_lab_meta.json`, fetched
from their public repo. That's their real annotation work (matching
FlyWire cell types to root IDs), which this project can't redo itself
without a FlyWire Codex account (see docs/STATUS.md) -- we verified all
3,963 of their root IDs exist in our own downloaded Completeness_783.csv
(100% overlap, both being FlyWire v783), then extract the actual synaptic
weights among exactly those neurons from our own Connectivity_783.parquet,
independently of their edges.bin. The wiring data reconstructed this way
IS the real connectome; only the neuron *list* is borrowed.

Run: python -m connectome.marksman_circuit  (prints a summary; import
`load_marksman_circuit` from other scripts)
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from connectome.loader import Connectome, load_shiu_connectome

REFERENCE_META = Path("data/marksman_reference/fruitfly_lab_meta.json")

# The functionally-named groups fly_doom.py's control loop actually reads/
# stimulates (see PANEL in data/marksman_reference/fruitfly_lab_fly_doom.py).
CONTROL_GROUPS = [
    "LC10a_L", "LC10a_R", "AOTU019_L", "AOTU019_R",
    "AOTU025_L", "AOTU025_R", "DNa02_L", "DNa02_R",
]


@dataclasses.dataclass
class MarksmanCircuit:
    connectome: Connectome           # re-indexed to just these 3,963 neurons
    groups: dict[str, np.ndarray]    # e.g. "LC10a_L" -> local indices
    pref_eccentricity: np.ndarray    # [n_neurons], 1 - "central" from their metadata


def load_marksman_circuit(data_dir: str = "data") -> MarksmanCircuit:
    meta = json.loads(REFERENCE_META.read_text())
    neurons = meta["neurons"]
    root_ids = np.array([int(n["id"]) for n in neurons], dtype=np.int64)
    central = np.array([n.get("central", 0.5) for n in neurons], dtype=np.float32)

    full = load_shiu_connectome(data_dir)
    missing = [r for r in root_ids if r not in full.root_id_to_index]
    if missing:
        raise ValueError(f"{len(missing)} of the reference circuit's root IDs "
                          f"aren't in our own connectome data (expected 0)")
    full_indices = np.array([full.root_id_to_index[r] for r in root_ids], dtype=np.int64)

    sub = full.subcircuit(full_indices)
    # `subcircuit` re-indexes 0..N-1 in the order `full_indices` was given,
    # so sub's index i corresponds exactly to neurons[i] / root_ids[i]
    pref_ecc = 1.0 - central

    type_by_root = {int(n["id"]): n["type"] for n in neurons}
    side_by_root = {int(n["id"]): n["side"] for n in neurons}
    groups: dict[str, list[int]] = {g: [] for g in CONTROL_GROUPS}
    for i, r in enumerate(root_ids):
        t, side = type_by_root[int(r)], side_by_root[int(r)]
        key = f"{t}_{'L' if side == 'left' else 'R'}"
        if key in groups:
            groups[key].append(i)
    groups_arr = {k: np.array(v, dtype=np.int64) for k, v in groups.items()}

    for k in CONTROL_GROUPS:
        if len(groups_arr[k]) == 0:
            raise ValueError(f"Group {k} is empty -- check the reference metadata's type/side fields")

    return MarksmanCircuit(connectome=sub, groups=groups_arr, pref_eccentricity=pref_ecc)


def scrambled_copy(circuit: MarksmanCircuit, seed: int = 0) -> MarksmanCircuit:
    """Degree-preserving rewiring: shuffle which pre/post pair each edge's
    weight belongs to, independently within excitatory and inhibitory
    edges, so each neuron's in-degree, out-degree, and the overall sign
    balance are unchanged -- only WHO is connected to whom is randomized.
    This is the "scrambled" control fly_doom.py compares against (real
    wiring: 10.2 kills/episode; scrambled: 0.7), reproduced here as a
    from-scratch implementation of the same concept, not their code."""
    rng = np.random.default_rng(seed)
    conn = circuit.connectome
    pre, post, w = conn.pre_idx.copy(), conn.post_idx.copy(), conn.signed_weight.copy()

    for sign in (1, -1):
        mask = (w > 0) if sign > 0 else (w < 0)
        idx = np.nonzero(mask)[0]
        # shuffle the post-synaptic endpoint among edges of the same sign;
        # preserves each edge's presynaptic source and each sign-class's
        # post-synaptic degree distribution (a standard degree-preserving
        # rewire via a random permutation of one endpoint)
        shuffled_post = post[idx].copy()
        rng.shuffle(shuffled_post)
        post[idx] = shuffled_post

    scrambled_conn = Connectome(
        n_neurons=conn.n_neurons, pre_idx=pre, post_idx=post, signed_weight=w,
        root_id=conn.root_id, root_id_to_index=conn.root_id_to_index, completed=conn.completed,
    )
    return MarksmanCircuit(connectome=scrambled_conn, groups=circuit.groups,
                            pref_eccentricity=circuit.pref_eccentricity)


if __name__ == "__main__":
    c = load_marksman_circuit()
    print(c.connectome.summary())
    for g, idx in c.groups.items():
        print(f"  {g}: {len(idx)} neurons")
