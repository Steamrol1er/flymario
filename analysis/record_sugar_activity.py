"""Run the sugar->MN9 experiment on a 2-hop subcircuit and record per-neuron,
per-time-bin spike activity, for visualization (see analysis/build_viz_data.py).

This reuses the same real connectome, real neuron IDs, and real validated LIF
parameters as tests/check_sugar_feeding.py, restricted to the 2-hop
subcircuit around the sugar-sensing neurons (14,450 of 138,639 neurons) so
the run finishes in a reasonable time while still containing MN9 and the
real multi-synaptic pathway to it.

Run: python -m analysis.record_sugar_activity
Writes: analysis/sugar_activity.npz
"""
from __future__ import annotations

import time

import numpy as np
import torch

from connectome.loader import load_shiu_connectome
from connectome.subcircuit import expand_by_hops
from sim.lif import FlyBrainSimulator, LIFParams

SUGAR_NEURON_IDS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9_ID = 720575940660219265

BIN_MS = 10.0          # time resolution for the activity "movie"
DURATION_MS = 300.0    # keep it short so the JS visualization stays snappy
HOPS = 2


def main():
    conn = load_shiu_connectome("data")
    sugar_idx_full = np.array(
        [conn.root_id_to_index[i] for i in SUGAR_NEURON_IDS if i in conn.root_id_to_index]
    )
    mn9_idx_full = conn.root_id_to_index[MN9_ID]

    print("Building 2-hop subcircuit around sugar neurons ...")
    t0 = time.time()
    keep = expand_by_hops(conn, sugar_idx_full, hops=HOPS)
    sub = conn.subcircuit(keep)
    print(f"  {sub.n_neurons} neurons, {len(sub.pre_idx)} edges ({time.time()-t0:.1f}s)")

    # map original sugar/MN9 indices into the subcircuit's re-indexed space
    old_to_new = {int(old): new for new, old in enumerate(keep)}
    sugar_idx = np.array([old_to_new[i] for i in sugar_idx_full.tolist()])
    mn9_idx = old_to_new[mn9_idx_full]

    sim = FlyBrainSimulator(sub, params=LIFParams(), device="cpu", batch_size=1)

    def ext_current_fn(t_ms: float):
        cur = torch.zeros(1, sim.n_neurons)
        cur[0, sugar_idx] = 400.0
        return cur

    print(f"Running {DURATION_MS}ms driven simulation on subcircuit ...")
    t0 = time.time()
    raster = sim.run(duration_ms=DURATION_MS, ext_current_fn=ext_current_fn)  # [T, 1, n]
    print(f"  done in {time.time()-t0:.1f}s")

    raster = raster[:, 0, :].numpy()  # [T, n_neurons], dt=0.1ms per row
    n_steps_per_bin = int(round(BIN_MS / sim.params.dt))
    n_bins = raster.shape[0] // n_steps_per_bin
    binned = raster[: n_bins * n_steps_per_bin].reshape(n_bins, n_steps_per_bin, -1).sum(axis=1)
    # binned: [n_bins, n_neurons] spike COUNT per BIN_MS window

    total_spikes = binned.sum(axis=0)
    active_mask = total_spikes > 0
    print(f"  {active_mask.sum()} / {sub.n_neurons} neurons fired at least once "
          f"(sugar-driven, {DURATION_MS}ms window)")

    np.savez_compressed(
        "analysis/sugar_activity.npz",
        root_id=sub.root_id,             # [n_neurons] FlyWire root ID per subcircuit index
        pre_idx=sub.pre_idx,             # [n_edges] subcircuit-local indices
        post_idx=sub.post_idx,
        signed_weight=sub.signed_weight,
        binned_spikes=binned.astype(np.int16),  # [n_bins, n_neurons]
        bin_ms=BIN_MS,
        sugar_idx=sugar_idx,
        mn9_idx=mn9_idx,
    )
    print("Wrote analysis/sugar_activity.npz")


if __name__ == "__main__":
    main()
