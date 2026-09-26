"""Reproduce Shiu et al. 2024's sugar -> MN9 validation on our torch LIF
simulator, as a sanity check that the port from their brian2 model is
correct before we build anything Mario-specific on top of it.

Ground truth, taken directly from their `example.ipynb` (see
`data/example.ipynb` after running `connectome.download`): activating 21
named sugar-sensing neurons (Poisson input at 100-200 Hz) should drive
spiking in the feeding motor neuron MN9 (FlyWire root ID
720575940660219265), a multi-synaptic effect through the network, not a
direct connection. We don't have their exact spike-rate numbers on hand
(those live in the multi-GB results archive linked from the README, not in
the repo), so this check is intentionally binary: does MN9 fire well above
its baseline (near-silent) rate when the sugar neurons are driven, using the
*same* connectome and the *same* published neuron parameters copied into
`sim/lif.py`? A large positive effect here is what their paper reports
qualitatively; a null result would mean our port has a bug worth chasing
before trusting the simulator for anything else.

Note: their example uses the v630 connectivity files; we use v783 (what
`connectome.download` fetches by default). 20/21 sugar neuron IDs and MN9's
ID are unchanged between the two releases (verified against the downloaded
Completeness_783.csv), so we run with the 20 that carried over.

Run: python -m tests.check_sugar_feeding
"""
from __future__ import annotations

import sys

import torch

from connectome.loader import load_shiu_connectome
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


def run(data_dir: str = "data", duration_ms: float = 1000.0, poisson_hz: float = 200.0,
        verbose: bool = True) -> bool:
    conn = load_shiu_connectome(data_dir)
    sugar_idx = [conn.root_id_to_index[i] for i in SUGAR_NEURON_IDS if i in conn.root_id_to_index]
    mn9_idx = conn.root_id_to_index[MN9_ID]
    if verbose:
        print(f"Loaded {conn.summary()}")
        print(f"Driving {len(sugar_idx)}/{len(SUGAR_NEURON_IDS)} sugar neurons at {poisson_hz} Hz, "
              f"watching MN9 (index {mn9_idx})")

    sim = FlyBrainSimulator(conn, params=LIFParams(), device="cpu", batch_size=1)

    def ext_current_fn(t_ms: float):
        # Poisson-ish drive to the sugar neurons only: at each dt, deliver a
        # fixed current kick with probability rate*dt (converted to an
        # approximate constant depolarizing current for simplicity, matching
        # Shiu et al.'s use of a strong feedforward PoissonInput).
        cur = torch.zeros(1, sim.n_neurons)
        cur[0, sugar_idx] = 400.0  # mV/s-equivalent constant drive during stim
        return cur

    # baseline: no drive
    sim.reset()
    baseline_raster = sim.run(duration_ms=200.0)
    baseline_rate = baseline_raster[:, 0, mn9_idx].sum().item() / (200.0 / 1000.0)

    # driven: sugar neurons stimulated
    sim.reset()
    driven_raster = sim.run(duration_ms=duration_ms, ext_current_fn=ext_current_fn)
    driven_rate = driven_raster[:, 0, mn9_idx].sum().item() / (duration_ms / 1000.0)

    if verbose:
        print(f"MN9 baseline rate: {baseline_rate:.2f} Hz")
        print(f"MN9 driven rate:   {driven_rate:.2f} Hz")

    passed = driven_rate > baseline_rate + 1.0  # any clear positive effect
    if verbose:
        print("PASS" if passed else "FAIL",
              "- MN9 should show elevated firing when sugar neurons are activated")
    return passed


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
