"""M1 milestone: validate the dopamine three-factor rule on a toy two-odor
conditioning task, independent of the full connectome, the LIF simulator,
and Mario.

Setup, mirroring classic fly olfactory conditioning (e.g. Tully & Quinn
1985-style shock-odor pairing, and the MB circuit model in Aso et al. 2014 /
Hige et al. 2015):

  - Two odors (A and B), each activating a distinct, non-overlapping random
    ~30% of a 200-KC population (Kenyon cells give a sparse random code for
    each odor).
  - Odor A is paired with reward (dopamine into the PAM/"approach"
    compartment). Odor B is unpaired (no dopamine).
  - We test whether, after training, presenting odor A drives higher
    "approach minus avoid" MBON value than odor B.

This only exercises `learning.dopamine.MushroomBodyLearner` directly on
rate-coded KC vectors (no spiking, no torch GPU needed) — it's the fastest
possible check that the learning rule does what it should before adding the
complexity of the spiking simulator or Mario.

Run: python -m tests.toy_conditioning
"""
from __future__ import annotations

import numpy as np
import torch

from learning.dopamine import MushroomBodyLearner, DopamineParams


def make_odor_codes(n_kc: int = 200, sparsity: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    n_active = int(n_kc * sparsity)
    idx_a = rng.choice(n_kc, size=n_active, replace=False)
    remaining = np.setdiff1d(np.arange(n_kc), idx_a)
    idx_b = rng.choice(remaining, size=n_active, replace=False)

    odor_a = torch.zeros(n_kc)
    odor_a[idx_a] = 1.0
    odor_b = torch.zeros(n_kc)
    odor_b[idx_b] = 1.0
    return odor_a, odor_b


def run(n_trials: int = 10, n_kc: int = 200, verbose: bool = True) -> bool:
    odor_a, odor_b = make_odor_codes(n_kc)

    compartments = {"approach": +1, "avoid": -1}
    learner = MushroomBodyLearner(n_kc, compartments, DopamineParams(eta=0.15))

    values_a, values_b = [], []
    for trial in range(n_trials):
        # naive value of each odor before this trial's exposure
        v_a_pre = float(learner.value(odor_a))
        v_b_pre = float(learner.value(odor_b))
        values_a.append(v_a_pre)
        values_b.append(v_b_pre)

        # present odor A -> KC activity -> eligibility -> reward -> PAM dopamine
        learner.reset_eligibility()
        learner.step(odor_a)
        learner.apply_dopamine({"approach": 0.0, "avoid": 1.0})
        # (dopamine depresses KC->MBON synapses for active KCs; delivering it
        # to the AVOID compartment depresses avoid-drive for odor A's KCs,
        # i.e. this is a rewarded/appetitive pairing per Hige et al. 2015's
        # depression-only rule)

        # present odor B -> no dopamine (unpaired control)
        learner.reset_eligibility()
        learner.step(odor_b)
        # no apply_dopamine call: odor B's KC->MBON synapses stay naive

        if verbose:
            print(f"trial {trial:2d}: V(A)={v_a_pre:+.3f}  V(B)={v_b_pre:+.3f}")

    v_a_final = float(learner.value(odor_a))
    v_b_final = float(learner.value(odor_b))
    if verbose:
        print(f"final  : V(A)={v_a_final:+.3f}  V(B)={v_b_final:+.3f}")

    passed = v_a_final > v_b_final and v_a_final > values_a[0]
    if verbose:
        print("PASS" if passed else "FAIL",
              "- odor A (rewarded) should end up valued higher than odor B (unpaired)")
    return passed


if __name__ == "__main__":
    ok = run()
    raise SystemExit(0 if ok else 1)
