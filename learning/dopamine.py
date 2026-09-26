"""Three-factor dopamine learning rule at KC -> MBON synapses.

Biology this is modeling (see Hige et al. 2015, Aso et al. 2014, Eschbach
et al. 2020): Kenyon cells (KCs) form a large, sparse, roughly random code
for whatever sensory situation the fly is in. Each KC synapses onto a small
number of mushroom body output neurons (MBONs), organized into
"compartments", each innervated by a distinct dopamine neuron (DAN)
population. When a DAN fires *while* a KC is active (coincidence), the
KC->MBON synapse for that pathway is **depressed** (weakened) — not
strengthened. Untouched synapses stay at their baseline (naive) weight.

Two DAN classes with opposite valence:
  PAM  neurons -> "reward" compartments (e.g. gamma4, gamma5, beta'2)
  PPL1 neurons -> "punishment" compartments (e.g. gamma1, gamma2, alpha'2)

Net effect after learning: for a rewarded stimulus, KC->avoid-MBON synapses
get depressed (less avoidance drive); for a punished stimulus, KC->approach-
MBON synapses get depressed (less approach drive). The behavioral output is
read out from the *balance* of approach vs. avoid MBON activity, which is
exactly what `MushroomBody.value()` below computes.

This module is deliberately kept independent of `sim.lif` internals: it
only needs (a) presynaptic KC spike/rate vectors, (b) per-compartment
dopamine drive, and (c) the weight matrix slice it's allowed to modify. That
keeps the "faithful" learning rule cleanly separated from the "engineering"
actor-critic layer in `learning/actor.py`.
"""
from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class DopamineParams:
    eta: float = 0.01          # learning rate
    trace_decay: float = 0.95  # eligibility trace decay per learning step
    w_min: float = 0.0
    w_max: float = 2.0         # relative to naive weight of 1.0


class MushroomBodyLearner:
    """Holds KC->MBON weights for a set of compartments and applies the
    three-factor depression rule.

    compartments: dict[name -> valence] where valence is +1 for a
    "reward"/PAM-driven (approach) compartment or -1 for a
    "punishment"/PPL1-driven (avoid) compartment. Each compartment gets its
    own [n_kc] weight vector (all KCs -> that compartment's MBON), matching
    the real anatomy where a KC synapses onto MBON dendrites separately in
    each compartment it passes through.
    """

    def __init__(
        self,
        n_kc: int,
        compartments: dict[str, int],
        params: DopamineParams | None = None,
        device: str | torch.device = "cpu",
    ):
        self.n_kc = n_kc
        self.compartments = compartments
        self.params = params or DopamineParams()
        self.device = torch.device(device)

        self.weights = {
            name: torch.ones(n_kc, device=self.device)
            for name in compartments
        }
        self.eligibility = {
            name: torch.zeros(n_kc, device=self.device)
            for name in compartments
        }

    def reset_eligibility(self) -> None:
        for name in self.eligibility:
            self.eligibility[name].zero_()

    def step(self, kc_spikes: torch.Tensor) -> None:
        """Update eligibility traces from this timestep's KC activity.
        `kc_spikes`: [n_kc] binary or rate vector."""
        for name in self.eligibility:
            self.eligibility[name].mul_(self.params.trace_decay).add_(kc_spikes)

    def apply_dopamine(self, dopamine: dict[str, float]) -> None:
        """Apply a dopamine pulse per compartment. `dopamine[name]` is a
        non-negative scalar DAN drive for that compartment (PAM or PPL1
        firing rate above baseline, for example). Depresses KC->MBON
        synapses for KCs with nonzero eligibility, proportional to
        eligibility * dopamine. Depression is universal regardless of
        compartment valence — the valence only determines what the
        compartment *means* behaviorally (see `value()`), not the sign of
        the plasticity rule, matching Hige et al. 2015."""
        p = self.params
        for name, da in dopamine.items():
            if name not in self.weights or da == 0:
                continue
            dw = -p.eta * da * self.eligibility[name]
            self.weights[name] = torch.clamp(self.weights[name] + dw, p.w_min, p.w_max)

    def mbon_output(self, kc_spikes: torch.Tensor) -> dict[str, torch.Tensor]:
        """Instantaneous MBON drive per compartment = sum_k w[k] * kc_spikes[k]."""
        return {name: (self.weights[name] * kc_spikes).sum() for name in self.weights}

    def value(self, kc_spikes: torch.Tensor) -> torch.Tensor:
        """Scalar value estimate = sum over approach compartments minus sum
        over avoid compartments of MBON drive. This is the "critic" signal
        used by the actor-critic layer, and is the most direct behavioral
        readout of what the mushroom body has learned."""
        out = self.mbon_output(kc_spikes)
        total = torch.zeros((), device=self.device)
        for name, valence in self.compartments.items():
            total = total + valence * out[name]
        return total
