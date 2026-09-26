"""Descending-neuron spikes -> Mario button presses.

Maps spike counts from named descending-neuron (DN) groups onto the
`SIMPLE_MOVEMENT` action set of gym-super-mario-bros, using DN behavioral
roles established in the fly literature (see docs/PLAN.md section 6):

    right        <- DNp09 ("P9"), drives forward walking
    left         <- MDN ("moonwalker"), drives backward walking
    A (jump)     <- DNp01 (giant fiber), triggers an escape jump
    B (run)      <- elevated overall forward-DN rate (faster walking)
    turn tie-break <- DNa01 / DNa02

This module works on named spike-COUNT dicts rather than raw neuron indices,
so it's decoupled from exactly where those DNs live in the full connectome
(that mapping lives in `connectome/cell_types.py`).
"""
from __future__ import annotations

import numpy as np

# Matches gym_super_mario_bros.actions.SIMPLE_MOVEMENT order.
SIMPLE_MOVEMENT = [
    ["NOOP"],
    ["right"],
    ["right", "A"],
    ["right", "B"],
    ["right", "A", "B"],
    ["A"],
    ["left"],
]


class SpikeToAction:
    def __init__(
        self,
        baseline_hz: dict[str, float] | None = None,
        run_threshold_hz: float = 40.0,
        jump_threshold_hz: float = 30.0,
    ):
        self.baseline_hz = baseline_hz or {
            "dn_forward": 5.0, "dn_backward": 5.0, "dn_jump": 5.0, "dn_turn": 5.0,
        }
        self.run_threshold_hz = run_threshold_hz
        self.jump_threshold_hz = jump_threshold_hz

    def decode(self, dn_rates_hz: dict[str, float]) -> int:
        """`dn_rates_hz`: {"dn_forward": hz, "dn_backward": hz, "dn_jump": hz,
        "dn_turn": hz} measured over the current decision window. Returns an
        index into SIMPLE_MOVEMENT."""
        fwd = dn_rates_hz.get("dn_forward", 0.0) - self.baseline_hz["dn_forward"]
        bwd = dn_rates_hz.get("dn_backward", 0.0) - self.baseline_hz["dn_backward"]
        jump = dn_rates_hz.get("dn_jump", 0.0)
        run = fwd  # sustained high forward-DN rate doubles as "run"

        want_jump = jump > self.jump_threshold_hz
        want_run = run > self.run_threshold_hz
        want_left = bwd > fwd and bwd > 0
        want_right = fwd > 0 and fwd >= bwd

        if want_left:
            return SIMPLE_MOVEMENT.index(["left"])
        if want_right and want_jump and want_run:
            return SIMPLE_MOVEMENT.index(["right", "A", "B"])
        if want_right and want_jump:
            return SIMPLE_MOVEMENT.index(["right", "A"])
        if want_right and want_run:
            return SIMPLE_MOVEMENT.index(["right", "B"])
        if want_right:
            return SIMPLE_MOVEMENT.index(["right"])
        if want_jump:
            return SIMPLE_MOVEMENT.index(["A"])
        return SIMPLE_MOVEMENT.index(["NOOP"])


def spike_counts_to_rate(spike_count: int, window_ms: float) -> float:
    return spike_count / (window_ms / 1000.0)
