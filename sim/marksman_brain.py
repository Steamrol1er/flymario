"""The untrained LIF controller for the Fly Marksman reproduction: turns
ViZDoom enemy bearings into LC10a Poisson drive, runs it through the real
extracted connectome (connectome/marksman_circuit.py) via our existing
simulator (sim/lif.py), and reads back AOTU019/DNa02 firing rates. No
learning anywhere in this file -- the whole point is that the WIRING alone
does the work (see experiments/run_marksman.py for the fixed-threshold
readout and the fly-vs-scrambled comparison this feeds).

This independently reimplements webergithub/fruitfly-lab's fly_doom.py
control loop (same published method: LC10a eccentricity tuning, Poisson
stimulation, per-Doom-tic LIF integration, exponentially smoothed rates)
against OUR OWN circuit extraction and OUR OWN simulator, not their code.
One real difference worth flagging: their simulator takes a stimulation
*rate* directly; ours (sim/lif.py) takes an external *current* per 0.1ms
step, since that's the interface the rest of this project already built and
validated against Shiu et al.'s sugar-feeding result. So here a target
firing rate is converted to independent per-step Poisson spikes
(probability = rate * dt), each of which injects one EXT_KICK_MV membrane
kick -- a single free parameter (tuned, not fit) so that MAX_RATE drives
LC10a to a strong, clearly-saturating response. Because the downstream
readout thresholds are calibrated from OUR OWN circuit's response curve
(experiments/run_marksman.py's calibrate()), this parameter only needs to
give a sane, monotonic azimuth-to-steering tuning curve, not to numerically
match their internal rate scale.
"""
from __future__ import annotations

import numpy as np
import torch

from connectome.marksman_circuit import MarksmanCircuit
from sim.lif import FlyBrainSimulator, LIFParams

DT_MS = 0.1                      # matches sim/lif.py's validated integration step
STEPS_PER_TIC = round(1000 / 35 / DT_MS)   # 286 steps per Doom tic (1/35 s)
HALF_FOV_DOOM = 45.0              # Doom's horizontal FOV is 90 deg (+/-45)
FLY_SCALE = 2.0                   # map screen +/-45 deg onto the fly's +/-90 deg visual field
BINO = 6.0                        # binocular overlap (fly-frame degrees): both eyes driven
MAX_RATE = 150.0                  # LC10a peak stimulation rate, Hz
SIGMA = 0.3                       # eccentricity tuning width (unitless, matches pref_eccentricity scale)
TAU_S = 0.1                       # rate-smoothing time constant, seconds
EXT_KICK_MV = 6.0                 # engineered: membrane kick per injected external spike (see docstring)

PANEL = ["LC10a_L", "LC10a_R", "AOTU019_L", "AOTU019_R",
         "AOTU025_L", "AOTU025_R", "DNa02_L", "DNa02_R"]


class FlyBrain:
    """One running instance of the circuit, with exponentially-smoothed
    firing rates for the 8-neuron-group readout panel."""

    def __init__(self, circuit: MarksmanCircuit, seed: int = 0,
                 silenced: np.ndarray | None = None, device: str = "cpu",
                 kick_mv: float | None = None):
        self.circuit = circuit
        self.kick_mv = EXT_KICK_MV if kick_mv is None else kick_mv
        self.rng = np.random.default_rng(seed)
        self.sim = FlyBrainSimulator(circuit.connectome, params=LIFParams(),
                                      device=device, batch_size=1)
        n = circuit.connectome.n_neurons
        self.silenced_mask = np.zeros(n, dtype=bool)
        if silenced is not None and len(silenced) > 0:
            self.silenced_mask[np.asarray(silenced, dtype=np.int64)] = True

        self.lc = {"L": circuit.groups["LC10a_L"], "R": circuit.groups["LC10a_R"]}
        self.pref_ecc = circuit.pref_eccentricity
        self.panel_idx = {k: circuit.groups[k] for k in PANEL}

        self.alpha = float(np.exp(-(1.0 / 35.0) / TAU_S))
        self.rate = {k: 0.0 for k in PANEL}

    def stim_rates(self, targets: list[tuple[float, float]]) -> np.ndarray:
        """targets: [(azimuth_deg, strength_0_to_1), ...] -> per-neuron Hz."""
        n = self.circuit.connectome.n_neurons
        rates = np.zeros(n, dtype=np.float32)
        for az, strength in targets:
            ecc = min(abs(az) / 90.0, 1.0)
            sides = ["L", "R"] if abs(az) < BINO else (["R"] if az > 0 else ["L"])
            for s in sides:
                ids = self.lc[s]
                tune = np.exp(-((self.pref_ecc[ids] - ecc) ** 2) / (2 * SIGMA ** 2))
                rates[ids] = np.minimum(MAX_RATE, rates[ids] + MAX_RATE * strength * tune)
        return rates

    def step(self, targets: list[tuple[float, float]]) -> dict[str, float]:
        """Advance one Doom tic (286 LIF steps), return updated panel rates."""
        rates = self.stim_rates(targets)
        rates[self.silenced_mask] = 0.0
        spike_prob = rates * (DT_MS / 1000.0)

        n = self.circuit.connectome.n_neurons
        counts = np.zeros(n, dtype=np.float64)
        for _ in range(STEPS_PER_TIC):
            ext_spikes = self.rng.random(n) < spike_prob
            ext_current = torch.zeros(1, n, dtype=self.sim.dtype, device=self.sim.device)
            if ext_spikes.any():
                ext_current[0, ext_spikes] = self.kick_mv / self.sim.params.dt
            spiked = self.sim.step(ext_current)
            if self.silenced_mask.any():
                # lesion: force silenced neurons to never spike, so they
                # can't drive anything downstream either
                spiked[0, self.silenced_mask] = 0.0
                self.sim.spikes[0, self.silenced_mask] = 0.0
            counts += spiked[0].cpu().numpy()

        hz = counts / (STEPS_PER_TIC * DT_MS / 1000.0)
        for k, ids in self.panel_idx.items():
            self.rate[k] = self.alpha * self.rate[k] + (1 - self.alpha) * float(hz[ids].mean())
        return dict(self.rate)


def steer_signal(r: dict[str, float]) -> float:
    """Positive => turn right. DNa02 is the descending steering command;
    AOTU019 is read too because in this subcircuit DNa02 has no baseline
    firing to disinhibit off of, so AOTU019's own L/R difference carries
    the central-visual-field steering signal that DNa02 can't express here."""
    return (r["DNa02_R"] - r["DNa02_L"]) + (r["AOTU019_R"] - r["AOTU019_L"])
