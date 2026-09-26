"""Leaky integrate-and-fire spiking simulator over a sparse connectome, in
torch so it can run on GPU.

Neuron/synapse parameters are taken directly from Shiu et al. 2024's
validated brian2 model (see ``data/model.py`` after running
``connectome.download``), so our simulator should reproduce their published
results as a sanity check before we change anything:

    v_0    = -52 mV   resting potential           (Kakaria & de Bivort 2017)
    v_rst  = -52 mV   reset potential after spike
    v_th   = -45 mV   spike threshold
    t_mbr  =  20 ms   membrane time constant
    tau    =   5 ms   synaptic conductance decay  (Juergensen et al.)
    t_rfc  = 2.2 ms   refractory period            (Lazar et al.)
    t_dly  = 1.8 ms   synaptic transmission delay  (Paul et al. 2015)
    w_syn  = 0.275 mV weight per synapse (free parameter, fit by Shiu et al.)

Membrane equation (alpha-synapse form):
    dv/dt = (v_0 - v + g) / t_mbr
    dg/dt = -g / tau
    spike when v > v_th; on spike: v <- v_rst, g <- 0

We integrate this with simple forward Euler at dt=0.1ms (well under both time
constants) rather than brian2's exact linear solver, since we need a
`torch`/GPU-batchable implementation. Validate against their published
sugar->MN9 result (`tests/check_sugar_feeding.py`) before trusting the port.

We do NOT implement the synaptic transmission delay (t_dly) as a genuine
per-synapse delay queue (that's expensive to batch); instead the whole
network step already runs at dt=0.1ms and conductance changes are applied to
`g` one step after a presynaptic spike, giving an effective delay of one
dt (~0.1ms) rather than 1.8ms. This is a known simplification — flag it if
you see timing-sensitive behavior (e.g. coincidence detection) misbehave.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import torch

from connectome.loader import Connectome


@dataclasses.dataclass
class LIFParams:
    v_0: float = -52.0      # mV
    v_rst: float = -52.0    # mV
    v_th: float = -45.0     # mV
    t_mbr: float = 20.0     # ms
    tau: float = 5.0        # ms
    t_rfc: float = 2.2      # ms
    w_syn: float = 0.275    # mV per unit signed synapse weight
    dt: float = 0.1         # ms, integration step


class FlyBrainSimulator:
    """Batched LIF simulator: state shape is [batch, n_neurons] so many
    independent Mario episodes (or Monte Carlo repeats) can run at once.
    """

    def __init__(
        self,
        connectome: Connectome,
        params: LIFParams | None = None,
        device: str | torch.device = "cpu",
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        self.params = params or LIFParams()
        self.device = torch.device(device)
        self.n_neurons = connectome.n_neurons
        self.batch_size = batch_size
        self.dtype = dtype

        # Sparse weight matrix W[post, pre] = signed_weight * w_syn, so that
        # g_post += (W @ spikes_pre) reproduces `syn.w = Excitatory *
        # Connectivity * w_syn`, `on_pre: g += w` from Shiu et al.
        idx = torch.tensor(
            np.stack([connectome.post_idx, connectome.pre_idx]), dtype=torch.long
        )
        vals = torch.tensor(connectome.signed_weight * self.params.w_syn, dtype=dtype)
        # CSR is substantially faster than COO for repeated sparse @ dense
        # matmuls on CPU (COO mm effectively falls back to a slow path) —
        # this matters a lot at whole-connectome scale (15M nonzeros).
        self.W = (
            torch.sparse_coo_tensor(idx, vals, size=(self.n_neurons, self.n_neurons))
            .coalesce()
            .to_sparse_csr()
            .to(self.device)
        )

        self.reset()

    def reset(self, batch_size: int | None = None) -> None:
        if batch_size is not None:
            self.batch_size = batch_size
        shape = (self.batch_size, self.n_neurons)
        p = self.params
        self.v = torch.full(shape, p.v_0, device=self.device, dtype=self.dtype)
        self.g = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.refrac_remaining = torch.zeros(shape, device=self.device, dtype=self.dtype)
        self.spikes = torch.zeros(shape, device=self.device, dtype=self.dtype)

    def step(self, ext_current: torch.Tensor | None = None) -> torch.Tensor:
        """Advance one dt. `ext_current` is [batch, n_neurons] in the same
        units as `g` (mV-ish, since this is a conductance-free simplified
        alpha-synapse model as in Shiu et al.). Returns spikes [batch, n]."""
        p = self.params
        if ext_current is None:
            ext_current = 0.0

        # synaptic input from the previous step's spikes, via the connectome
        syn_input = torch.sparse.mm(self.W, self.spikes.T).T  # [batch, n]

        self.g = self.g + syn_input
        dv = (p.v_0 - self.v + self.g) / p.t_mbr * p.dt
        dg = -self.g / p.tau * p.dt

        not_refractory = self.refrac_remaining <= 0
        self.v = torch.where(not_refractory, self.v + dv + ext_current * p.dt, self.v)
        self.g = self.g + dg

        spiked = (self.v > p.v_th) & not_refractory
        self.v = torch.where(spiked, torch.full_like(self.v, p.v_rst), self.v)
        self.g = torch.where(spiked, torch.zeros_like(self.g), self.g)

        self.refrac_remaining = torch.where(
            spiked, torch.full_like(self.refrac_remaining, p.t_rfc),
            torch.clamp(self.refrac_remaining - p.dt, min=0.0),
        )

        self.spikes = spiked.to(self.dtype)
        return self.spikes

    def run(self, duration_ms: float, ext_current_fn=None) -> torch.Tensor:
        """Run for `duration_ms`, returns spike raster [T, batch, n_neurons].
        `ext_current_fn(t_ms) -> tensor[batch, n]` optional time-varying input.
        """
        n_steps = int(round(duration_ms / self.params.dt))
        raster = torch.zeros(
            n_steps, self.batch_size, self.n_neurons, device=self.device, dtype=self.dtype
        )
        for t in range(n_steps):
            cur = ext_current_fn(t * self.params.dt) if ext_current_fn else None
            raster[t] = self.step(cur)
        return raster
