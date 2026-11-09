"""Encoder built from the REAL FlyWire v783 mushroom-body input wiring.

Kenyon cells (5,177, from FlyWire's cell_class annotation) take their
sensory drive from antennal-lobe projection neurons (ALPN). This module
extracts the actual ALPN -> KC synapse counts and uses them as a fixed,
feed-forward encoder:

  game input (256 RAM bytes) --random fixed projection--> ALPN activity
  --real ALPN->KC synapse counts--> KC drive --APL-like global inhibition
  (a threshold tied to total drive, calibrated so ~5% of KCs fire)--> sparse KC code

What is real: which KCs exist, which ALPNs wire to which KCs, and the
synapse counts. What is not: Pong is not an odor, so how the game maps onto
ALPNs is an arbitrary fixed random projection (as for any non-olfactory
input), and APL inhibition is a top-k rule, not a simulated neuron. This is
a static feed-forward pass, not LIF dynamics, to keep millions of Pong steps
affordable.

`shuffle=True` is the control: the same ALPN->KC edges with their KC targets
permuted (every ALPN's out-degree and every edge weight preserved, who
connects to whom randomized), to test whether the real wiring matters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from connectome.loader import load_shiu_connectome

ANNOT = "data/Supplemental_file1_neuron_annotations.tsv"
CACHE = "data/mb_alpn_kc.npz"
INHIB_Z = 1.8   # calibrated so about 5% of KCs fire on typical input


def build_alpn_kc(data_dir: str = "data"):
    """Returns W[n_kc, n_pn] of real ALPN->KC synapse counts (+ ids)."""
    try:
        z = np.load(CACHE)
        return z["W"], z["kc_ids"], z["pn_ids"]
    except FileNotFoundError:
        pass
    c = load_shiu_connectome(data_dir)
    a = pd.read_csv(ANNOT, sep="\t", low_memory=False)
    a = a[a.root_id.isin(c.root_id_to_index.keys())]
    idx = a.root_id.map(c.root_id_to_index).to_numpy()
    kind = np.full(c.n_neurons, "", dtype=object)
    kind[idx] = a.cell_class.fillna("").to_numpy()
    kc = np.nonzero(kind == "Kenyon_Cell")[0]
    kc_pos = {k: i for i, k in enumerate(kc)}
    m = (kind[c.pre_idx] == "ALPN") & (kind[c.post_idx] == "Kenyon_Cell")
    pre, post, w = c.pre_idx[m], c.post_idx[m], np.abs(c.signed_weight[m])
    pn = np.unique(pre)
    pn_pos = {p: i for i, p in enumerate(pn)}
    W = np.zeros((len(kc), len(pn)), dtype=np.float32)
    for p, q, x in zip(pre, post, w):
        W[kc_pos[q], pn_pos[p]] += x
    np.savez_compressed(CACHE, W=W, kc_ids=c.root_id[kc], pn_ids=c.root_id[pn])
    return W, c.root_id[kc], c.root_id[pn]


def make_mb_encoder(seed: int, n_in: int, shuffle: bool = False):
    W, _, _ = build_alpn_kc()
    rng = np.random.default_rng(seed)
    n_kc, n_pn = W.shape
    if shuffle:
        r, cidx = np.nonzero(W)
        vals = W[r, cidx]
        W = np.zeros_like(W)
        W[rng.permutation(r), cidx] = vals
    Ws = torch.tensor(W / np.maximum(W.sum(1, keepdims=True), 1e-6)).to_sparse_csr()  # [n_kc, n_pn], each KC averages its inputs
    P = torch.tensor(rng.standard_normal((n_in, n_pn)).astype(np.float32) * 0.5)

    def enc(x):
        pn = torch.relu(x @ P)
        drive = (Ws @ pn.T).T
        # APL-like feedback inhibition: threshold rises with total drive (mean + z*std per sample)
        thr = drive.mean(1, keepdim=True) + INHIB_Z * drive.std(1, keepdim=True)
        kc = torch.relu(drive - thr)
        return kc / (kc.sum(1, keepdim=True) + 1e-6) * 4.0
    return enc, n_kc
