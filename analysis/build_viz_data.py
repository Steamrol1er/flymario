"""Turn analysis/sugar_activity.npz (real simulation output) into a small
JSON file for the brain-activity visualization artifact.

- Restricts to neurons that actually fired (~400 of 14,450) plus the sugar
  seed neurons, since those are what tell the "which parts of the brain
  work" story.
- Computes a hop-distance-from-sugar-neurons layer (via BFS over the
  induced excitatory+inhibitory graph among these neurons) for a left
  (sensory) -> right (motor) layout.
- Emits per-neuron per-time-bin spike activity so the viewer can scrub
  through the actual 300ms simulated response and watch activity spread.

Run: python -m analysis.build_viz_data
Writes: analysis/brain_activity.json
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np


def main():
    d = np.load("analysis/sugar_activity.npz")
    root_id = d["root_id"]
    pre_idx = d["pre_idx"]
    post_idx = d["post_idx"]
    signed_weight = d["signed_weight"]
    binned = d["binned_spikes"]          # [n_bins, n_neurons]
    bin_ms = float(d["bin_ms"])
    sugar_idx = d["sugar_idx"]
    mn9_idx = int(d["mn9_idx"])

    total_spikes = binned.sum(axis=0)
    active = set(np.nonzero(total_spikes > 0)[0].tolist())
    seed = set(sugar_idx.tolist())
    keep = sorted(active | seed | {mn9_idx})
    keep_arr = np.array(keep)
    old_to_new = {old: new for new, old in enumerate(keep)}
    print(f"Visualizing {len(keep)} neurons (active + sugar seeds + MN9) "
          f"out of {len(root_id)} in the subcircuit")
    print(f"MN9 fired: {mn9_idx in active}, total MN9 spikes: {int(total_spikes[mn9_idx])}")

    # induced subgraph among kept neurons
    mask = np.isin(pre_idx, keep_arr) & np.isin(post_idx, keep_arr)
    epre = np.array([old_to_new[p] for p in pre_idx[mask].tolist()])
    epost = np.array([old_to_new[p] for p in post_idx[mask].tolist()])
    ew = signed_weight[mask]

    # BFS hop distance from sugar seeds, ignoring edge sign/direction of
    # weight but respecting pre->post direction (information flow)
    n = len(keep)
    fwd = [[] for _ in range(n)]
    for p, q in zip(epre.tolist(), epost.tolist()):
        fwd[p].append(q)

    seed_new = [old_to_new[s] for s in seed]
    dist = [-1] * n
    dq = deque()
    for s in seed_new:
        dist[s] = 0
        dq.append(s)
    while dq:
        u = dq.popleft()
        for v in fwd[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                dq.append(v)
    max_hop = max((d for d in dist if d >= 0), default=0)
    # neurons unreachable from seeds via forward edges (e.g. only reached
    # via an inhibitory feedback path) get placed one layer past max_hop
    dist = [d if d >= 0 else max_hop + 1 for d in dist]

    nodes = []
    for new_i, old_i in enumerate(keep):
        nodes.append({
            "id": new_i,
            "root_id": str(int(root_id[old_i])),
            "hop": int(dist[new_i]),
            "is_sugar": bool(old_i in seed),
            "is_mn9": bool(old_i == mn9_idx),
            "activity": [int(c) for c in binned[:, old_i].tolist()],
            "total_spikes": int(total_spikes[old_i]),
        })

    edges = []
    seen = set()
    for p, q, w in zip(epre.tolist(), epost.tolist(), ew.tolist()):
        key = (p, q)
        if key in seen:
            continue
        seen.add(key)
        edges.append({"source": p, "target": q, "sign": 1 if w > 0 else -1})

    out = {
        "bin_ms": bin_ms,
        "n_bins": int(binned.shape[0]),
        "max_hop": max_hop + 1,
        "nodes": nodes,
        "edges": edges,
    }
    with open("analysis/brain_activity.json", "w") as f:
        json.dump(out, f)
    print(f"Wrote analysis/brain_activity.json: {len(nodes)} nodes, {len(edges)} edges")


if __name__ == "__main__":
    main()
