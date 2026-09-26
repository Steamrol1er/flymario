"""Extract the total (network-wide) spike count per time bin from
analysis/sugar_activity.npz, for driving the pulse in viz/brain3d.html.

This is a much smaller/simpler summary than build_viz_data.py's per-neuron
export: just one number per 10ms bin, summed across all neurons in the
subcircuit (equivalently, across the 396 that actually fired).

Run: python -m analysis.build_activity_timeline
Writes: analysis/activity_timeline.json
"""
from __future__ import annotations

import json

import numpy as np


def main():
    d = np.load("analysis/sugar_activity.npz")
    timeline = {
        "bin_ms": float(d["bin_ms"]),
        "total_spikes_per_bin": d["binned_spikes"].sum(axis=1).tolist(),
    }
    with open("analysis/activity_timeline.json", "w") as f:
        json.dump(timeline, f)
    print("Wrote analysis/activity_timeline.json:", timeline["total_spikes_per_bin"])


if __name__ == "__main__":
    main()
