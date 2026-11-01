"""Fast regression checks: `python -m pytest tests -q`.

The slow full-brain sugar -> MN9 check stays a script (tests/check_sugar_feeding.py).
"""
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from tests import toy_conditioning

HAS_DATA = Path("data/Connectivity_783.parquet").exists()


def test_toy_conditioning_learns():
    assert toy_conditioning.run(verbose=False)


@pytest.mark.skipif(not HAS_DATA, reason="needs data/ from connectome.download")
def test_marksman_circuit_matches_published_size():
    from connectome.marksman_circuit import load_marksman_circuit, scrambled_copy

    c = load_marksman_circuit()
    w = c.connectome.signed_weight
    assert c.connectome.n_neurons == 3963
    assert len(w) == 354879
    assert ((w > 0).sum(), (w < 0).sum()) == (220442, 134437)
    assert (len(c.groups["LC10a_L"]), len(c.groups["LC10a_R"])) == (115, 119)

    # scramble keeps every neuron's out-degree, in-degree and the sign balance
    s = scrambled_copy(c, seed=0).connectome
    assert np.array_equal(np.sort(s.signed_weight), np.sort(w))
    assert Counter(s.pre_idx.tolist()) == Counter(c.connectome.pre_idx.tolist())
    assert Counter(s.post_idx.tolist()) == Counter(c.connectome.post_idx.tolist())
    assert not np.array_equal(s.post_idx, c.connectome.post_idx)
