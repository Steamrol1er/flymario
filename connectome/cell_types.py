"""Named neuron groups within the full connectome.

The Shiu et al. connectivity tables give us wiring by FlyWire root ID/dense
index only — no cell-type names. To find "the Kenyon cells" or "the PAM
dopamine neurons" or "DNp09" inside those 138,639 indices, you need FlyWire's
separate cell-type annotation tables (`classification.csv` / `cell_types.csv`
from Codex, see `connectome/download.py --source flywire`).

This module is the single place that turns "give me the KC indices" into
concrete dense indices, and is deliberately kept separate from `loader.py`
so the rest of the codebase never hardcodes root IDs.

STATUS: annotation download/matching is not yet implemented — see
`docs/STATUS.md`. Until then, `resolve_groups` raises `NotImplementedError`
with instructions, and everything else in the project (simulator, dopamine
rule, actor-critic, Mario env) is developed/tested against
`connectome.loader.load_toy_connectome`, which has its own hand-built KC/MBON
groups and needs no annotation file.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Cell-type substrings (case-insensitive) used to select each functional
# group from FlyWire's `cell_type` / `hemibrain_type` annotation columns.
# These are the conventional names used in the FlyWire/hemibrain literature
# (Aso et al. 2014 for MB compartments; Namiki et al. 2018, Card lab papers
# for descending neurons) but should be spot-checked against whatever
# annotation version you actually download, since naming has changed across
# FlyWire releases.
GROUP_PATTERNS = {
    "kenyon_cell": ["KC"],
    "mbon": ["MBON"],
    "pam_dan": ["PAM"],          # reward-signaling dopamine neurons
    "ppl1_dan": ["PPL1"],        # punishment-signaling dopamine neurons
    "photoreceptor": ["R1-6", "R7", "R8"],
    "dn_forward": ["DNp09"],     # walk forward
    "dn_backward": ["MDN"],      # walk backward
    "dn_jump": ["DNp01"],        # giant fiber, escape jump
    "dn_turn": ["DNa01", "DNa02"],
}


def load_annotations(data_dir: Path | str = "data") -> pd.DataFrame:
    path = Path(data_dir) / "classification.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run:\n"
            "  python -m connectome.download --source flywire --dest data/\n"
            "and follow the printed instructions to fetch classification.csv "
            "from FlyWire Codex (requires a free account; can't be scripted)."
        )
    return pd.read_csv(path)


def resolve_groups(
    root_id_to_index: dict[int, int],
    data_dir: Path | str = "data",
) -> dict[str, list[int]]:
    """Return {group_name: [dense_index, ...]} for each entry in
    GROUP_PATTERNS, by substring-matching FlyWire cell-type annotations and
    mapping matched root IDs through `root_id_to_index` (from
    `connectome.loader.load_shiu_connectome(...).root_id_to_index`)."""
    ann = load_annotations(data_dir)
    type_col = next(
        (c for c in ["cell_type", "hemibrain_type", "cell_class"] if c in ann.columns),
        None,
    )
    if type_col is None:
        raise ValueError(f"Couldn't find a cell-type column in {list(ann.columns)}")

    groups: dict[str, list[int]] = {}
    for group, patterns in GROUP_PATTERNS.items():
        mask = ann[type_col].astype(str).str.contains("|".join(patterns), case=False, na=False)
        root_ids = ann.loc[mask, "root_id"].astype(int).tolist()
        groups[group] = [root_id_to_index[r] for r in root_ids if r in root_id_to_index]
    return groups
