from itertools import combinations
from typing import List

import numpy as np
import pandas as pd


def feasibility_score(
    action: pd.Series,
    train_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    no_action: str = '-',
) -> float:
    """Mean feasibility across changed features.

    Numerical: 1 / (1 + |delta| / std).  Large delta relative to training
    std → score near 0 (infeasible).  Categorical changes always score 1.0.
    Returns 0.0 if no features changed.
    """
    scores = []
    for col in num_cols:
        delta = abs(action[col])
        if delta < 1e-6:
            continue
        std = train_df[col].std()
        scores.append(1.0 / (1.0 + delta / std) if std > 0 else 0.0)
    for col in cat_cols:
        if action[col] != no_action:
            scores.append(1.0)
    return float(np.mean(scores)) if scores else 0.0


def dominant_feature_concentration(
    action: pd.Series,
    train_df: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    no_action: str = '-',
) -> float:
    """Fraction of total normalised change attributable to the single largest feature.

    Numerical deltas are normalised by training range; categorical changes
    contribute 1.0 each.  Returns 0.0 if no features changed.
    Score near 1.0 → one feature dominates (e.g. capital-gain exploit).
    Score near 1/K → K features contribute equally.
    """
    contributions = []
    for col in num_cols:
        delta = abs(action[col])
        if delta < 1e-6:
            continue
        col_range = train_df[col].max() - train_df[col].min()
        contributions.append(delta / col_range if col_range > 0 else 0.0)
    for col in cat_cols:
        if action[col] != no_action:
            contributions.append(1.0)
    if not contributions:
        return 0.0
    total = sum(contributions)
    return float(max(contributions) / total) if total > 0 else 0.0


def action_diversity(
    actions: List[pd.Series],
    num_cols: List[str],
    cat_cols: List[str],
    no_action: str = '-',
) -> float:
    """1 − mean pairwise Jaccard similarity over changed-feature sets.

    0.0 → all actions change identical features.
    1.0 → all actions change completely disjoint features.
    """
    feature_sets = []
    for action in actions:
        changed = set()
        for col in num_cols:
            if abs(action[col]) > 1e-6:
                changed.add(col)
        for col in cat_cols:
            if action[col] != no_action:
                changed.add(col)
        feature_sets.append(changed)

    if len(feature_sets) < 2:
        return 0.0

    sims = []
    for i, j in combinations(range(len(feature_sets)), 2):
        a, b = feature_sets[i], feature_sets[j]
        union = len(a | b)
        sims.append(len(a & b) / union if union > 0 else 1.0)

    return float(1.0 - np.mean(sims))
