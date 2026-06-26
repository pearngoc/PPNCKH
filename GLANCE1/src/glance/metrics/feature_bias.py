# GLANCE/src/glance/metrics/feature_bias.py
from typing import List
import pandas as pd


def compute_feature_type_bias(
    action: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    no_action_token: str = "-",
) -> float:
    """Measure how skewed an action is toward one feature type.

    Returns a value in [0, 1]:
      0 = equally balanced between numerical and categorical changes
      1 = all changes are on one feature type only

    When no features change at all, returns 0.0 (no bias to report).
    """
    if not num_cols and not cat_cols:
        return 0.0

    n_num = len(num_cols)
    n_cat = len(cat_cols)

    num_changed = sum(1 for c in num_cols if action[c] != 0 and action[c] != no_action_token)
    cat_changed = sum(1 for c in cat_cols if action[c] != no_action_token)

    total_changed = num_changed + cat_changed
    if total_changed == 0:
        return 0.0

    num_ratio = num_changed / n_num if n_num > 0 else 0.0
    cat_ratio = cat_changed / n_cat if n_cat > 0 else 0.0

    return abs(num_ratio - cat_ratio)


def compute_actions_feature_type_bias(
    actions: List[pd.Series],
    num_cols: List[str],
    cat_cols: List[str],
    no_action_token: str = "-",
) -> List[float]:
    """Compute feature-type bias for each action in the list."""
    return [
        compute_feature_type_bias(a, num_cols, cat_cols, no_action_token)
        for a in actions
    ]
