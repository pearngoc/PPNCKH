"""
Baseline GLANCE on the Adult dataset — for comparison with TabGLANCE.

Same black-box model (BBMLPCLF) and same affected instances as run_tabglance_adult.py,
but using a standard local CF generator (NearestNeighbors, RandomSampling, or Dice).

Usage:
    cd /Users/ngocle/Projects/Learning/GLANCE
    /opt/homebrew/Caskroom/miniconda/base/envs/tabcf/bin/python run_glance_baseline_adult.py --generator NearestNeighbors
    /opt/homebrew/Caskroom/miniconda/base/envs/tabcf/bin/python run_glance_baseline_adult.py --generator RandomSampling
    /opt/homebrew/Caskroom/miniconda/base/envs/tabcf/bin/python run_glance_baseline_adult.py --generator Dice
"""
import sys
import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# ── Paths ──────────────────────────────────────────────────────────────────
GLANCE_SRC = os.path.join(os.path.dirname(__file__), 'src')
TABCF_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'TABCF'))

sys.path.insert(0, GLANCE_SRC)
sys.path.insert(0, TABCF_ROOT)

os.chdir(TABCF_ROOT)  # TABCF data lives here

from glance.glance.glance import GLANCE
from tabcf.vae.model import BBMLPCLF
from utils_train import preprocess

DATANAME     = 'adult'
DATA_DIR     = f'data/{DATANAME}'
BB_MODEL_PATH = f'{DATA_DIR}/black_box_mlp_hidden_16.pkl'
DEVICE       = 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--generator', type=str, default='NearestNeighbors',
        choices=['Dice', 'RandomSampling', 'NearestNeighbors'],
        help='Local CF generator to use',
    )
    parser.add_argument('--n-affected',       type=int, default=100,
                        help='Number of negatively-predicted test instances to explain')
    parser.add_argument('--initial-clusters', type=int, default=20)
    parser.add_argument('--final-clusters',   type=int, default=5)
    parser.add_argument('--n-local-cfs',      type=int, default=3,
                        help='Local CFs per cluster centroid')
    args = parser.parse_args()

    # ── Load dataset info ───────────────────────────────────────────────────
    with open(f'{DATA_DIR}/info.json') as f:
        info = json.load(f)

    num_cols   = [info['column_names'][i] for i in info['num_col_idx']]
    cat_cols   = [info['column_names'][i] for i in info['cat_col_idx']]
    target_col = info['target_col']
    target_cls = info['target_class']

    train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
    test_df  = pd.read_csv(f'{DATA_DIR}/test.csv')

    for col in cat_cols + [target_col]:
        train_df[col] = train_df[col].str.strip()
        test_df[col]  = test_df[col].str.strip()

    train_df['target'] = (train_df[target_col] == target_cls).astype(int)
    test_df['target']  = (test_df[target_col]  == target_cls).astype(int)
    train_df = train_df.drop(columns=[target_col])
    test_df  = test_df.drop(columns=[target_col])

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # ── OHE transformer (same as TabGLANCE script) ──────────────────────────
    ohe_transformer = ColumnTransformer([
        ('num', MinMaxScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
    ])
    ohe_transformer.fit(train_df[num_cols + cat_cols])

    # ── Load black-box BBMLPCLF ─────────────────────────────────────────────
    print("Loading pretrained black-box model...")
    preprocess(DATA_DIR, task_type=info['task_type'], inverse=True, num_encoding='min_max_torch')

    input_shape = ohe_transformer.transform(train_df[num_cols + cat_cols][:1]).shape[1]
    black_box_clf = BBMLPCLF(input_shape)
    black_box_clf.load_state_dict(torch.load(BB_MODEL_PATH, map_location=DEVICE))
    black_box_clf.eval()

    # ── sklearn-compatible model wrapper ────────────────────────────────────
    class TorchModelWrapper:
        def predict(self, df: pd.DataFrame) -> np.ndarray:
            X = ohe_transformer.transform(df[num_cols + cat_cols])
            X_t = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                probs = black_box_clf(X_t).numpy()
            return (probs[:, 1] >= 0.5).astype(int)

        def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
            X = ohe_transformer.transform(df[num_cols + cat_cols])
            X_t = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                return black_box_clf(X_t).numpy()

        # DiCE needs a score() method for sklearn backend
        def score(self, df: pd.DataFrame, y) -> float:
            return float(np.mean(self.predict(df) == np.array(y)))

    model = TorchModelWrapper()

    # ── Select affected instances (same seed as TabGLANCE script) ───────────
    feature_cols  = num_cols + cat_cols
    test_features = test_df[feature_cols]
    test_preds    = model.predict(test_features)
    affected      = test_features[test_preds == 0].head(args.n_affected)
    print(f"Affected instances: {len(affected)}")

    # ── Build training data for GLANCE ──────────────────────────────────────
    train_for_glance = train_df[feature_cols + ['target']]

    # Immutable features excluded from interventions (same as TabGLANCE run)
    feat_to_vary = [c for c in feature_cols if c != 'Sex']

    # ── Fit and run GLANCE ───────────────────────────────────────────────────
    print(f"Running GLANCE with {args.generator} "
          f"(initial={args.initial_clusters}, final={args.final_clusters}, "
          f"local_cfs={args.n_local_cfs})...")

    glance = GLANCE(
        model=model,
        initial_clusters=args.initial_clusters,
        final_clusters=args.final_clusters,
        num_local_counterfactuals=args.n_local_cfs,
        verbose=True,
    )
    glance.fit(
        X=train_for_glance.drop(columns=['target']),
        y=train_for_glance['target'],
        train_dataset=train_for_glance,
        cf_generator=args.generator,
        numeric_features_names=num_cols,
        categorical_features_names=cat_cols,
        feat_to_vary=feat_to_vary,
    )

    eff, cost, clusters, clusters_res, chosen_actions, final_costs = glance.explain_group(affected)

    # ── Print results ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Generator:            {args.generator}")
    print(f"GLOBAL EFFECTIVENESS: {eff}/{len(affected)} = {eff/len(affected):.2%}")
    if eff > 0:
        print(f"MEAN RECOURSE COST:   {cost/eff:.4f}")
    print(f"{'='*60}\n")

    for idx, stats in clusters_res.items():
        print(f"--- Action {idx+1} (cluster size: {stats['size']}) ---")
        action = stats['action']
        for col in num_cols:
            if action[col] != 0:
                print(f"  {col}: {action[col]:+.2f}")
        for col in cat_cols:
            if action[col] != '-':
                print(f"  {col}: → {action[col]}")
        print(f"  Effectiveness: {stats['effectiveness']:.2%}  "
              f"Cost: {stats['cost']:.4f}  "
              f"Feature-Type Bias: {stats['feature_type_bias']:.3f}")
        print()


if __name__ == '__main__':
    main()
