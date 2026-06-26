"""
TabGLANCE on the Adult dataset — end-to-end runnable script.

Usage:
    cd /Users/ngocle/Projects/Learning/GLANCE
    /opt/homebrew/Caskroom/miniconda/base/envs/tabcf/bin/python run_tabglance_adult.py

What it does:
  1. Loads the pretrained TABCF VAE (encoder + decoder) and black-box MLP
     from TABCF/tabcf/vae/ckpt/adult/
  2. Fits a TabCFMethod wrapper around those models
  3. Runs GLANCE with TabCFMethod as the local CF generator
     (+ optional latent-aware merge with --latent-weight)
  4. Prints global actions, effectiveness, cost, and feature-type-bias
     for each cluster
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

os.chdir(TABCF_ROOT)  # TABCF uses relative paths like 'data/adult/...'

from glance.local_cfs.tabcf_method import TabCFMethod
from glance.glance.glance import GLANCE

from tabcf.vae.model import BBMLPCLF, Decoder_model, Encoder_model_Z
from tabcf.latent_utils import split_num_cat_target, recover_data
from utils_train import preprocess


# ── Config ─────────────────────────────────────────────────────────────────
DATANAME      = 'adult'
DATA_DIR      = f'data/{DATANAME}'
VAE_CKPT_DIR  = (
    f'tabcf/vae/ckpt/{DATANAME}/'
    'num_encoding[min_max_torch]_num_loss[L2]_cat_actv[gumbel_softmax, tau=1.0]_cont_actv[sigmoid]_reparam'
)
BB_MODEL_PATH = f'{DATA_DIR}/black_box_mlp_hidden_16.pkl'
TOKEN_DIM     = 4
DEVICE        = 'cpu'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initial-clusters', type=int, default=100)
    parser.add_argument('--final-clusters',   type=int, default=4)
    parser.add_argument('--n-local-cfs',      type=int, default=10,
                        help='Local CFs per cluster centroid')
    parser.add_argument('--latent-weight',    type=float, default=0.5,
                        help='Weight for latent-space distance in merge heuristic (0 = disabled)')
    parser.add_argument('--max-iter',         type=int, default=5000,
                        help='Gradient descent iterations per CF')
    args = parser.parse_args()

    # ── Load dataset info ───────────────────────────────────────────────────
    with open(f'{DATA_DIR}/info.json') as f:
        info = json.load(f)

    num_cols    = [info['column_names'][i] for i in info['num_col_idx']]
    cat_cols    = [info['column_names'][i] for i in info['cat_col_idx']]
    target_col  = info['target_col']
    target_cls  = info['target_class']   # e.g. '>50K'
    neg_cls     = info['negative_class'] # e.g. '<=50K'

    train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
    test_df  = pd.read_csv(f'{DATA_DIR}/test.csv')

    # Strip whitespace from string columns (Adult dataset has leading spaces)
    for col in cat_cols + [target_col]:
        train_df[col] = train_df[col].str.strip()
        test_df[col]  = test_df[col].str.strip()

    # Map target to 0/1
    train_df['target'] = (train_df[target_col] == target_cls).astype(int)
    test_df['target']  = (test_df[target_col]  == target_cls).astype(int)
    train_df = train_df.drop(columns=[target_col])
    test_df  = test_df.drop(columns=[target_col])

    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # ── OHE transformer for TabCFMethod ─────────────────────────────────────
    ohe_transformer = ColumnTransformer([
        ('num', MinMaxScaler(), num_cols),
        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), cat_cols),
    ])
    ohe_transformer.fit(train_df[num_cols + cat_cols])
    X_train_ohe = ohe_transformer.transform(train_df[num_cols + cat_cols])

    # Figure out OHE index groups
    n_num = len(num_cols)
    cont_feature_indexes = list(range(n_num))

    cat_feature_indexes = []
    offset = n_num
    ohe_enc = ohe_transformer.named_transformers_['cat']
    for cats in ohe_enc.categories_:
        n_cats = len(cats)
        cat_feature_indexes.append(list(range(offset, offset + n_cats)))
        offset += n_cats

    # ── Load pretrained TABCF models ─────────────────────────────────────────
    print("Loading pretrained TABCF models...")

    _, _, categories, d_numerical, num_inverse, cat_inverse, _ = preprocess(
        DATA_DIR, task_type=info['task_type'], inverse=True, num_encoding='min_max_torch'
    )

    pre_decoder = Decoder_model(
        2, d_numerical, categories, TOKEN_DIM,
        n_head=1, factor=32, gumbel_softmax=True, sigmoid=True, tau=1.0
    )
    pre_decoder.load_state_dict(torch.load(f'{VAE_CKPT_DIR}/decoder.pt', map_location=DEVICE))
    pre_decoder.eval()

    pre_encoder = Encoder_model_Z(2, d_numerical, categories, TOKEN_DIM, n_head=1, factor=32)
    pre_encoder.load_state_dict(torch.load(f'{VAE_CKPT_DIR}/encoder.pt', map_location=DEVICE))
    pre_encoder.eval()

    input_shape = X_train_ohe.shape[1]
    black_box_clf = BBMLPCLF(input_shape)  # default hidden_size=1000
    black_box_clf.load_state_dict(torch.load(BB_MODEL_PATH, map_location=DEVICE))
    black_box_clf.eval()

    # Store models in info for split_num_cat_target
    info['pre_decoder'] = pre_decoder
    info['token_dim']   = TOKEN_DIM

    idx_name_mapping = {int(k): v for k, v in info['idx_name_mapping'].items()}

    # ── Fit TabCFMethod ──────────────────────────────────────────────────────
    print("Fitting TabCFMethod...")
    tabcf = TabCFMethod()
    tabcf.fit(
        pre_encoder=pre_encoder,
        pre_decoder=pre_decoder,
        black_box_clf=black_box_clf,
        token_dim=TOKEN_DIM,
        num_inverse=num_inverse,
        cat_inverse=cat_inverse,
        info=info,
        ohe_transformer=ohe_transformer,
        input_encoded_continuous_feature_indexes=cont_feature_indexes,
        input_encoded_categorical_feature_indexes=cat_feature_indexes,
        num_cols=num_cols,
        cat_cols=cat_cols,
        device=DEVICE,
        max_iter=args.max_iter,
        lr=0.05,
        validity_weight=1.0,
        input_proximity_weight=1.0,
        latent_proximity_weight=1.0,
    )

    # ── GLANCE model wrapper ─────────────────────────────────────────────────
    class TorchModelWrapper:
        """Wraps BBMLPCLF to give sklearn-style predict(DataFrame)."""
        def predict(self, df: pd.DataFrame) -> np.ndarray:
            X = ohe_transformer.transform(df[num_cols + cat_cols])
            X_t = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                probs = black_box_clf(X_t).numpy()
            return (probs[:, 1] >= 0.5).astype(int)

    model = TorchModelWrapper()

    # ── Select affected instances (predicted negative) ───────────────────────
    feature_cols = num_cols + cat_cols
    test_features = test_df[feature_cols]
    test_preds = model.predict(test_features)
    affected = test_features[test_preds == 0]
    print(f"Affected instances: {len(affected)}")

    # ── Fit and run GLANCE ───────────────────────────────────────────────────
    print(f"Running GLANCE (initial={args.initial_clusters}, final={args.final_clusters}, "
          f"local_cfs={args.n_local_cfs}, latent_weight={args.latent_weight})...")

    train_for_glance = train_df[feature_cols + ['target']]

    glance = GLANCE(
        model=model,
        initial_clusters=args.initial_clusters,
        final_clusters=args.final_clusters,
        num_local_counterfactuals=args.n_local_cfs,
        latent_heuristic_weight=args.latent_weight,
        verbose=True,
    )
    glance.fit(
        X=train_for_glance.drop(columns=['target']),
        y=train_for_glance['target'],
        train_dataset=train_for_glance,
        cf_generator=tabcf,
        latent_encoder=tabcf.encode_instances if args.latent_weight > 0 else None,
        numeric_features_names=num_cols,
        categorical_features_names=cat_cols,
    )

    eff, cost, clusters, clusters_res, chosen_actions, final_costs = glance.explain_group(affected)

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
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
