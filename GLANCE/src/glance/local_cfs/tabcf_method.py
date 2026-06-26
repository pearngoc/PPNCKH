# GLANCE/src/glance/local_cfs/tabcf_method.py
import sys
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..base import LocalCounterfactualMethod

_TABCF_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'TABCF')
)
if _TABCF_ROOT not in sys.path:
    sys.path.insert(0, _TABCF_ROOT)


class TabCFMethod(LocalCounterfactualMethod):
    """GLANCE LocalCounterfactualMethod backed by TABCF's Transformer-VAE
    latent-space optimization.

    Call fit() with pretrained TABCF artefacts, then use explain_instances()
    as a drop-in replacement for DiceMethod / NearestNeighborMethod.
    """

    def __init__(self):
        super().__init__()
        self.decoder_bb = None

    def fit(
        self,
        pre_encoder,
        pre_decoder,
        black_box_clf,
        token_dim: int,
        num_inverse,
        cat_inverse,
        info: dict,
        ohe_transformer,
        input_encoded_continuous_feature_indexes: list,
        input_encoded_categorical_feature_indexes: list,
        num_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
        feat_to_vary: str = "all",
        device: str = "cpu",
        max_iter: int = 500,
        lr: float = 0.05,
        validity_weight: float = 1.0,
        input_proximity_weight: float = 0.5,
        latent_proximity_weight: float = 0.5,
    ) -> "TabCFMethod":
        from tabcf.sample import Decoder_Black_Box

        self.device = device
        self.num_cols = num_cols or []
        self.cat_cols = cat_cols or []
        self.feat_to_vary = feat_to_vary
        self.info = info
        self.num_inverse = num_inverse
        self.cat_inverse = cat_inverse
        self.ohe_transformer = ohe_transformer
        self.max_iter = max_iter
        self.lr = lr
        self.validity_weight = validity_weight
        self.input_proximity_weight = input_proximity_weight
        self.latent_proximity_weight = latent_proximity_weight

        # Build index mapping once for DataFrame reconstruction
        idx_name_mapping = info.get("idx_name_mapping", {})
        self.idx_name_mapping = {int(k): v for k, v in idx_name_mapping.items()}

        # Store token_dim so encode_instances can reshape
        self.token_dim = token_dim

        self.decoder_bb = Decoder_Black_Box(
            black_box_clf=black_box_clf,
            pre_decoder=pre_decoder,
            pre_encoder=pre_encoder,
            token_dim=token_dim,
            input_encoded_continuous_feature_indexes=input_encoded_continuous_feature_indexes,
            input_encoded_categorical_feature_indexes=input_encoded_categorical_feature_indexes,
        ).to(device)
        self.decoder_bb.eval()

        return self

    def explain_instances(
        self, instances: pd.DataFrame, num_counterfactuals: int
    ) -> pd.DataFrame:
        if self.decoder_bb is None:
            raise ValueError("Call fit() before explain_instances().")

        all_rows = []
        for _, row in instances.iterrows():
            row_df = pd.DataFrame([row])
            x_ohe = self.ohe_transformer.transform(row_df)  # [1, n_ohe_features]
            x_ohe_t = torch.tensor(x_ohe, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                z_orig = self.decoder_bb.encode_z(x_ohe_t)  # [latent_dim]

            for _ in range(num_counterfactuals):
                z_cf = z_orig.clone().detach().requires_grad_(True)
                optimizer = torch.optim.Adam([z_cf], lr=self.lr)

                for _ in range(self.max_iter):
                    optimizer.zero_grad()
                    pred = self.decoder_bb.forward(z_cf.unsqueeze(0))  # [1, 2]
                    validity_loss = -torch.log(pred[:, 1] + 1e-8).mean()
                    lat_prox = F.mse_loss(z_cf, z_orig.detach())
                    x_cf_ohe = self.decoder_bb.decode_z(z_cf.unsqueeze(0))
                    inp_prox = F.l1_loss(x_cf_ohe, x_ohe_t)
                    loss = (
                        self.validity_weight * validity_loss
                        + self.latent_proximity_weight * lat_prox
                        + self.input_proximity_weight * inp_prox
                    )
                    loss.backward()
                    optimizer.step()

                cf_row = self._decode_z_to_row(z_cf.detach())
                if cf_row is not None:
                    all_rows.append(cf_row)

        if not all_rows:
            return pd.DataFrame(columns=instances.columns)

        result = pd.DataFrame(all_rows).reset_index(drop=True)
        # Keep only columns that exist in instances
        shared = [c for c in instances.columns if c in result.columns]
        return result[shared]

    def encode_instances(self, instances: pd.DataFrame) -> np.ndarray:
        """Encode instances to TABCF latent space.
        Returns float32 array of shape [N, latent_dim].
        Used by latent-aware cluster merging in GLANCE.
        """
        if self.decoder_bb is None:
            raise ValueError("Call fit() before encode_instances().")

        vectors = []
        for _, row in instances.iterrows():
            row_df = pd.DataFrame([row])
            x_ohe = self.ohe_transformer.transform(row_df)
            x_ohe_t = torch.tensor(x_ohe, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                z = self.decoder_bb.encode_z(x_ohe_t)
            vectors.append(z.cpu().numpy())

        return np.stack(vectors, axis=0)

    def _decode_z_to_row(self, z: torch.Tensor) -> Optional[dict]:
        """Decode a latent vector z to a dict of original feature values."""
        from tabcf.latent_utils import split_num_cat_target, recover_data

        z_np = z.detach().cpu().numpy().reshape(1, -1)
        try:
            syn_num, syn_cat = split_num_cat_target(
                z_np, self.info, self.num_inverse, self.cat_inverse, self.device
            )
            cf_df = recover_data(syn_num, syn_cat, self.info)
            if self.idx_name_mapping:
                cf_df.rename(columns=self.idx_name_mapping, inplace=True)
            # Strip whitespace from string columns so comparisons against stripped train data work
            for col in cf_df.select_dtypes(include='object').columns:
                cf_df[col] = cf_df[col].str.strip()
            return cf_df.iloc[0].to_dict()
        except Exception:
            return None
