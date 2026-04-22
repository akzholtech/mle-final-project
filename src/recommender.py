from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares


class ALSRecommender:
    def __init__(
        self,
        model: AlternatingLeastSquares,
        user_encoder: Any,
        item_encoder: Any,
        user_items: sp.csr_matrix | None,
        popular_items: pd.DataFrame,
    ) -> None:
        self.model = model
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder
        self.user_items = user_items
        self.popular_items = popular_items
        self.known_users = set(user_encoder.classes_)

    @classmethod
    def from_local_dir(cls, model_dir: str | Path) -> "ALSRecommender":
        model_dir = Path(model_dir)

        # model = AlternatingLeastSquares.load(str(model_dir / "artifacts" / "als_model.npz"))
        with open(model_dir / "artifacts" / "als_model.pkl", "rb") as f:
            model = pickle.load(f)

        with open(model_dir / "artifacts" / "user_encoder.pkl", "rb") as f:
            user_encoder = pickle.load(f)

        with open(model_dir / "artifacts" / "item_encoder.pkl", "rb") as f:
            item_encoder = pickle.load(f)

        popular_items = pd.read_parquet(model_dir / "artifacts"  / "popular_items.parquet")

        user_items_path = model_dir / "artifacts" / "user_items_matrix.npz"
        user_items = sp.load_npz(user_items_path) if user_items_path.exists() else None

        return cls(
            model=model,
            user_encoder=user_encoder,
            item_encoder=item_encoder,
            user_items=user_items,
            popular_items=popular_items,
        )

    def recommend(self, user_id: int | str, top_k: int = 10) -> dict:
        if user_id not in self.known_users:
            recs = self.popular_items["item_id"].head(top_k).tolist()
            return {
                "user_id": user_id,
                "recommendations": recs,
                "scores": [0.0] * len(recs),
                "source": "popular_fallback",
            }

        if self.user_items is None:
            raise ValueError("user_items matrix is required for ALS recommendations")

        user_idx = self.user_encoder.transform([user_id])[0]

        item_ids, scores = self.model.recommend(
            userid=user_idx,
            user_items=self.user_items[user_idx],
            N=top_k,
            filter_already_liked_items=True,
        )

        decoded_items = self.item_encoder.inverse_transform(item_ids)

        return {
            "user_id": user_id,
            "recommendations": decoded_items.tolist(),
            "scores": [float(x) for x in scores.tolist()],
            "source": "als",
        }