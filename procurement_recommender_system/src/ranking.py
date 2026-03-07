from __future__ import annotations

import numpy as np
import pandas as pd


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def score_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame()

    scored = candidates.copy()
    ref_price = scored["reference_price"].replace(0, np.nan)

    scored["price_advantage"] = ((ref_price - scored["candidate_price"]) / ref_price).fillna(0.0)
    scored["price_advantage"] = scored["price_advantage"].clip(lower=0.0)

    # Simple baseline ranker proxy for p_buy.
    linear = (
        1.2 * scored["global_popularity"]
        + 1.0 * scored["manufacturer_familiarity"]
        + 1.5 * scored["price_advantage"]
        - 0.2
    )

    scored["p_buy"] = _sigmoid(linear.to_numpy())
    scored["final_score"] = scored["p_buy"] * (1.0 + scored["price_advantage"])

    scored = scored.sort_values("final_score", ascending=False).reset_index(drop=True)
    scored["rank"] = scored.index + 1

    return scored[["customer_id", "candidate_sku", "p_buy", "price_advantage", "final_score", "rank"]]
