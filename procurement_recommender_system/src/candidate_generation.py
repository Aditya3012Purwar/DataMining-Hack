from __future__ import annotations

import pandas as pd


def _normalize(series: pd.Series) -> pd.Series:
    min_v = float(series.min())
    max_v = float(series.max())
    if max_v == min_v:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - min_v) / (max_v - min_v)


def generate_candidates(
    customer_id: str,
    transactions: pd.DataFrame,
    sku_aggregates: pd.DataFrame,
    max_candidates: int = 500,
) -> pd.DataFrame:
    user_hist = transactions[transactions["customer_id"] == customer_id].copy()
    if user_hist.empty:
        return pd.DataFrame()

    owned_skus = set(user_hist["sku"].unique())
    user_eclass = set(user_hist["eclass"].unique())
    user_manufacturers = set(user_hist["manufacturer"].unique())

    ref_price_by_eclass = user_hist.groupby("eclass", as_index=False).agg(
        reference_price=("price", "mean")
    )

    pool = sku_aggregates[sku_aggregates["eclass"].isin(user_eclass)].copy()
    pool = pool[~pool["sku"].isin(owned_skus)].copy()

    if pool.empty:
        return pd.DataFrame()

    pool = pool.merge(ref_price_by_eclass, on="eclass", how="left")
    pool["customer_id"] = customer_id
    pool["candidate_sku"] = pool["sku"]
    pool["candidate_price"] = pool["avg_price"]
    pool["global_popularity"] = _normalize(pool["purchase_count"])
    pool["manufacturer_familiarity"] = pool["manufacturer"].isin(user_manufacturers).astype(float)

    return pool[
        [
            "customer_id",
            "candidate_sku",
            "eclass",
            "manufacturer",
            "candidate_price",
            "reference_price",
            "global_popularity",
            "manufacturer_familiarity",
        ]
    ].head(max_candidates)
