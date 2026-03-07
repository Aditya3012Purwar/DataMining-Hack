from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_transactions(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


def aggregate_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        transactions.groupby(["sku", "eclass", "manufacturer"], as_index=False)
        .agg(
            avg_price=("price", "mean"),
            purchase_count=("sku", "count"),
            last_seen_date=("date", "max"),
        )
        .sort_values("purchase_count", ascending=False)
    )
    return grouped


def save_aggregates(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = df.to_dict(orient="records")
    with path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
