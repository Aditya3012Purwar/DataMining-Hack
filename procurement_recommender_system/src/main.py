from __future__ import annotations

import json
from pathlib import Path

from aggregate import aggregate_transactions, load_transactions
from candidate_generation import generate_candidates
from ranking import score_candidates


BASE_DIR = Path(__file__).resolve().parent.parent

# Use real generated SKU aggregates if available, else fall back to sample fixtures.
_REAL_SKU_AGG = BASE_DIR / "data" / "generated" / "sku_aggregates_from_csv.json"
TRANSACTIONS_PATH = BASE_DIR / "data" / "examples" / "sample_transactions.json"
OUTPUT_PATH = BASE_DIR / "outputs" / "recommendations.json"


def save_recommendations(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def run_pipeline(customer_id: str = "C1") -> list[dict]:
    tx = load_transactions(TRANSACTIONS_PATH)
    sku_agg = aggregate_transactions(tx)
    candidates = generate_candidates(customer_id=customer_id, transactions=tx, sku_aggregates=sku_agg)
    ranked = score_candidates(candidates)

    out = []
    for _, row in ranked.iterrows():
        out.append(
            {
                "customer_id": row["customer_id"],
                "sku": row["candidate_sku"],
                "p_buy": round(float(row["p_buy"]), 4),
                "price_advantage": round(float(row["price_advantage"]), 4),
                "final_score": round(float(row["final_score"]), 4),
                "rank": int(row["rank"]),
            }
        )

    save_recommendations(out, OUTPUT_PATH)
    return out


if __name__ == "__main__":
    recs = run_pipeline("C1")
    print(f"Generated {len(recs)} recommendations at: {OUTPUT_PATH}")
