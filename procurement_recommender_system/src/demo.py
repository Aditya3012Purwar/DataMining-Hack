"""
Procurement Recommender — Demo Script
======================================
Step 1 (one-time preprocessing, run once):
    python src/csv_to_json_artifacts.py --mode fast --max-skus 10000

Step 2 (demo, instant):
    python src/demo.py --customer 41303727
    python src/demo.py --customer 41303727 --top 10
    python src/demo.py --customer 41303727 --eclass 27141104
    python src/demo.py --list-customers

What it shows:
    Given a buyer's real purchase history, find the top N
    cheaper-but-equivalent items they are most likely to buy next,
    ranked by:  final_score = P(buy) × (1 + price_advantage)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from textwrap import shorten
from typing import Any

import numpy as np
import pandas as pd

from cold_start import (
    build_warm_profiles,
    find_closest_warm_customer,
    load_cold_customer_features,
    load_nace_lookup,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_CONFIG_PATH = BASE_DIR / "config" / "source_data_paths.json"
GEN_DIR = BASE_DIR / "data" / "generated"
OUT_DIR = BASE_DIR / "outputs"

SKU_AGG_PATH = GEN_DIR / "sku_aggregates_from_csv.json"
SKU_FEAT_PATH = GEN_DIR / "sku_feature_profiles_sample_from_csv.json"
FEAT_SUMMARY_PATH = GEN_DIR / "feature_types_summary_from_csv.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _check_artifacts() -> None:
    missing = [str(p) for p in [SKU_AGG_PATH] if not p.exists()]
    if missing:
        raise SystemExit(
            "\n[ERROR] Pre-generated JSON artifacts not found.\n"
            "Run this first:\n\n"
            "    python src/csv_to_json_artifacts.py --mode fast --max-skus 10000\n"
        )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _plis_path() -> Path:
    cfg = _load_json(SOURCE_CONFIG_PATH)
    return Path(cfg["plis_training_csv"])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ---------------------------------------------------------------------------
# Load customer history from real CSV (fast: reads only matching rows via chunks)
# ---------------------------------------------------------------------------
def load_customer_history(
    customer_id: int,
    plis_path: Path,
    chunk_progress_cb=None,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    chunk_num = 0
    for chunk in pd.read_csv(
        plis_path,
        sep="\t",
        usecols=["legal_entity_id", "sku", "eclass", "manufacturer", "vk_per_item", "quantityvalue", "orderdate"],
        dtype={"sku": "string", "eclass": "string", "manufacturer": "string", "vk_per_item": "float64", "quantityvalue": "float64"},
        parse_dates=["orderdate"],
        chunksize=300_000,
        low_memory=False,
    ):
        chunk_num += 1
        sub = chunk[chunk["legal_entity_id"] == customer_id]
        if not sub.empty:
            parts.append(sub)
        if chunk_progress_cb and chunk_num % 5 == 0:
            rows_scanned = chunk_num * 300_000
            match_count = sum(len(p) for p in parts)
            chunk_progress_cb(
                f"Scanned {rows_scanned:,} rows… {match_count:,} matching transactions found so far"
            )

    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Core pipeline — returns structured data (used by CLI and web GUI)
# ---------------------------------------------------------------------------
def run_demo_data(
    customer_id: int,
    top_n: int = 20,
    filter_eclass: str | None = None,
    min_price: float = 0.50,
    progress_cb=None,
) -> dict:
    """Run the recommendation pipeline and return a structured result dict."""

    def _progress(step: int, message: str) -> None:
        if progress_cb:
            progress_cb({"step": step, "message": message})

    _check_artifacts()

    # 1. Load catalogue
    _progress(1, "Loading pre-generated SKU catalogue…")
    sku_agg: list[dict] = _load_json(SKU_AGG_PATH)
    sku_df = pd.DataFrame(sku_agg)
    sku_df["sku"] = sku_df["sku"].astype("string")
    sku_df["eclass"] = sku_df["eclass"].astype("string")
    sku_df["manufacturer"] = sku_df["manufacturer"].astype("string")
    catalogue_size = len(sku_df)
    _progress(1, f"Catalogue loaded: {catalogue_size:,} unique SKUs")

    feat_profiles: dict[str, dict] = {}
    if SKU_FEAT_PATH.exists():
        feat_profiles = _load_json(SKU_FEAT_PATH)

    # 2. Load customer history
    _progress(2, f"Fetching purchase history for customer {customer_id}…")
    plis_path = _plis_path()
    hist = load_customer_history(
        customer_id,
        plis_path,
        chunk_progress_cb=lambda msg: _progress(2, msg),
    )

    # --- Cold-start detection & proxy resolution ---
    cold_start_info = None
    proxy_customer_id = None
    if hist.empty:
        _progress(2, f"No history for {customer_id} \u2014 activating cold-start matching\u2026")
        try:
            cold_features = load_cold_customer_features(customer_id)
        except ValueError:
            cold_features = {"customer_id": customer_id, "nace_code": None,
                            "secondary_nace_code": None,
                            "estimated_number_employees": None, "task": "unknown"}

        _progress(2, "Building warm-customer profiles for similarity matching\u2026")
        warm_df = build_warm_profiles(
            progress_cb=lambda msg: _progress(2, msg),
        )
        matches = find_closest_warm_customer(
            cold_features.get("nace_code"),
            cold_features.get("secondary_nace_code"),
            cold_features.get("estimated_number_employees"),
            warm_df,
            top_k=5,
        )
        if not matches or matches[0]["similarity"] <= 0:
            raise ValueError(
                f"Customer {customer_id} has no purchase history and no similar "
                "warm customer could be found. Cannot generate recommendations."
            )

        proxy_customer_id = matches[0]["customer_id"]
        nace_desc = load_nace_lookup()
        cold_nace = cold_features.get("nace_code")
        proxy_nace = matches[0].get("nace_code")
        cold_start_info = {
            "is_cold_start": True,
            "original_customer_id": customer_id,
            "proxy_customer_id": proxy_customer_id,
            "similarity_score": matches[0]["similarity"],
            "cold_customer_features": cold_features,
            "cold_nace_description": nace_desc.get(cold_nace, "") if cold_nace else "",
            "proxy_nace_description": nace_desc.get(proxy_nace, "") if proxy_nace else "",
            "top_matches": matches,
        }
        _progress(2, f"Cold-start: using proxy customer {proxy_customer_id} "
                     f"(similarity {matches[0]['similarity']:.0%})")

        # Reload history with the proxy customer
        hist = load_customer_history(
            proxy_customer_id,
            plis_path,
            chunk_progress_cb=lambda msg: _progress(2, msg),
        )
        if hist.empty:
            raise ValueError(
                f"Proxy customer {proxy_customer_id} also has no history. "
                "Cannot generate recommendations."
            )

    effective_customer_id = proxy_customer_id or customer_id

    eclass_breakdown = (
        hist.groupby("eclass")["sku"].count().nlargest(10)
        .reset_index().rename(columns={"sku": "count"})
    )
    _progress(2, f"History: {len(hist):,} rows | {hist['sku'].nunique():,} SKUs | {hist['eclass'].nunique()} E-classes"
               + (f" (via proxy {proxy_customer_id})" if proxy_customer_id else ""))

    ref_price = (
        hist.groupby("eclass", as_index=False)["vk_per_item"]
        .mean()
        .rename(columns={"vk_per_item": "reference_price"})
    )
    owned_skus = set(hist["sku"].dropna().unique())
    user_eclass = set(hist["eclass"].dropna().unique())
    user_manufacturers = set(hist["manufacturer"].dropna().unique())

    if filter_eclass:
        user_eclass = user_eclass & {filter_eclass}
        if not user_eclass:
            raise ValueError(f"Customer {customer_id} has no history for e-class {filter_eclass}.")

    # 3. Candidate generation
    _progress(3, f"Generating candidates from {len(user_eclass)} E-class(es)…")
    pool = sku_df[sku_df["eclass"].isin(user_eclass)].copy()
    pool = pool[~pool["sku"].isin(owned_skus)].copy()
    pool = pool.merge(ref_price, on="eclass", how="left")
    pool["manufacturer_familiarity"] = pool["manufacturer"].isin(user_manufacturers).astype(float)
    max_pop = float(pool["purchase_count"].max()) if not pool.empty else 1.0
    pool["global_popularity"] = pool["purchase_count"] / max(max_pop, 1)
    pool = pool[pool["avg_price"] >= min_price].copy()
    candidate_count = len(pool)
    _progress(3, f"{candidate_count:,} candidates after filtering")

    # 4. Scoring — no artificial cap on price_advantage so real savings are shown
    _progress(4, "Scoring candidates…")
    ref = pool["reference_price"].clip(lower=0.01)
    pool["price_advantage"] = ((ref - pool["avg_price"]) / ref).clip(lower=0.0)

    linear = (
        1.2 * pool["global_popularity"]
        + 0.8 * pool["manufacturer_familiarity"]
        + 1.5 * pool["price_advantage"]
        - 0.2
    )
    pool["p_buy"] = _sigmoid(linear.to_numpy())
    pool["final_score"] = pool["p_buy"] * (1.0 + pool["price_advantage"])
    pool = pool.sort_values("final_score", ascending=False)
    _progress(4, f"Scored {candidate_count:,} candidates")

    # 5. Enrich with features
    _progress(5, "Enriching top candidates with product features…")
    top = pool.head(top_n).reset_index(drop=True)
    top["rank"] = top.index + 1
    top["features_summary"] = top["sku"].apply(lambda s: _summarise_features(s, feat_profiles))

    out_records = _build_output_records(top)
    total_saving = float((top["reference_price"] - top["avg_price"]).clip(lower=0).sum())
    _progress(5, f"Done — {len(out_records)} recommendations generated")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"demo_{customer_id}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_records, f, indent=2)

    return {
        "query": {
            "customer_id": customer_id,
            "top_n": top_n,
            "filter_eclass": filter_eclass,
            "min_price": min_price,
        },
        "cold_start_info": cold_start_info,
        "catalogue_size": int(catalogue_size),
        "history": {
            "transaction_count": int(len(hist)),
            "unique_skus": int(hist["sku"].nunique()),
            "unique_eclasses": int(hist["eclass"].nunique()),
            "avg_price_paid_eur": round(float(hist["vk_per_item"].dropna().mean()), 2),
            "top_eclasses": [
                {"eclass": str(r["eclass"]), "count": int(r["count"])}
                for r in eclass_breakdown.to_dict(orient="records")
            ],
        },
        "candidate_count": int(candidate_count),
        "recommendations": out_records,
        "total_estimated_saving_eur": round(total_saving, 2),
    }


def run_demo(
    customer_id: int,
    top_n: int = 20,
    filter_eclass: str | None = None,
    min_price: float = 0.50,
) -> list[dict]:
    """CLI wrapper: runs the pipeline, prints a formatted table, returns records."""

    def _on_progress(event: dict) -> None:
        print(f"  [{event['step']}/5] {event['message']}")

    print(f"\n{'='*64}")
    print(f"  Procurement Recommender — Customer {customer_id}")
    print(f"{'='*64}\n")

    try:
        result = run_demo_data(customer_id, top_n, filter_eclass, min_price, progress_cb=_on_progress)
    except ValueError as exc:
        raise SystemExit(f"\n[ERROR] {exc}")

    top_df = pd.DataFrame(result["recommendations"]).rename(columns={
        "candidate_price_eur": "avg_price",
        "reference_price_eur": "reference_price",
    })
    top_df["price_advantage"] = top_df["price_advantage_pct"] / 100.0
    _print_recommendations(top_df, customer_id)
    print(f"\n  Saved to: {OUT_DIR / f'demo_{customer_id}.json'}\n")
    return result["recommendations"]


def _summarise_features(sku: str, profiles: dict) -> str:
    if sku not in profiles:
        return "—"
    kv = profiles[sku]
    parts = []
    # Prefer known semantic groups
    priority = ["color", "Farbe", "size", "groesse", "material", "dimension"]
    shown: set[str] = set()
    for p_key in priority:
        for k, vals in kv.items():
            if p_key.lower() in k.lower() and k not in shown:
                parts.append(f"{_short_key(k)}: {', '.join(vals[:2])}")
                shown.add(k)
            if len(parts) >= 3:
                break
        if len(parts) >= 3:
            break
    # Fill remaining slots from whatever keys are left
    if len(parts) < 3:
        for k, vals in kv.items():
            if k not in shown:
                parts.append(f"{_short_key(k)}: {', '.join(vals[:2])}")
                shown.add(k)
            if len(parts) >= 3:
                break
    return " | ".join(parts) if parts else "—"


def _short_key(key: str) -> str:
    # Strip the context suffix (e.g. "_main_nitrilhandschuh") and decode HTML codes
    base = key.split("-_")[0].replace("_", " ").strip()
    return shorten(base, width=20, placeholder="…")


def _print_recommendations(df: pd.DataFrame, customer_id: int) -> None:
    col_widths = {"rank": 4, "sku": 22, "eclass": 12, "manufacturer": 18,
                  "avg_price": 9, "ref_price": 9, "saved%": 7, "score": 7, "features": 50}
    header = (
        f"{'#':>4}  {'SKU':<22}  {'E-Class':<12}  {'Manufacturer':<18}  "
        f"{'Cand. €':>9}  {'Ref. €':>9}  {'Saved%':>7}  {'Score':>7}  Features"
    )
    sep = "-" * len(header)

    print(f"\n{'='*64}")
    print(f"  Top {len(df)} Recommendations for Customer {customer_id}")
    print(f"{'='*64}")
    print(header)
    print(sep)

    for _, row in df.iterrows():
        saved_pct = f"{row['price_advantage'] * 100:.1f}%"
        ref = row.get("reference_price", float("nan"))
        ref_str = f"{ref:.2f}" if not pd.isna(ref) else "  —"
        print(
            f"{int(row['rank']):>4}  "
            f"{str(row['sku']):<22}  "
            f"{str(row['eclass']):<12}  "
            f"{shorten(str(row['manufacturer']), 18, placeholder='…'):<18}  "
            f"{row['avg_price']:>9.2f}  "
            f"{ref_str:>9}  "
            f"{saved_pct:>7}  "
            f"{row['final_score']:>7.4f}  "
            f"{row['features_summary']}"
        )

    print(sep)
    total_savings = (df["reference_price"] - df["avg_price"]).clip(lower=0).sum()
    print(f"\n  Estimated total saving across top {len(df)} items: €{total_savings:.2f}")


def _build_output_records(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, row in df.iterrows():
        out.append({
            "rank": int(row["rank"]),
            "customer_id": int(row["legal_entity_id"]) if "legal_entity_id" in df.columns else None,
            "sku": str(row["sku"]),
            "eclass": str(row["eclass"]),
            "manufacturer": str(row["manufacturer"]),
            "candidate_price_eur": round(float(row["avg_price"]), 4),
            "reference_price_eur": round(float(row["reference_price"]), 4) if not pd.isna(row["reference_price"]) else None,
            "price_advantage_pct": round(float(row["price_advantage"]) * 100, 2),
            "p_buy": round(float(row["p_buy"]), 4),
            "final_score": round(float(row["final_score"]), 4),
            "features_summary": row["features_summary"],
        })
    return out


# ---------------------------------------------------------------------------
# List customers helper
# ---------------------------------------------------------------------------
def list_top_customers_data(n: int = 50) -> list[dict]:
    """Return top customers by transaction count as a list of dicts (no printing)."""
    plis_path = _plis_path()
    counts: dict[int, int] = {}
    for chunk in pd.read_csv(
        plis_path,
        sep="\t",
        usecols=["legal_entity_id"],
        dtype={"legal_entity_id": "Int64"},
        chunksize=500_000,
        low_memory=False,
    ):
        vc = chunk["legal_entity_id"].value_counts()
        for cid, cnt in vc.items():
            counts[int(cid)] = counts.get(int(cid), 0) + int(cnt)
    return [
        {"customer_id": cid, "transaction_count": cnt}
        for cid, cnt in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
    ]


def list_top_customers(n: int = 20) -> None:
    for row in list_top_customers_data(n):
        if row == list_top_customers_data(n)[0]:
            print(f"\n{'Customer ID':>14}   {'Transactions':>14}")
            print("-" * 32)
        print(f"{row['customer_id']:>14}   {row['transaction_count']:>14,}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Procurement Recommender Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/demo.py --list-customers
  python src/demo.py --customer 41303727
  python src/demo.py --customer 41303727 --top 15
  python src/demo.py --customer 41303727 --eclass 27141104
        """,
    )
    parser.add_argument("--customer", type=int, help="Customer (legal_entity_id) to recommend for")
    parser.add_argument("--top", type=int, default=20, help="Number of recommendations to show (default: 20)")
    parser.add_argument("--eclass", type=str, default=None, help="Restrict to a specific E-class code")
    parser.add_argument("--min-price", type=float, default=0.50, help="Minimum candidate item price in EUR (default: 0.50)")

    parser.add_argument("--list-customers", action="store_true", help="Print the top 20 customers by transaction volume and exit")

    args = parser.parse_args()

    if args.list_customers:
        list_top_customers()
    elif args.customer:
        run_demo(customer_id=args.customer, top_n=args.top, filter_eclass=args.eclass, min_price=args.min_price)
    else:
        parser.print_help()
