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
SETID_FEAT_PATH = GEN_DIR / "setid_feature_profiles_from_csv.json"
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


def _build_customer_feature_profile(
    owned_skus: set[str],
    owned_set_ids: set[str],
    feat_profiles: dict[str, dict],
    setid_profiles: dict[str, dict],
) -> dict[str, set[str]]:
    """Build an aggregated feature profile from the customer's purchased products.

    Merges features from all owned SKUs/set_ids into a single profile:
      { key: {val1, val2, ...}, ... }
    This represents 'what features this customer buys'.
    """
    profile: dict[str, set[str]] = {}
    # From SKU-level profiles
    for sku in owned_skus:
        if sku in feat_profiles:
            for k, vals in feat_profiles[sku].items():
                profile.setdefault(k, set()).update(vals if isinstance(vals, (list, set)) else [vals])
    # From set_id-level profiles (richer, consolidated across suppliers)
    for sid in owned_set_ids:
        if sid in setid_profiles:
            for k, vals in setid_profiles[sid].items():
                profile.setdefault(k, set()).update(vals if isinstance(vals, (list, set)) else [vals])
    return profile


def _feature_similarity(
    customer_profile: dict[str, set[str]],
    candidate_profile: dict,
) -> float:
    """Compute feature similarity between customer profile and a candidate product.

    Uses a weighted Jaccard-like approach:
    - For each feature key present in BOTH profiles, compute value overlap
    - Shared keys with matching values score highest
    - Having more documented features is a secondary signal

    Returns a score in [0, 1].
    """
    if not customer_profile or not candidate_profile:
        return 0.0

    cand_keys = set(candidate_profile.keys())
    cust_keys = set(customer_profile.keys())

    shared_keys = cand_keys & cust_keys
    if not shared_keys:
        # No overlapping feature keys — use key coverage as weak signal
        all_keys = cand_keys | cust_keys
        return 0.1 * len(cand_keys) / max(len(all_keys), 1)

    value_scores = []
    for key in shared_keys:
        cust_vals = customer_profile[key]
        cand_vals = candidate_profile[key]
        if isinstance(cand_vals, list):
            cand_vals = set(cand_vals)
        if isinstance(cust_vals, list):
            cust_vals = set(cust_vals)
        # Jaccard on values for this key
        intersection = len(cust_vals & cand_vals)
        union = len(cust_vals | cand_vals)
        if union > 0:
            value_scores.append(intersection / union)

    # Combine: average value overlap across shared keys, weighted by key coverage
    avg_value_overlap = sum(value_scores) / len(value_scores) if value_scores else 0.0
    key_coverage = len(shared_keys) / max(len(cust_keys), 1)
    return 0.7 * avg_value_overlap + 0.3 * key_coverage

def _build_feature_clusters(
    records: list[dict],
    feat_profiles: dict[str, dict],
    setid_profiles: dict[str, dict],
) -> tuple[list[dict], list[dict]]:
    """Group recommendation records into feature-based clusters.

    Each cluster represents a 'type of product' defined by its shared feature
    keys and dominant values — e.g. "color: blue, material: steel, size: L".
    This is the 'recommend a set of features' concept from the supervisor.

    Returns (updated_records_with_cluster_id, cluster_summaries).
    """
    if not records:
        return records, []

    # Compute a feature fingerprint for each record (key-sorted set of key=val pairs)
    fingerprints: list[frozenset] = []
    for rec in records:
        fd = rec.get("features_detail", {})
        pairs: set[str] = set()
        for k, vals in fd.items():
            short_k = k.split("-_")[0].replace("_", " ").strip().lower()
            if isinstance(vals, list):
                for v in vals[:3]:
                    pairs.add(f"{short_k}={v}")
            else:
                pairs.add(f"{short_k}={vals}")
        fingerprints.append(frozenset(pairs))

    # Simple greedy clustering: two records are in the same cluster if they share
    # at least 40% of their feature pairs (Jaccard >= 0.4) AND same eclass.
    cluster_ids: list[int] = [-1] * len(records)
    cluster_counter = 0
    for i in range(len(records)):
        if cluster_ids[i] >= 0:
            continue
        cluster_ids[i] = cluster_counter
        for j in range(i + 1, len(records)):
            if cluster_ids[j] >= 0:
                continue
            if records[i]["eclass"] != records[j]["eclass"]:
                continue
            fp_i, fp_j = fingerprints[i], fingerprints[j]
            if not fp_i and not fp_j:
                # Both have no features but same eclass
                cluster_ids[j] = cluster_counter
                continue
            union = len(fp_i | fp_j)
            if union == 0:
                continue
            jaccard = len(fp_i & fp_j) / union
            if jaccard >= 0.4:
                cluster_ids[j] = cluster_counter
        cluster_counter += 1

    # Build cluster summaries
    from collections import Counter as _Counter
    cluster_map: dict[int, list[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        cluster_map.setdefault(cid, []).append(idx)

    cluster_summaries: list[dict] = []
    for cid in sorted(cluster_map.keys()):
        members = cluster_map[cid]
        eclasses = set()
        all_feature_pairs: _Counter = _Counter()
        total_saving = 0.0
        for idx in members:
            rec = records[idx]
            eclasses.add(rec["eclass"])
            fd = rec.get("features_detail", {})
            for k, vals in fd.items():
                short_k = k.split("-_")[0].replace("_", " ").strip()
                if isinstance(vals, list):
                    for v in vals[:3]:
                        all_feature_pairs[f"{short_k}: {v}"] += 1
                else:
                    all_feature_pairs[f"{short_k}: {vals}"] += 1
            ref = rec.get("reference_price_eur") or 0
            cand = rec.get("candidate_price_eur") or 0
            total_saving += max(0, ref - cand)

        # Shared features = those that appear in majority of cluster members
        threshold = max(1, len(members) // 2)
        shared_features = [feat for feat, cnt in all_feature_pairs.most_common(20) if cnt >= threshold]
        distinguishing_features = [feat for feat, cnt in all_feature_pairs.most_common(10) if cnt >= threshold]

        cluster_summaries.append({
            "cluster_id": cid,
            "size": len(members),
            "eclass": sorted(eclasses)[0] if eclasses else "",
            "shared_features": shared_features[:10],
            "representative_sku": records[members[0]]["sku"],
            "representative_set_id": records[members[0]].get("set_id"),
            "avg_score": round(sum(records[idx]["final_score"] for idx in members) / len(members), 4),
            "total_saving_eur": round(total_saving, 2),
            "member_ranks": [records[idx]["rank"] for idx in members],
        })

    # Attach cluster_id to each record
    updated = []
    for idx, rec in enumerate(records):
        rec_copy = dict(rec)
        rec_copy["feature_cluster_id"] = cluster_ids[idx]
        # Find this cluster's shared features
        for cs in cluster_summaries:
            if cs["cluster_id"] == cluster_ids[idx]:
                rec_copy["feature_cluster_label"] = ", ".join(cs["shared_features"][:5]) if cs["shared_features"] else rec["eclass"]
                break
        updated.append(rec_copy)

    return updated, cluster_summaries


# ---------------------------------------------------------------------------
# Collaborative filtering — co-purchase signal
# ---------------------------------------------------------------------------
_COPURCHASE_CACHE: dict[str, dict[str, float]] | None = None


def _build_copurchase_matrix(plis_path: Path, progress_cb=None) -> dict[str, dict[str, float]]:
    """Build a simplified co-purchase model: for each eclass, which other eclasses
    are commonly bought by the same customers?

    Returns {eclass -> {other_eclass -> affinity_score}}.
    This is a lightweight collaborative filtering approach: instead of full
    matrix factorization, we build pairwise eclass affinities from customer
    co-purchase patterns.  If many customers who buy eclass A also buy eclass B,
    then B has high affinity with A.
    """
    global _COPURCHASE_CACHE
    if _COPURCHASE_CACHE is not None:
        return _COPURCHASE_CACHE

    if progress_cb:
        progress_cb("Building co-purchase matrix from transaction data…")

    # Step 1: Build customer → set[eclass] mapping
    customer_eclasses: dict[int, set[str]] = {}
    chunk_num = 0
    for chunk in pd.read_csv(
        plis_path,
        sep="\t",
        usecols=["legal_entity_id", "eclass"],
        dtype={"eclass": "string"},
        chunksize=500_000,
        low_memory=False,
    ):
        chunk_num += 1
        for cid, ec in zip(chunk["legal_entity_id"].values, chunk["eclass"].values):
            if pd.notna(cid) and pd.notna(ec) and str(ec).strip():
                customer_eclasses.setdefault(int(cid), set()).add(str(ec).strip())
        if progress_cb and chunk_num % 5 == 0:
            progress_cb(f"Co-purchase scan: {chunk_num * 500_000:,} rows, {len(customer_eclasses):,} customers…")

    if progress_cb:
        progress_cb(f"Co-purchase: {len(customer_eclasses):,} customers scanned, computing affinities…")

    # Step 2: Count co-occurrences (how many customers buy both eclass A and B)
    from collections import Counter as _Counter2
    cooccur: dict[str, _Counter2] = {}
    eclass_count: _Counter2 = _Counter2()

    for cid, eclasses in customer_eclasses.items():
        ec_list = sorted(eclasses)
        for ec in ec_list:
            eclass_count[ec] += 1
        # Only process customers with reasonable basket size (2-200 eclasses)
        if 2 <= len(ec_list) <= 200:
            for i, ec_a in enumerate(ec_list):
                cooccur.setdefault(ec_a, _Counter2())
                for ec_b in ec_list[i + 1:]:
                    cooccur[ec_a][ec_b] += 1
                    cooccur.setdefault(ec_b, _Counter2())[ec_a] += 1

    # Step 3: Normalize to affinity scores (Jaccard-like: co-occur / (count_a + count_b - co-occur))
    affinity: dict[str, dict[str, float]] = {}
    for ec_a, partners in cooccur.items():
        affinity[ec_a] = {}
        count_a = eclass_count[ec_a]
        for ec_b, co_count in partners.most_common(50):  # Keep top 50 affinities per eclass
            count_b = eclass_count[ec_b]
            denom = count_a + count_b - co_count
            if denom > 0:
                affinity[ec_a][ec_b] = round(co_count / denom, 4)

    _COPURCHASE_CACHE = affinity
    if progress_cb:
        progress_cb(f"Co-purchase matrix: {len(affinity):,} eclasses with affinities")
    return affinity


def _collab_score(
    candidate_eclass: str,
    user_eclasses: set[str],
    copurchase: dict[str, dict[str, float]],
) -> float:
    """Compute collaborative filtering score for a candidate.

    How strongly is the candidate's eclass associated with the eclasses
    this customer already buys?  Returns max affinity across user's eclasses.
    For same-eclass items, returns a moderate baseline (0.5) since
    co-purchase within the same category is expected.
    """
    if candidate_eclass in user_eclasses:
        return 0.5  # Baseline for same-eclass — these are expected purchases
    best = 0.0
    for user_ec in user_eclasses:
        af = copurchase.get(user_ec, {}).get(candidate_eclass, 0.0)
        if af > best:
            best = af
    return best


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
        usecols=["legal_entity_id", "set_id", "sku", "eclass", "manufacturer", "vk_per_item", "quantityvalue", "orderdate"],
        dtype={"set_id": "string", "sku": "string", "eclass": "string", "manufacturer": "string", "vk_per_item": "float64", "quantityvalue": "float64"},
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
    sku_df["set_id"] = sku_df["set_id"].astype("string") if "set_id" in sku_df.columns else pd.Series("", index=sku_df.index, dtype="string")
    sku_df["eclass"] = sku_df["eclass"].astype("string")
    sku_df["manufacturer"] = sku_df["manufacturer"].astype("string")
    catalogue_size = len(sku_df)
    _progress(1, f"Catalogue loaded: {catalogue_size:,} unique SKUs")

    feat_profiles: dict[str, dict] = {}
    if SKU_FEAT_PATH.exists():
        feat_profiles = _load_json(SKU_FEAT_PATH)

    setid_profiles: dict[str, dict] = {}
    if SETID_FEAT_PATH.exists():
        setid_profiles = _load_json(SETID_FEAT_PATH)
    _progress(1, f"Feature profiles: {len(feat_profiles):,} SKUs, {len(setid_profiles):,} set_ids")

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
    owned_set_ids = set(hist["set_id"].dropna().unique()) if "set_id" in hist.columns else set()
    user_eclass = set(hist["eclass"].dropna().unique())
    user_manufacturers = set(hist["manufacturer"].dropna().unique())

    if filter_eclass:
        user_eclass = user_eclass & {filter_eclass}
        if not user_eclass:
            raise ValueError(f"Customer {customer_id} has no history for e-class {filter_eclass}.")

    # 3. Candidate generation — include user's eclasses + co-purchase related eclasses
    _progress(3, "Building co-purchase matrix for collaborative filtering…")
    copurchase = _build_copurchase_matrix(
        plis_path,
        progress_cb=lambda msg: _progress(3, msg),
    )

    # Find related eclasses: high co-purchase affinity with user's eclasses
    related_eclasses: set[str] = set()
    for uec in user_eclass:
        for related_ec, affinity in copurchase.get(uec, {}).items():
            if related_ec not in user_eclass and affinity >= 0.05:
                related_eclasses.add(related_ec)
    all_eclasses = user_eclass | related_eclasses

    _progress(3, f"Generating candidates from {len(user_eclass)} user E-class(es) + {len(related_eclasses)} co-purchase related…")
    pool = sku_df[sku_df["eclass"].isin(all_eclasses)].copy()
    pool = pool[~pool["sku"].isin(owned_skus)].copy()
    # NOTE: We intentionally do NOT exclude candidates whose set_id the customer
    # already owns.  The same product from a *different* (cheaper) supplier is a
    # valid recommendation — "you buy this notebook from Store A for €50, but
    # Store B sells it for €30".  Dedup in the output (step 4) ensures no
    # duplicate set_ids appear in the final list, keeping only the cheapest SKU
    # per product.
    pool = pool.merge(ref_price, on="eclass", how="left")
    # For cross-category candidates (from co-purchase), use their own avg_price as ref
    pool["reference_price"] = pool["reference_price"].fillna(pool["avg_price"])
    pool["manufacturer_familiarity"] = pool["manufacturer"].isin(user_manufacturers).astype(float)
    max_pop = float(pool["purchase_count"].max()) if not pool.empty else 1.0
    pool["global_popularity"] = pool["purchase_count"] / max(max_pop, 1)
    pool = pool[pool["avg_price"] >= min_price].copy()
    candidate_count = len(pool)
    _progress(3, f"{candidate_count:,} candidates after filtering")

    # 4. Scoring — feature-similarity + collaborative filtering
    _progress(4, "Building customer feature profile & scoring candidates…")
    ref = pool["reference_price"].clip(lower=0.01)
    pool["price_advantage"] = ((ref - pool["avg_price"]) / ref).clip(lower=0.0)

    # Build customer's aggregated feature profile from their purchases
    customer_feat_profile = _build_customer_feature_profile(
        owned_skus, owned_set_ids, feat_profiles, setid_profiles,
    )
    _progress(4, f"Customer feature profile: {len(customer_feat_profile)} feature keys from purchases")

    # Feature similarity: compare candidate features against customer's feature profile
    def _get_candidate_features(row):
        """Get best available feature profile for a candidate (set_id > sku)."""
        sid = str(row["set_id"]) if pd.notna(row.get("set_id")) else ""
        sku = str(row["sku"])
        if sid and sid in setid_profiles:
            return setid_profiles[sid]
        return feat_profiles.get(sku, {})

    pool["_cand_features"] = pool.apply(_get_candidate_features, axis=1)
    pool["feature_similarity"] = pool["_cand_features"].apply(
        lambda cp: _feature_similarity(customer_feat_profile, cp)
    )
    pool["feature_count"] = pool["_cand_features"].apply(lambda cp: len(cp))
    max_feat = float(pool["feature_count"].max()) if not pool.empty else 1.0
    pool["feature_richness"] = pool["feature_count"] / max(max_feat, 1)

    # Collaborative filtering: co-purchase affinity between this candidate's
    # eclass and the eclasses this customer already buys
    _progress(4, "Computing collaborative filtering scores…")
    pool["collab_score"] = pool["eclass"].apply(
        lambda ec: _collab_score(ec, user_eclass, copurchase)
    )

    linear = (
        1.2 * pool["global_popularity"]
        + 0.8 * pool["manufacturer_familiarity"]
        + 1.5 * pool["price_advantage"]
        + 1.0 * pool["feature_similarity"]
        + 0.3 * pool["feature_richness"]
        + 0.6 * pool["collab_score"]
        - 0.2
    )
    pool["p_buy"] = _sigmoid(linear.to_numpy())
    pool["final_score"] = pool["p_buy"] * (1.0 + pool["price_advantage"])
    pool = pool.sort_values("final_score", ascending=False)

    # Deduplicate by set_id: keep only the best-scoring SKU per product
    if "set_id" in pool.columns:
        has_set_id = pool["set_id"].notna() & (pool["set_id"] != "")
        deduped_with = pool[has_set_id].drop_duplicates(subset="set_id", keep="first")
        without_set_id = pool[~has_set_id]
        pool = pd.concat([deduped_with, without_set_id], ignore_index=True)
        pool = pool.sort_values("final_score", ascending=False)
    _progress(4, f"Scored {candidate_count:,} candidates ({len(pool):,} unique products after set_id dedup)")

    # 5. Enrich with features
    _progress(5, "Enriching top candidates with product features…")
    top = pool.head(top_n).reset_index(drop=True)
    top["rank"] = top.index + 1
    top["features_summary"] = top["sku"].apply(lambda s: _summarise_features(s, feat_profiles))
    top["features_detail"] = top["_cand_features"]

    out_records = _build_output_records(top)

    # 6. Feature-cluster grouping — group recommendations by shared features
    _progress(5, "Building feature-cluster groups…")
    out_records, cluster_summaries = _build_feature_clusters(
        out_records, feat_profiles, setid_profiles,
    )

    total_saving = float((top["reference_price"] - top["avg_price"]).clip(lower=0).sum())
    _progress(5, f"Done — {len(out_records)} recommendations in {len(cluster_summaries)} feature clusters")

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
        "customer_feature_profile_keys": sorted(customer_feat_profile.keys())[:50],
        "customer_feature_profile_key_count": len(customer_feat_profile),
        "recommendations": out_records,
        "feature_clusters": cluster_summaries,
        "total_estimated_saving_eur": round(total_saving, 2),
        "scoring_formula": "P(buy) = sigmoid(1.2*popularity + 0.8*brand_fam + 1.5*price_adv + 1.0*feat_sim + 0.3*feat_rich + 0.6*collab - 0.2); score = P(buy) * (1 + price_adv)",
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
            "set_id": str(row["set_id"]) if "set_id" in df.columns and pd.notna(row.get("set_id")) else None,
            "eclass": str(row["eclass"]),
            "manufacturer": str(row["manufacturer"]),
            "candidate_price_eur": round(float(row["avg_price"]), 4),
            "reference_price_eur": round(float(row["reference_price"]), 4) if not pd.isna(row["reference_price"]) else None,
            "price_advantage_pct": round(float(row["price_advantage"]) * 100, 2),
            "global_popularity": round(float(row["global_popularity"]), 4) if "global_popularity" in df.columns else None,
            "manufacturer_familiarity": round(float(row["manufacturer_familiarity"]), 4) if "manufacturer_familiarity" in df.columns else None,
            "feature_similarity": round(float(row["feature_similarity"]), 4) if "feature_similarity" in df.columns else None,
            "feature_richness": round(float(row["feature_richness"]), 4) if "feature_richness" in df.columns else None,
            "feature_count": int(row["feature_count"]) if "feature_count" in df.columns else 0,
            "collab_score": round(float(row["collab_score"]), 4) if "collab_score" in df.columns else None,
            "p_buy": round(float(row["p_buy"]), 4),
            "final_score": round(float(row["final_score"]), 4),
            "features_summary": row["features_summary"],
            "features_detail": row["features_detail"] if "features_detail" in df.columns else {},
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
