"""
Cold-Start Customer Matching
=============================
For customers with NO purchase history (cold start), find the most
similar warm customer from the training data and borrow their profile.

Similarity is based on:
  1. NACE code match (same industry = strong signal)
  2. Secondary NACE code match
  3. Employee count similarity (log-scaled)

The module builds a warm-customer feature table from plis_training.csv
(one row per unique customer with their NACE + employee features) and
uses nearest-neighbor lookup to map any cold customer to the best proxy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_CONFIG_PATH = BASE_DIR / "config" / "source_data_paths.json"
GEN_DIR = BASE_DIR / "data" / "generated"
WARM_PROFILES_PATH = GEN_DIR / "warm_customer_profiles.json"


def _load_source_config() -> dict:
    with SOURCE_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Build warm-customer profile table (one-time, cached to JSON)
# ---------------------------------------------------------------------------
def build_warm_profiles(force: bool = False, progress_cb=None) -> pd.DataFrame:
    """Extract one row per warm customer: nace, secondary_nace, employees."""
    if WARM_PROFILES_PATH.exists() and not force:
        data = json.loads(WARM_PROFILES_PATH.read_text(encoding="utf-8"))
        return pd.DataFrame(data)

    cfg = _load_source_config()
    plis_path = Path(cfg["plis_training_csv"])

    records: dict[int, dict] = {}
    chunk_num = 0
    for chunk in pd.read_csv(
        plis_path,
        sep="\t",
        usecols=["legal_entity_id", "nace_code", "secondary_nace_code",
                 "estimated_number_employees"],
        dtype={"legal_entity_id": "Int64", "nace_code": "float64",
               "secondary_nace_code": "float64",
               "estimated_number_employees": "float64"},
        chunksize=500_000,
        low_memory=False,
    ):
        chunk_num += 1
        # Take the first non-null value per customer (features are repeated)
        for _, row in chunk.drop_duplicates("legal_entity_id").iterrows():
            cid = int(row["legal_entity_id"])
            if cid in records:
                continue
            records[cid] = {
                "customer_id": cid,
                "nace_code": int(row["nace_code"]) if pd.notna(row["nace_code"]) else None,
                "secondary_nace_code": int(row["secondary_nace_code"]) if pd.notna(row["secondary_nace_code"]) else None,
                "estimated_number_employees": float(row["estimated_number_employees"]) if pd.notna(row["estimated_number_employees"]) else None,
            }
        if progress_cb and chunk_num % 4 == 0:
            progress_cb(f"Profiling warm customers… {len(records):,} found so far")

    df = pd.DataFrame(list(records.values()))

    # Cache to disk
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    WARM_PROFILES_PATH.write_text(
        json.dumps(list(records.values()), indent=None),
        encoding="utf-8",
    )
    if progress_cb:
        progress_cb(f"Built {len(df):,} warm customer profiles")
    return df


# ---------------------------------------------------------------------------
# Similarity scoring
# ---------------------------------------------------------------------------
def _nace_similarity(a_nace: int | None, a_sec: int | None,
                     b_nace, b_sec) -> float:
    """Score NACE match: exact 4-digit > 3-digit prefix > 2-digit > section."""
    if a_nace is None:
        return np.zeros(len(np.asarray(b_nace)))

    a_nace = int(a_nace)
    b_nace_arr = np.asarray(b_nace, dtype="float64")

    score = np.zeros(len(b_nace_arr))

    # Exact primary NACE match (4-digit)
    exact = b_nace_arr == a_nace
    score = np.where(exact, 1.0, score)

    # 3-digit prefix match
    a_3 = a_nace // 10
    prefix3 = (b_nace_arr // 10) == a_3
    score = np.where(prefix3 & ~exact, 0.7, score)

    # 2-digit prefix match
    a_2 = a_nace // 100
    prefix2 = (b_nace_arr // 100) == a_2
    score = np.where(prefix2 & ~prefix3 & ~exact, 0.4, score)

    # 1-digit section match
    a_1 = a_nace // 1000
    prefix1 = (b_nace_arr // 1000) == a_1
    score = np.where(prefix1 & ~prefix2 & ~prefix3 & ~exact, 0.15, score)

    # Secondary NACE bonus
    if a_sec is not None:
        b_sec_arr = np.asarray(b_sec, dtype="float64")
        sec_match = b_sec_arr == a_sec
        score = score + np.where(sec_match, 0.3, 0.0)

    return score  # type: ignore[return-value]


def _employee_similarity(a_emp: float | None, b_emp_arr) -> np.ndarray:
    """Log-ratio similarity: 1.0 when identical, decays with distance."""
    if a_emp is None or a_emp <= 0:
        return np.zeros(len(b_emp_arr))

    b = np.asarray(b_emp_arr, dtype="float64")
    b = np.where(np.isnan(b) | (b <= 0), 1.0, b)
    log_ratio = np.abs(np.log10(a_emp) - np.log10(b))
    return np.clip(1.0 - log_ratio / 3.0, 0.0, 1.0)


def find_closest_warm_customer(
    cold_nace: int | None,
    cold_secondary_nace: int | None,
    cold_employees: float | None,
    warm_df: pd.DataFrame,
    top_k: int = 5,
) -> list[dict]:
    """Return the top_k most similar warm customers with similarity scores."""
    nace_scores = _nace_similarity(
        cold_nace, cold_secondary_nace,
        warm_df["nace_code"].values,
        warm_df["secondary_nace_code"].values,
    )
    emp_scores = _employee_similarity(cold_employees, warm_df["estimated_number_employees"].values)

    # Weighted combination: industry is the primary signal
    combined = 0.65 * np.asarray(nace_scores) + 0.35 * np.asarray(emp_scores)

    top_idx = np.argsort(-combined)[:top_k]
    results = []
    for idx in top_idx:
        row = warm_df.iloc[idx]
        results.append({
            "customer_id": int(row["customer_id"]),
            "similarity": round(float(combined[idx]), 4),
            "nace_score": round(float(nace_scores[idx]), 4),
            "employee_score": round(float(emp_scores[idx]), 4),
            "nace_code": int(row["nace_code"]) if pd.notna(row["nace_code"]) else None,
            "secondary_nace_code": int(row["secondary_nace_code"]) if pd.notna(row["secondary_nace_code"]) else None,
            "estimated_number_employees": float(row["estimated_number_employees"]) if pd.notna(row["estimated_number_employees"]) else None,
        })
    return results


# ---------------------------------------------------------------------------
# Cold-start customer info loader
# ---------------------------------------------------------------------------
def load_cold_customer_features(customer_id: int) -> dict:
    """Load a cold customer's features from customer_test.csv."""
    cfg = _load_source_config()
    test_path = Path(cfg["customer_test_csv"])
    df = pd.read_csv(test_path, sep="\t", encoding="utf-8-sig")
    row = df[df["legal_entity_id"] == customer_id]
    if row.empty:
        raise ValueError(f"Customer {customer_id} not found in customer_test.csv")
    r = row.iloc[0]
    return {
        "customer_id": customer_id,
        "nace_code": int(r["nace_code"]) if pd.notna(r["nace_code"]) else None,
        "secondary_nace_code": int(r["secondary_nace_code"]) if pd.notna(r["secondary_nace_code"]) else None,
        "estimated_number_employees": float(r["estimated_number_employees"]) if pd.notna(r["estimated_number_employees"]) else None,
        "task": str(r["task"]).strip(),
    }


def load_nace_lookup() -> dict[int, str]:
    """Load NACE code descriptions for display."""
    cfg = _load_source_config()
    nace_path = Path(cfg.get("nace_codes_csv", ""))
    if not nace_path.exists():
        # Try standard location
        nace_path = Path(cfg["customer_test_csv"]).parent.parent / "nace_codes.csv" / "nace_codes.csv"
    if not nace_path.exists():
        return {}
    df = pd.read_csv(nace_path, sep="\t", encoding="utf-8-sig")
    return dict(zip(df["nace_code"].astype(int), df["n_nace_description"].astype(str)))
