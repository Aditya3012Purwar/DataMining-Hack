from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_CONFIG_PATH = BASE_DIR / "config" / "source_data_paths.json"
OUTPUT_DIR = BASE_DIR / "data" / "generated"
MAX_UNIQUE_FVALUES_PER_KEY = 20_000


def _load_source_paths(config_path: Path) -> dict[str, Path]:
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    paths = {k: Path(v) for k, v in cfg.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing source CSV paths: {missing}")
    return paths


def _normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype("string").str.strip()


def _group_feature_key(key: str) -> str:
    key_l = key.lower()

    # Prioritize material and dimensional terms before size to reduce misclassification.
    if any(tok in key_l for tok in ["material", "werkstoff", "nitril", "stahl", "steel", "cotton", "baumwolle", "poly", "aluminium", "holz", "wood", "metal"]):
        return "material"
    if any(tok in key_l for tok in ["dimension", "abmess", "laenge", "länge", "breite", "hoehe", "höhe", "durchmesser", "diameter", "mm", "cm", "meter"]):
        return "dimension"
    if any(tok in key_l for tok in ["color", "colour", "farbe", "farb", "ral"]):
        return "color"
    if any(tok in key_l for tok in ["size", "groesse", "größe", "handschuhgroesse", "schuhgroesse", "xl", "xxl", "xs", "xxs"]):
        return "size"
    return "other"


def generate_transaction_aggregates(plis_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    usecols = ["orderdate", "legal_entity_id", "sku", "eclass", "manufacturer", "quantityvalue", "vk_per_item"]
    dtypes = {
        "legal_entity_id": "Int64",
        "sku": "string",
        "eclass": "string",
        "manufacturer": "string",
        "quantityvalue": "float64",
        "vk_per_item": "float64",
    }

    sku_parts: list[pd.DataFrame] = []
    em_parts: list[pd.DataFrame] = []
    em_unique_skus: dict[tuple[str, str], set[str]] = defaultdict(set)
    customer_counts: dict[int, int] = defaultdict(int)

    for chunk in pd.read_csv(
        plis_path,
        sep="\t",
        usecols=usecols,
        dtype=dtypes,
        parse_dates=["orderdate"],
        chunksize=300_000,
        low_memory=False,
    ):
        chunk["sku"] = _normalize_series(chunk["sku"])
        chunk["eclass"] = _normalize_series(chunk["eclass"])
        chunk["manufacturer"] = _normalize_series(chunk["manufacturer"])

        chunk = chunk[(chunk["sku"] != "") & (chunk["eclass"] != "")]
        if chunk.empty:
            continue

        chunk["price_x_count"] = chunk["vk_per_item"]

        sku_partial = (
            chunk.groupby(["sku", "eclass", "manufacturer"], as_index=False, dropna=False)
            .agg(
                purchase_count=("sku", "size"),
                price_sum=("price_x_count", "sum"),
                total_quantity=("quantityvalue", "sum"),
                last_seen_date=("orderdate", "max"),
            )
        )
        sku_parts.append(sku_partial)

        em_partial = (
            chunk.groupby(["eclass", "manufacturer"], as_index=False, dropna=False)
            .agg(
                purchase_count=("eclass", "size"),
                price_sum=("price_x_count", "sum"),
                total_quantity=("quantityvalue", "sum"),
            )
        )
        em_parts.append(em_partial)

        # Track unique SKUs per (eclass, manufacturer) incrementally.
        em_sku_pairs = chunk[["eclass", "manufacturer", "sku"]].drop_duplicates()
        for row in em_sku_pairs.itertuples(index=False):
            em_unique_skus[(row.eclass, row.manufacturer)].add(row.sku)

        # Accumulate per-customer transaction counts.
        cid_counts = chunk["legal_entity_id"].dropna().value_counts()
        for cid, cnt in cid_counts.items():
            customer_counts[int(cid)] += int(cnt)

    sku_agg = pd.concat(sku_parts, ignore_index=True)
    sku_agg = (
        sku_agg.groupby(["sku", "eclass", "manufacturer"], as_index=False, dropna=False)
        .agg(
            purchase_count=("purchase_count", "sum"),
            price_sum=("price_sum", "sum"),
            total_quantity=("total_quantity", "sum"),
            last_seen_date=("last_seen_date", "max"),
        )
        .sort_values("purchase_count", ascending=False)
    )
    sku_agg["avg_price"] = sku_agg["price_sum"] / sku_agg["purchase_count"].clip(lower=1)
    sku_agg = sku_agg[["sku", "eclass", "manufacturer", "avg_price", "purchase_count", "total_quantity", "last_seen_date"]]

    eclass_manufacturer_agg = pd.concat(em_parts, ignore_index=True)
    eclass_manufacturer_agg = (
        eclass_manufacturer_agg.groupby(["eclass", "manufacturer"], as_index=False, dropna=False)
        .agg(
            purchase_count=("purchase_count", "sum"),
            price_sum=("price_sum", "sum"),
            total_quantity=("total_quantity", "sum"),
        )
        .sort_values("purchase_count", ascending=False)
    )
    eclass_manufacturer_agg["avg_price"] = (
        eclass_manufacturer_agg["price_sum"] / eclass_manufacturer_agg["purchase_count"].clip(lower=1)
    )
    eclass_manufacturer_agg["unique_skus"] = eclass_manufacturer_agg.apply(
        lambda row: len(em_unique_skus.get((row["eclass"], row["manufacturer"]), set())), axis=1
    )
    eclass_manufacturer_agg = eclass_manufacturer_agg[
        ["eclass", "manufacturer", "avg_price", "purchase_count", "total_quantity", "unique_skus"]
    ]

    customer_agg = sorted(
        [{"customer_id": k, "transaction_count": v} for k, v in customer_counts.items()],
        key=lambda x: x["transaction_count"],
        reverse=True,
    )
    return sku_agg, eclass_manufacturer_agg, customer_agg


def _init_feature_accumulators() -> tuple[
    set[str],
    dict[str, int],
    dict[str, set[str]],
    dict[str, Counter[str]],
    dict[str, set[str]],
    dict[str, dict[str, set[str]]],
    dict[str, bool],
]:
    unique_keys: set[str] = set()
    key_counts: dict[str, int] = defaultdict(int)
    key_unique_values: dict[str, set[str]] = defaultdict(set)
    key_value_samples: dict[str, Counter[str]] = defaultdict(Counter)
    grouped_keys: dict[str, set[str]] = defaultdict(set)
    sku_profiles: dict[str, dict[str, set[str]]] = {}
    unique_values_truncated: dict[str, bool] = defaultdict(bool)
    return (
        unique_keys,
        key_counts,
        key_unique_values,
        key_value_samples,
        grouped_keys,
        sku_profiles,
        unique_values_truncated,
    )


def generate_feature_summaries_and_profiles(
    features_path: Path,
    max_skus: int = 10_000,
    chunksize: int = 300_000,
    fast_mode: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, list[str]]]]:
    (
        unique_keys,
        key_counts,
        key_unique_values,
        key_value_samples,
        grouped_keys,
        sku_profiles,
        unique_values_truncated,
    ) = _init_feature_accumulators()

    total_rows = 0
    rows_with_value = 0

    for chunk in pd.read_csv(features_path, sep="\t", dtype="string", chunksize=chunksize, low_memory=False):
        total_rows += len(chunk)
        chunk["sku"] = _normalize_series(chunk["sku"])
        chunk["key"] = _normalize_series(chunk["key"])
        chunk["fvalue"] = _normalize_series(chunk["fvalue"])

        non_empty_key = chunk["key"] != ""
        chunk_k = chunk.loc[non_empty_key, ["sku", "key", "fvalue"]]
        if chunk_k.empty:
            continue

        chunk_keys = chunk_k["key"].unique().tolist()
        unique_keys.update(chunk_keys)
        for key in chunk_keys:
            grouped_keys[_group_feature_key(key)].add(key)

        key_count_series = chunk_k["key"].value_counts(dropna=False)
        for key, count in key_count_series.items():
            key_counts[str(key)] += int(count)

        with_value = chunk_k[chunk_k["fvalue"] != ""]
        rows_with_value += int(len(with_value))

        if not with_value.empty:
            kv_counts = with_value.groupby(["key", "fvalue"]).size().rename("cnt").reset_index()
            for row in kv_counts.itertuples(index=False):
                key = row.key
                fvalue = row.fvalue
                cnt = int(row.cnt)

                if not fast_mode:
                    if len(key_unique_values[key]) < MAX_UNIQUE_FVALUES_PER_KEY:
                        key_unique_values[key].add(fvalue)
                    else:
                        unique_values_truncated[key] = True

                if len(key_value_samples[key]) < 50 or fvalue in key_value_samples[key]:
                    key_value_samples[key][fvalue] += cnt

            # Build a bounded sample of SKU feature profiles in the same pass.
            sku_kv = with_value[with_value["sku"] != ""]["sku key fvalue".split()]
            existing_skus = set(sku_profiles.keys())
            if len(existing_skus) >= max_skus:
                sku_kv = sku_kv[sku_kv["sku"].isin(existing_skus)]
            else:
                remaining = max_skus - len(existing_skus)
                new_skus = sku_kv.loc[~sku_kv["sku"].isin(existing_skus), "sku"].drop_duplicates().head(remaining)
                allowed_skus = existing_skus.union(set(new_skus.tolist()))
                sku_kv = sku_kv[sku_kv["sku"].isin(allowed_skus)]

            sku_kv = sku_kv.drop_duplicates()
            for row in sku_kv.itertuples(index=False):
                sku = row.sku
                sku_profiles.setdefault(sku, {}).setdefault(row.key, set()).add(row.fvalue)

    per_key_stats = []
    for key in sorted(unique_keys):
        top_values = [value for value, _ in key_value_samples[key].most_common(10)]
        per_key_stats.append(
            {
                "key": key,
                "row_count": int(key_counts[key]),
                "unique_fvalue_count": None if fast_mode else int(len(key_unique_values[key])),
                "unique_fvalue_count_truncated": bool(unique_values_truncated[key]) if not fast_mode else False,
                "sample_top_fvalues": top_values,
                "group": _group_feature_key(key),
            }
        )

    summary = {
        "total_feature_rows": int(total_rows),
        "rows_with_non_empty_fvalue": int(rows_with_value),
        "unique_feature_keys_count": int(len(unique_keys)),
        "unique_feature_keys": sorted(unique_keys),
        "feature_key_groups": {
            group: sorted(keys) for group, keys in grouped_keys.items()
        },
        "per_key_stats": per_key_stats,
    }

    compact_profiles: dict[str, dict[str, list[str]]] = {}
    for sku, features in sku_profiles.items():
        compact_profiles[sku] = {k: sorted(list(v))[:10] for k, v in features.items()}

    return summary, compact_profiles


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    if "last_seen_date" in out.columns:
        out["last_seen_date"] = out["last_seen_date"].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_json_artifacts(fast_mode: bool = True, max_skus: int = 10_000) -> None:
    paths = _load_source_paths(SOURCE_CONFIG_PATH)

    sku_agg_df, eclass_manu_df, customer_agg = generate_transaction_aggregates(paths["plis_training_csv"])
    feature_summary, sku_feature_profiles = generate_feature_summaries_and_profiles(
        paths["features_per_sku_csv"], max_skus=max_skus, fast_mode=fast_mode
    )

    _write_json(OUTPUT_DIR / "sku_aggregates_from_csv.json", _df_to_records(sku_agg_df))
    _write_json(OUTPUT_DIR / "customer_aggregates_from_csv.json", customer_agg)
    _write_json(OUTPUT_DIR / "eclass_manufacturer_aggregates_from_csv.json", _df_to_records(eclass_manu_df))
    _write_json(OUTPUT_DIR / "feature_types_summary_from_csv.json", feature_summary)
    _write_json(OUTPUT_DIR / "sku_feature_profiles_sample_from_csv.json", sku_feature_profiles)

    metadata = {
        "source_paths": {k: str(v) for k, v in paths.items()},
        "run_mode": "fast" if fast_mode else "full",
        "max_sku_profiles": int(max_skus),
        "generated_files": [
            "sku_aggregates_from_csv.json",
            "eclass_manufacturer_aggregates_from_csv.json",
            "feature_types_summary_from_csv.json",
            "sku_feature_profiles_sample_from_csv.json",
        ],
        "notes": [
            "All JSON files were generated from the real CSVs by path reference only.",
            "No files outside procurement_recommender_system were modified.",
            "feature_types_summary_from_csv.json contains unique feature keys from the full features CSV.",
        ],
    }
    _write_json(OUTPUT_DIR / "generation_metadata.json", metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate JSON artifacts from challenge CSV files.")
    parser.add_argument("--mode", choices=["fast", "full"], default="fast", help="fast: speed-optimized, full: exact per-key unique fvalue counts")
    parser.add_argument("--max-skus", type=int, default=10000, help="Maximum number of SKU feature profiles to persist")
    args = parser.parse_args()

    build_json_artifacts(fast_mode=args.mode == "fast", max_skus=args.max_skus)
    print(f"Generated JSON artifacts in: {OUTPUT_DIR}")
