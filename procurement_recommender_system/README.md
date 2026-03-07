# Procurement Recommender System

Recommend **cheaper-but-equivalent items** that a buyer is likely to purchase next, ranked by:

```
final_score = P(buy) × (1 + price_advantage)
```

---

## How to use & demonstrate

### Step 1 — Preprocess (run once, takes a few minutes)

Reads the real CSVs and writes aggregated JSON artefacts to `data/generated/`.

```bash
python src/csv_to_json_artifacts.py --mode fast --max-skus 10000
```

This produces:
- `data/generated/sku_aggregates_from_csv.json` — 2.7M unique SKUs with avg price and popularity
- `data/generated/eclass_manufacturer_aggregates_from_csv.json` — category-level stats
- `data/generated/feature_types_summary_from_csv.json` — 14 286 unique product feature types (color, size, material, dimension, …)
- `data/generated/sku_feature_profiles_sample_from_csv.json` — sample of per-SKU feature key-value pairs

> The source CSVs are never modified. Only `data/generated/` is written.

---

### Step 2 — Demo (instant, any time)

```bash
# Show top 20 buyers by transaction volume
python src/demo.py --list-customers

# Recommend 15 cheaper alternatives for a specific buyer
python src/demo.py --customer 41303727 --top 15

# Restrict to one product category (E-class)
python src/demo.py --customer 41303727 --eclass 29120102

# Raise minimum item price floor (default €0.50)
python src/demo.py --customer 41303727 --top 15 --min-price 2.0
```

The demo output shows a formatted table:

```
   #  SKU                     E-Class       Manufacturer          Cand. €     Ref. €   Saved%    Score  Features
   1  102-777398-BP           29120102      HAN                      3.94      81.63    85.0%   1.7683  …
```

And saves `outputs/demo_<customer_id>.json` for further use.

---

## Pipeline stages

```
plis_training.csv (18M rows)
        │
        │ csv_to_json_artifacts.py
        ▼
data/generated/sku_aggregates_from_csv.json (2.7M SKUs)
        │
        │ demo.py: load customer history from CSV
        ▼
Customer purchase history (warm buyer path)
        │
        │ candidate_generation: same E-class, not already owned
        ▼
Candidate pool (~1.8M items)
        │
        │ scoring: P(buy) × (1 + price_advantage)
        │          capped at 85% max savings
        ▼
Top N ranked recommendations
        │
        │ feature enrichment from sku_feature_profiles_sample.json
        ▼
outputs/demo_<customer_id>.json
```

---

## Score formula

```
price_advantage = clip((ref_price − candidate_price) / ref_price, 0, 0.85)

final_score = P(buy) × (1 + price_advantage)
```

Where `P(buy)` is estimated from:
- `global_popularity` — how often this SKU is bought across all customers
- `manufacturer_familiarity` — does the buyer already buy from this maker?
- `price_advantage` — how much cheaper is it vs what the buyer normally pays?

---

## What each file does

| File | Purpose |
|---|---|
| `src/csv_to_json_artifacts.py` | One-time preprocessing, reads real CSVs, writes JSON artefacts |
| `src/demo.py` | Demo CLI — takes a customer ID, prints ranked recommendations |
| `src/main.py` | Minimal pipeline run on sample fixtures |
| `config/source_data_paths.json` | Paths to real CSV files (no data is copied) |
| `config/scoring_weights.json` | Tunable score weights |
| `config/features.json` | Feature taxonomy |
| `data/contracts/` | JSON schemas for all data structures |
| `data/examples/` | Small synthetic fixtures |
| `data/generated/` | Real-data artefacts produced by preprocessing |
| `outputs/` | Recommendation results per customer |

