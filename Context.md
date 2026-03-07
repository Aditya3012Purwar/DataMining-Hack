# Demandra — Core Demand Prediction & Value Optimization

## Executive Summary

**Demandra** is an AI-powered procurement intelligence platform that predicts which products a buyer will need on a recurring basis, constructs an economically optimal portfolio of those needs, and recommends cheaper-but-equivalent alternatives. It transforms messy, real-world B2B procurement data (8M+ transactions, 64K buyers, 2.7M SKUs) into actionable demand forecasts that maximize net economic benefit.

The system solves an economic portfolio optimization problem: each predicted "Core Demand" element generates savings when correct but incurs a fixed monthly fee — so the objective is not just prediction accuracy but **net euro value**.

```
Score = Sum(Savings from correct predictions) - Sum(Fees from all predictions)
```

---

## Problem Statement

In B2B procurement, a small set of recurring needs accounts for the majority of purchasing volume, while a long tail of ad-hoc purchases occurs irregularly. The challenge is to:

1. **Identify which needs are truly recurring** (Core Demand) versus one-off (Long Tail)
2. **Optimize the portfolio** — recommending too much incurs excessive fees; recommending too little misses savings
3. **Handle cold-start buyers** — new customers with no transaction history
4. **Operate at multiple abstraction levels** — from product category down to exact product specifications

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DEMANDRA PLATFORM                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Raw Data     │  │  NACE Codes  │  │  Product Features        │  │
│  │  8.3M rows    │  │  Industry    │  │  18M attribute rows      │  │
│  │  plis_training│  │  taxonomy    │  │  features_per_sku        │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────────┘  │
│         │                 │                      │                  │
│         ▼                 ▼                      ▼                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │              DATA PREPROCESSING & CLEANING                   │   │
│  │  • Filter valid 8-digit E-Class codes                       │   │
│  │  • Parse dates, compute spend = quantity × price            │   │
│  │  • Normalize manufacturer names                              │   │
│  │  • Map SKUs to feature signatures                            │   │
│  └──────────────────────┬──────────────────────────────────────┘   │
│                         │                                           │
│         ┌───────────────┼───────────────┐                          │
│         ▼               ▼               ▼                          │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐               │
│  │ LEVEL 1    │  │ LEVEL 2    │  │ LEVEL 3         │               │
│  │ E-Class    │  │ E-Class +  │  │ E-Class +       │               │
│  │ Categories │  │ Manufacturer│ │ Feature Clusters│               │
│  └─────┬──────┘  └─────┬──────┘  └──────┬─────────┘               │
│        │               │                │                          │
│        ▼               ▼                ▼                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                PREDICTION ENGINE                             │   │
│  │                                                              │   │
│  │   ┌─────────────┐          ┌──────────────┐                 │   │
│  │   │ WARM-START  │          │  COLD-START   │                 │   │
│  │   │ RFM Scoring │          │  NACE-based   │                 │   │
│  │   │ Economic    │          │  Similarity   │                 │   │
│  │   │ Filter      │          │  Matching     │                 │   │
│  │   └──────┬──────┘          └──────┬───────┘                 │   │
│  │          │                        │                          │   │
│  │          ▼                        ▼                          │   │
│  │   ┌─────────────────────────────────────────┐               │   │
│  │   │     PORTFOLIO OPTIMIZER                  │               │   │
│  │   │  Net Benefit = Savings - Fee per item    │               │   │
│  │   │  Include only if net_benefit > 0         │               │   │
│  │   └──────────────┬──────────────────────────┘               │   │
│  └──────────────────┼──────────────────────────────────────────┘   │
│                     ▼                                               │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               SUBMISSION (CSV)                               │   │
│  │   buyer_id, predicted_id                                     │   │
│  │   (one row per predicted Core Demand element)                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │          PROCUREMENT RECOMMENDER SYSTEM                      │   │
│  │   • Candidate generation (same E-Class, different SKU)      │   │
│  │   • Price-aware ranking: score = P(buy) × (1 + savings%)   │   │
│  │   • Flask web dashboard for interactive exploration          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.10+ | Core implementation |
| **Data Processing** | Pandas, NumPy | DataFrame operations, vectorized computation |
| **Machine Learning** | scikit-learn | Clustering, similarity computation |
| **Scientific Computing** | SciPy | Statistical functions, optimization |
| **Web Framework** | Flask | Interactive recommendation dashboard |
| **Serialization** | JSON | Config, data contracts, cached artifacts |
| **Version Control** | Git / GitHub | Collaboration and code management |

### Key Libraries (from `requirements.txt`)
```
pandas >= 2.0.0
numpy >= 1.24.0
scikit-learn >= 1.3.0
scipy >= 1.17.0
flask >= 3.0.0
joblib >= 1.5.0
```

---

## Data Pipeline

### Input Data

| File | Size | Description |
|------|------|-------------|
| `plis_training.csv` | 8.3M rows | Historical procurement transactions (2023-01 to 2025-06) |
| `customer_test.csv` | 100 rows | Test buyers — 48 warm-start, 52 cold-start |
| `nace_codes.csv` | ~900 rows | Industry classification codes with descriptions |
| `features_per_sku.csv` | 18M rows | Product feature attributes (size, material, color, etc.) |

### Data Schema (plis_training.csv)

| Column | Type | Description |
|--------|------|-------------|
| `orderdate` | date | Transaction date |
| `legal_entity_id` | int | Buyer identifier |
| `set_id` | string | Order identifier |
| `sku` | string | Product SKU |
| `eclass` | string | 8-digit product category code |
| `manufacturer` | string | Product manufacturer |
| `quantityvalue` | float | Order quantity |
| `vk_per_item` | float | Unit price (EUR) |
| `nace_code` | string | Buyer's industry code |
| `secondary_nace_code` | string | Buyer's secondary industry code |
| `estimated_number_employees` | float | Company size |

### Data Characteristics (real-world messiness)
- Duplicate products across different SKUs
- Missing attributes and inconsistent descriptions
- Multiple SKUs representing the same functional need
- 6,225 unique E-Class categories across 64,898 buyers

---

## Module Descriptions

### Module 1: E-Class Level Prediction (`milestone1_eclass/`)

**Objective**: Predict recurring needs at the functional product category level.

**Algorithm — Warm-Start (RFM + Economic Filter)**:

```
                  ┌──────────────────────────┐
                  │  Buyer Transaction History │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  Per-EClass Aggregation    │
                  │  • order_count             │
                  │  • total_spend             │
                  │  • quarters_active         │
                  │  • recency_days            │
                  │  • avg_price               │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  Recurrence Scoring        │
                  │                            │
                  │  score = 0.20 × freq_norm  │
                  │        + 0.25 × consistency│
                  │        + 0.35 × monetary   │
                  │        + 0.20 × recency    │
                  └────────────┬─────────────┘
                               │
                  ┌────────────▼─────────────┐
                  │  Economic Filter           │
                  │                            │
                  │  savings = √(price) × freq │
                  │            × multiplier    │
                  │  net = savings - annual_fee │
                  │                            │
                  │  Include if:               │
                  │    score ≥ 0.50 AND        │
                  │    net > 0 AND             │
                  │    quarters ≥ 2            │
                  └────────────┬─────────────┘
                               │
                               ▼
                      Selected E-Classes
```

**Algorithm — Cold-Start (NACE-based Similarity)**:

```
  Cold Buyer Profile          Training Buyers
  ┌──────────────┐           ┌──────────────┐
  │ NACE code    │           │ NACE code    │
  │ Employees    │           │ Employees    │
  └──────┬───────┘           └──────┬───────┘
         │                          │
         └──────────┬───────────────┘
                    │
         ┌──────────▼───────────┐
         │  Similarity Scoring   │
         │  • Exact NACE = 4pts  │
         │  • 3-digit  = 3pts   │
         │  • 2-digit  = 2pts   │
         │  • Size sim (log)    │
         └──────────┬───────────┘
                    │
         ┌──────────▼───────────┐
         │  Top-50 Similar Buyers│
         │  Aggregate Purchases  │
         │  Filter: penetration  │
         │  ≥ 30% of peers buy it│
         └──────────┬───────────┘
                    │
                    ▼
           Top-20 E-Classes
```

**Key Metrics (validated via temporal split)**:
- ~90% hit rate on warm-start predictions
- ~261 predictions per warm buyer (economics-filtered, no hard cap)
- ~15 predictions per cold buyer (conservative)

**Files**:
| File | Purpose |
|------|---------|
| `solution_level1.py` | Main prediction pipeline |
| `validate_and_optimize.py` | Temporal validation framework and parameter sweep (14,400 configs tested) |
| `data_exploration.py` | Data analysis and statistics |
| `submission.csv` | Generated predictions |

---

### Module 2: E-Class + Manufacturer Prediction (`milestone2_eclass_manufacturer/`)

**Objective**: Add brand/supplier specificity to category-level predictions.

**Key Difference from Level 1**: Predictions are `eclass|manufacturer` pairs (e.g., `27050401|Siemens`). Higher precision but more noise-sensitive.

**Algorithm — Warm-Start (Tiered Selection)**:

```
  Tier 1: High Expected Value
  ├── expected_savings = 10% × monthly_spend × purchase_prob
  ├── expected_net = expected_savings - €10 fee
  └── Include if expected_net > 0

  Tier 2: Recently Active
  ├── Bought in 2+ of last 6 months
  └── Recent monthly spend ≥ €50

  Tier 3: Consistently Recurring
  ├── Bought in 30%+ of months
  ├── Monthly spend ≥ €30
  └── Last purchase within 6 months

  Final = Union(Tier1, Tier2, Tier3), capped at 150/buyer
```

**Cold-Start Strategy**: Skipped entirely — at L2 granularity the expected value per prediction is negative for cold-start buyers.

**Files**:
| File | Purpose |
|------|---------|
| `solution_level2.py` | Main prediction pipeline with tiered warm-start logic |
| `data_exploration.py` | Data analysis |
| `submission.csv` | Generated predictions |

---

### Module 3: E-Class + Feature Clusters (`milestone3_feature_clusters/`)

**Objective**: Predict product *requirements* rather than exact products. Uses feature engineering and hash-based clustering.

**Clustering Workflow**:

```
  features_per_sku.csv (18M rows)
          │
          ▼
  Map SKUs to E-Classes via training data
          │
          ▼
  Identify Top-5 Feature Keys per E-Class
  (most common: size, material, voltage, color, etc.)
          │
          ▼
  Build Feature Signature per SKU
  e.g., "Material=Kupfer|Spannung=230V|Farbe=schwarz"
          │
          ▼
  Hash-Based Clustering (MD5 of signature)
  Cluster ID = "{eclass}__{hash8}"
  e.g., "27050401__a3f8b2c1"
          │
          ▼
  6,225+ clusters across all E-Classes
```

**Methodological Highlights**:
- Feature-based abstraction captures **product requirements**, not specific SKUs
- Hash-based signatures are deterministic and reproducible
- Multiple SKUs with identical features collapse into one cluster (handles duplicates)
- Portfolio discipline: prefer specific clusters over `__default` catch-all

**Files**:
| File | Purpose |
|------|---------|
| `solution_level3.py` | Clustering + prediction pipeline |
| `data_exploration.py` | Data analysis |
| `submission.csv` | Generated predictions |
| `cluster_mapping.csv` | Maps cluster_id to E-Class + feature signature |

---

### Module 4: Procurement Recommender System (`procurement_recommender_system/`)

**Objective**: Recommend **cheaper-but-equivalent** products a buyer is likely to purchase next.

**Pipeline**:

```
  plis_training.csv (8.3M rows)
          │
          │  csv_to_json_artifacts.py (one-time preprocessing)
          ▼
  SKU Aggregates (2.7M SKUs: avg price, popularity)
          │
          │  demo.py: load customer history
          ▼
  Customer Purchase History
          │
          │  candidate_generation.py: same E-Class, not already owned
          ▼
  Candidate Pool
          │
          │  ranking.py: P(buy) × (1 + price_advantage)
          ▼
  Top-N Ranked Recommendations
          │
          │  Feature enrichment
          ▼
  outputs/demo_{customer_id}.json
```

**Scoring Formula**:
```
price_advantage = clip((reference_price - candidate_price) / reference_price, 0, 0.85)

P(buy) = sigmoid(1.2 × global_popularity + 1.0 × manufacturer_familiarity
                 + 1.5 × price_advantage - 0.2)

final_score = P(buy) × (1 + price_advantage)
```

**Cold-Start Matching**: For buyers with no history, the system finds the most similar warm buyer using a weighted combination of:
- **NACE code similarity** (65% weight): exact 4-digit > 3-digit prefix > 2-digit > section
- **Employee count similarity** (35% weight): log-ratio distance

**Web Dashboard** (Flask):
- Interactive customer selection
- Real-time recommendation generation with progress tracking
- Filterable by E-Class and minimum price
- Async job processing via threading

**Files**:
| File | Purpose |
|------|---------|
| `src/demo.py` | CLI demo — recommend alternatives for any buyer |
| `src/webapp.py` | Flask web application (port 5000) |
| `src/csv_to_json_artifacts.py` | One-time CSV-to-JSON preprocessing |
| `src/candidate_generation.py` | Candidate pool generation (same E-Class, different SKU) |
| `src/ranking.py` | Sigmoid-based scoring and ranking |
| `src/cold_start.py` | NACE + employee similarity matching |
| `src/main.py` | Minimal pipeline runner |
| `config/` | Scoring weights, feature taxonomy, data paths |
| `data/contracts/` | JSON schemas for all data structures |
| `data/examples/` | Sample fixtures for testing |
| `outputs/` | Generated recommendation JSON files |

---

## Folder Structure

```
DataMining-Hack/
│
├── Context.md                          # This file — project overview
├── README.md                           # Repository description
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git exclusion rules
├── transcribed_meeting.txt             # Meeting notes / problem description
│
├── Challenge2/                         # Raw challenge data (not in git)
│   ├── README_Core_Demand_Challenge.md # Official challenge specification
│   ├── plis_training.csv(.gz)          # 8.3M transaction rows
│   ├── customer_test.csv(.gz)          # 100 test buyers
│   ├── nace_codes.csv(.gz)             # Industry classification
│   └── features_per_sku.csv(.gz)       # 18M product feature rows
│
├── milestone1_eclass/                  # Level 1: E-Class predictions
│   ├── solution_level1.py              # Main pipeline (optimized)
│   ├── validate_and_optimize.py        # Parameter sweep framework
│   ├── data_exploration.py             # Data analysis utilities
│   └── submission.csv                  # Output predictions
│
├── milestone2_eclass_manufacturer/     # Level 2: E-Class + Manufacturer
│   ├── solution_level2.py              # Tiered warm-start pipeline
│   ├── data_exploration.py             # Data analysis utilities
│   └── submission.csv                  # Output predictions
│
├── milestone3_feature_clusters/        # Level 3: E-Class + Feature Clusters
│   ├── solution_level3.py              # Clustering + prediction pipeline
│   ├── data_exploration.py             # Data analysis utilities
│   ├── submission.csv                  # Output predictions
│   └── cluster_mapping.csv             # Cluster ID documentation
│
└── procurement_recommender_system/     # Cheaper-alternative recommender
    ├── README.md                       # Module documentation
    ├── requirements.txt                # Module-specific dependencies
    ├── config/                         # Configuration files
    │   ├── scoring_weights.json        # Tunable score parameters
    │   ├── features.json               # Feature taxonomy
    │   ├── pipeline.json               # Pipeline config
    │   └── source_data_paths.json      # Paths to source CSVs
    ├── data/
    │   ├── contracts/                  # JSON schemas (5 schemas)
    │   ├── examples/                   # Sample fixtures for testing
    │   └── generated/                  # Preprocessed artifacts (not in git)
    ├── docs/
    │   └── architecture.md             # Architecture notes
    ├── sql/
    │   └── aggregate_transactions.sql  # SQL aggregation reference
    ├── src/
    │   ├── main.py                     # Pipeline runner
    │   ├── demo.py                     # CLI recommendation demo
    │   ├── webapp.py                   # Flask web dashboard
    │   ├── csv_to_json_artifacts.py    # One-time preprocessing
    │   ├── candidate_generation.py     # Candidate pool builder
    │   ├── ranking.py                  # Scoring & ranking engine
    │   ├── cold_start.py              # Cold-start similarity matching
    │   ├── aggregate.py                # Transaction aggregation
    │   └── templates/
    │       └── index.html              # Web dashboard UI
    └── outputs/                        # Generated recommendations (JSON)
```

---

## Algorithms & Techniques

### 1. Monetary-Weighted RFM Scoring
Extended RFM (Recency, Frequency, Monetary) with consistency weighting. Each buyer-product pair is scored across four normalized dimensions, with weights optimized via grid search over 14,400 parameter combinations.

### 2. Temporal Validation Framework
Instead of random train/test splits (which would leak temporal information), the system uses a strict temporal split:
- **Train**: 2023-01 to 2024-06
- **Validate**: 2024-07 to 2025-06

This simulates the real scenario: predict future demand using only past data.

### 3. Economic Portfolio Optimization
Every prediction is evaluated through a cost-benefit lens:
```
net_benefit = estimated_savings - annual_fee
```
Only items with positive net benefit AND sufficient recurrence evidence are included. This prevents fee bloat from low-confidence predictions.

### 4. NACE-Based Cold-Start Similarity
For buyers with no history, the system uses industry classification (NACE codes) with hierarchical matching:
- Exact code match (4-digit): highest confidence
- Progressive prefix fallback: 3-digit, 2-digit, section level
- Combined with company size similarity (log-scale employee count)

### 5. Hash-Based Feature Clustering (Level 3)
Product features are distilled into deterministic signatures using the top-N most common feature keys per E-Class, then clustered via MD5 hashing. This handles catalog duplicates and creates stable, reproducible groups.

### 6. Price-Aware Recommendation Scoring
The recommender uses a sigmoid-based purchase probability model combined with price advantage:
```
final_score = P(buy) × (1 + price_advantage)
```
This ranks items that are both **likely to be purchased** and **cheaper than current alternatives**.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| No hard cap on warm predictions | Let economics (net_benefit > 0) decide portfolio size — validated 90% hit rate |
| Conservative cold-start (15-20 items) | Low confidence means high fee risk; keep portfolio small |
| Skip cold-start at Level 2 | Manufacturer-level predictions are too noisy without history |
| Monetary-focused weights (0.35) | High-spend recurring items generate the most savings |
| Temporal validation split | Prevents data leakage; simulates real-world prediction scenario |
| Hash-based clustering over ML clustering | Deterministic, reproducible, no hyperparameter tuning needed |
| Tiered selection in Level 2 | Captures different buyer patterns: high-value, recently active, consistently recurring |

---

## How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Generate Predictions (Levels 1-3)
```bash
# Level 1: E-Class predictions (~2 min)
cd milestone1_eclass && python solution_level1.py

# Level 2: E-Class + Manufacturer (~2 min)
cd milestone2_eclass_manufacturer && python solution_level2.py

# Level 3: Feature Clusters (~5 min)
cd milestone3_feature_clusters && python solution_level3.py
```

### Run Parameter Optimization (Level 1)
```bash
cd milestone1_eclass && python validate_and_optimize.py
```

### Launch Recommender Dashboard
```bash
cd procurement_recommender_system
python src/csv_to_json_artifacts.py --mode fast --max-skus 10000  # one-time preprocessing
python src/webapp.py  # opens at http://localhost:5000
```

### CLI Recommendations
```bash
cd procurement_recommender_system
python src/demo.py --customer 41303727 --top 15
```

---

## Results Summary

| Level | Metric | Value |
|-------|--------|-------|
| **Level 1** | Warm-start hit rate (validation) | ~90% |
| **Level 1** | Avg predictions per warm buyer | ~261 |
| **Level 1** | Avg predictions per cold buyer | ~15 |
| **Level 1** | Estimated net score | ~€2M (avg across scoring scenarios) |
| **Level 2** | Warm buyers covered | 47/48 |
| **Level 2** | Selection tiers | 3 (high-value, recent, recurring) |
| **Level 3** | Total feature clusters | 6,225+ |
| **Level 3** | Clustering method | Hash-based (deterministic) |
| **Recommender** | SKUs in catalogue | 2.7M |
| **Recommender** | Score formula | P(buy) × (1 + price_advantage) |

---

## Team Workflow

```
  Raw Challenge Data
        │
        ├──▶ Data Exploration (per-milestone data_exploration.py)
        │
        ├──▶ Feature Engineering & Algorithm Design
        │         │
        │         ├── Level 1: RFM + Economic Filter
        │         ├── Level 2: Tiered Selection + Recency Decay
        │         └── Level 3: Feature Clustering + Portfolio Discipline
        │
        ├──▶ Validation & Parameter Optimization
        │         └── Temporal split + grid search (14,400 configs)
        │
        ├──▶ Submission Generation (submission.csv per level)
        │
        └──▶ Recommender System (interactive demo + web dashboard)
```

---

*Built for the Unite Data Mining Hackathon — Core Demand Prediction & Value Optimization Challenge.*
