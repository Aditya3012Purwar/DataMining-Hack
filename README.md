# Demandra

**Demandra** is our procurement intelligence platform built for **Challenge 2 of the Unite x TUM.ai Hackathon**, where it won **1st place**.

It predicts recurring procurement needs, optimizes them for **net economic value**, and recommends cheaper-but-equivalent alternatives. Instead of simply forecasting what buyers may purchase, Demandra identifies which needs are truly worth including in a buyer's **Core Demand portfolio**, where each correct prediction creates savings but every predicted item also carries a fee.

---

## The Problem

In B2B procurement, buyers have two types of demand:

- **Core Demand**: recurring, predictable, high-value needs
- **Long Tail Demand**: irregular, ad-hoc purchases

The challenge is not just predicting future purchases, but deciding **which recurring needs are economically meaningful enough to justify a recurring fee**.

```text
Score = Savings - Fees
```

If we recommend too many items, fees explode.
If we recommend too few, we miss savings.

**Demandra solves this portfolio optimization problem.**

---

## What Demandra Does

Demandra works at **three levels of abstraction**:

### 1. E-Class Level
Predicts recurring demand at the functional category level.

Example:
- office paper
- nitrile gloves
- printer toner

### 2. E-Class + Manufacturer Level
Adds brand specificity to category-level demand.

Example:
- office paper from HP
- gloves from Ansell

### 3. E-Class + Feature Cluster Level
Predicts product **requirements**, not just categories or brands.

Example:
- protective gloves + nitrile + powder-free + size L

---

## Project Highlights

- Built for **Challenge 2 of the Unite x TUM.ai Hackathon**
- Won **1st place**
- Handles both **warm-start** and **cold-start** buyers
- Optimizes predictions using **economic benefit, not just accuracy**
- Uses **feature clustering** to group similar products into stable requirement-based clusters
- Includes a **procurement recommender system** for suggesting cheaper-but-equivalent alternatives

---

## How It Works

```text
Raw Procurement Data
        │
        ▼
Buyer + Product + Feature Processing
        │
        ├── Milestone 1: E-Class demand prediction
        ├── Milestone 2: E-Class + Manufacturer prediction
        ├── Milestone 3: Feature-cluster demand prediction
        │
        ▼
Economic Filtering
(keep only predictions with positive expected value)
        │
        ▼
Submission / Recommendations
```

---

## Core Ideas

### Warm-Start Buyers
For buyers with transaction history, Demandra identifies recurring needs using:
- **frequency**
- **consistency across time**
- **monetary value**
- **recency**

This is combined with an **economic filter** so only predictions with positive expected benefit are selected.

### Cold-Start Buyers
For buyers with little or no history, Demandra finds similar buyers using:
- **NACE industry codes**
- **secondary industry information**
- **company size**

Then it borrows demand patterns from similar companies.

### Feature Clustering
For the most advanced milestone, products are grouped by shared feature combinations.

Instead of predicting one exact SKU, Demandra predicts:
> "the buyer likely needs a product with these specifications."

This makes the system more robust to duplicate SKUs and noisy product catalogs.

---

## Architecture

```text
┌───────────────────────────────────────────────┐
│                 Demandra                      │
├───────────────────────────────────────────────┤
│                                               │
│  Raw transaction data                         │
│  + customer metadata                          │
│  + product feature data                       │
│             │                                 │
│             ▼                                 │
│   Data cleaning & preprocessing               │
│             │                                 │
│             ▼                                 │
│   Demand prediction engine                    │
│   ├─ Level 1: E-Class                         │
│   ├─ Level 2: E-Class + Manufacturer          │
│   └─ Level 3: E-Class + Feature Clusters      │
│             │                                 │
│             ▼                                 │
│   Economic portfolio optimization             │
│             │                                 │
│             ▼                                 │
│   Final predictions / recommendations         │
│                                               │
└───────────────────────────────────────────────┘
```

---

## Modules

### `milestone1_eclass/`
Predicts recurring procurement needs at the **E-Class** level.

**Main idea:**
Use recurrence scoring plus an economic threshold to determine which categories are worth predicting.

**Key files:**
- `solution_level1.py`
- `validate_and_optimize.py`
- `data_exploration.py`

### `milestone2_eclass_manufacturer/`
Predicts recurring needs at the **E-Class + Manufacturer** level.

**Main idea:**
Go one level deeper by capturing supplier preference, while being more conservative to avoid fee bloat.

**Key files:**
- `solution_level2.py`
- `data_exploration.py`

### `milestone3_feature_clusters/`
Predicts recurring needs at the **E-Class + Feature Combination** level.

**Main idea:**
Create stable feature-based clusters of SKUs so we predict requirements rather than exact products.

**Key files:**
- `solution_level3.py`
- `data_exploration.py`

### `procurement_recommender_system/`
Recommends **cheaper-but-equivalent** products that a buyer is likely to purchase next.

**Main idea:**
Rank candidates using purchase likelihood and price advantage.

**Key files:**
- `src/demo.py`
- `src/webapp.py`
- `src/candidate_generation.py`
- `src/ranking.py`
- `src/cold_start.py`

---

## Algorithms, Concepts, and Tools

### Algorithms / Concepts
- **RFM-style recurrence scoring**
- **Economic portfolio optimization**
- **Temporal validation**
- **Cold-start similarity matching**
- **Feature-based clustering**
- **Price-aware recommendation ranking**

### Tools / Libraries
- **Python**
- **Pandas**
- **NumPy**
- **scikit-learn**
- **SciPy**
- **Flask**

---

## Milestones in Brief

### Milestone 1
Predict recurring demand at the category level.
Best for robust, broad demand forecasting.

### Milestone 2
Predict recurring demand at the category + manufacturer level.
Adds brand precision, but with higher risk.

### Milestone 3
Predict recurring demand at the category + feature level.
Captures product requirements instead of exact items.

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run Milestone 1

```bash
cd milestone1_eclass
python solution_level1.py
```

### Run Milestone 2

```bash
cd milestone2_eclass_manufacturer
python solution_level2.py
```

### Run Milestone 3

```bash
cd milestone3_feature_clusters
python solution_level3.py
```

### Run the recommender system

```bash
cd procurement_recommender_system
pip install -r requirements.txt
python src/webapp.py
```

Then open:

```text
http://localhost:5000
```

---

## Repository Structure

```text
DataMining-Hack/
├── Challenge2/
├── milestone1_eclass/
├── milestone2_eclass_manufacturer/
├── milestone3_feature_clusters/
├── procurement_recommender_system/
├── Context.md
├── README.md
└── requirements.txt
```

---

## Why Demandra Stands Out

Demandra is not just a forecasting project. It is a practical procurement intelligence system that combines:

- demand prediction
- economic optimization
- cold-start handling
- product abstraction
- cheaper-alternative recommendations

It turns messy real-world procurement data into a system that helps buyers make better, cheaper, and more strategic purchasing decisions.

---

## Achievement

**Demandra** was built for **Challenge 2 of the Unite x TUM.ai Hackathon** and won **1st place**.