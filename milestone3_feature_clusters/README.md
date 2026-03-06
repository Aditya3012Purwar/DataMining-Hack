# Milestone 3 — E-Class + Feature Combination (Clustered) Core Demand Prediction

## Overview
Predicts Core Demand at the **E-Class + Feature Cluster** level.
Instead of predicting exact products, this level predicts **product requirements**.

Each prediction is a `buyer_id → cluster_id` pair where clusters are groups of SKUs
sharing the same eclass and similar feature attributes.

## Approach

### Feature Engineering & Clustering
1. **Load feature data** (`features_per_sku.csv`): 18M rows of SKU feature attributes
2. **Map SKUs to eclasses** via the training data
3. **Identify top feature keys** per eclass (most common/discriminative)
4. **Build feature signatures**: For each SKU, concatenate its top-N feature key-value pairs
5. **Hash-based clustering**: SKUs with identical feature signatures within the same eclass form a cluster
6. **Cluster IDs**: Format `{eclass}__{signature_hash}` or `{eclass}__default` for SKUs without features

### Warm-Start Buyers
1. Map historical purchases to feature clusters via SKU → cluster mapping
2. **RFM scoring** per (buyer, cluster) pair
3. **Portfolio discipline**: Prefer specific (non-default) clusters; deduplicate per eclass

### Cold-Start Buyers
1. **Similar buyer matching** (NACE code + company size)  
2. **Aggregate cluster patterns** from similar buyers
3. **Conservative portfolio**: Dominant cluster per eclass only

### Methodological Highlights
- **Feature-based abstraction**: Captures product requirements rather than specific SKUs
- **Stable clusters**: Hash-based signatures are deterministic and reproducible
- **Handles duplicates**: Multiple SKUs with same features map to the same cluster
- **Economic reasoning**: Same savings-vs-fee optimization framework as Levels 1 & 2

## Files
- `solution_level3.py` — Main prediction pipeline (includes clustering)
- `data_exploration.py` — Data analysis utilities
- `submission.csv` — Output file (generated after running)
- `cluster_mapping.csv` — Maps cluster_id → eclass + feature signature + SKU count

## How to Run
```bash
cd milestone3_feature_clusters
python solution_level3.py
```

## Output Format
```csv
buyer_id,predicted_id
41165867,27050401__a3f8b2c1
41165867,19030312__default
...
```
The `predicted_id` is a cluster ID encoding the eclass and feature signature hash.

## Cluster Mapping
The `cluster_mapping.csv` file documents what each cluster represents:
```csv
cluster_id,eclass,feature_signature,sku_count
27050401__a3f8b2c1,27050401,"Spannung-spannung_v=230|Material-material=Kupfer|...",42
```
