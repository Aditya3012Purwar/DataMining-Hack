# Milestone 2 — E-Class + Manufacturer Level Core Demand Prediction

## Overview
Predicts Core Demand at the **E-Class + Manufacturer** level.
Each prediction is a `buyer_id → eclass_manufacturer_id` pair.

This adds brand/supplier specificity to the functional category, enabling more precise targeting.

## Approach

### Manufacturer Normalization
- Strip common suffixes (GmbH, AG, Inc., Ltd.)
- Lowercase and remove special characters
- Handle missing manufacturers as "UNKNOWN"

### Warm-Start Buyers
1. **RFM scoring** per (buyer, eclass, manufacturer_normalized) triple
2. **Portfolio discipline**: Avoid recommending multiple manufacturers for the same eclass unless both have independently strong recurring signals
3. **Economic threshold**: Higher bar than Level 1 because more granular = more fee risk

### Cold-Start Buyers  
1. **Similar buyer matching** (same as Level 1)
2. **Aggregate (eclass, manufacturer) patterns** from similar buyers
3. **Deduplication**: Keep the dominant manufacturer per eclass, allow secondary only with strong penetration (≥40%)

### Key Differences from Level 1
- Higher recurrence threshold (0.40 vs 0.35) — more confidence needed at finer granularity
- Manufacturer deduplication logic prevents fee bloat
- Normalized manufacturer names reduce catalog noise

## Files
- `solution_level2.py` — Main prediction pipeline
- `data_exploration.py` — Data analysis utilities
- `submission.csv` — Output file (generated after running)

## How to Run
```bash
cd milestone2_eclass_manufacturer
python solution_level2.py
```

## Output Format
```csv
buyer_id,predicted_id
41165867,27050401||siemens
41165867,19030312||3m
...
```
The `predicted_id` format is `eclass||manufacturer_normalized`.
