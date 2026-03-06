# Milestone 1 — E-Class Level Core Demand Prediction

## Overview
Predicts Core Demand at the **E-Class** (functional product category) level.
Each prediction is a `buyer_id → eclass_id` pair indicating a recurring procurement need.

## Approach

### Warm-Start Buyers (48 buyers with transaction history)
1. **RFM-style scoring** per (buyer, eclass) pair:
   - **Frequency**: Number of distinct orders containing this eclass
   - **Consistency**: How many quarters this eclass was purchased (temporal spread)
   - **Monetary value**: Total spend on this eclass
   - **Recency**: How recently the eclass was last purchased
2. **Economic threshold**: Estimated annual savings vs. annual fee per element
3. Only include eclasses that pass both recurrence AND economic viability thresholds

### Cold-Start Buyers (52 buyers with no history)
1. **Similar buyer matching** based on:
   - NACE code similarity (exact → 3-digit → 2-digit fallback)
   - Company size similarity (log-scale employee count)
2. **Aggregate patterns**: From similar buyers, identify eclasses with high penetration rate
3. **Conservative selection**: Fewer predictions to avoid excessive fees

### Portfolio Discipline
- Maximum ~50 items per warm buyer, ~15 per cold buyer
- Threshold tuning to balance savings vs. fees

## Files
- `solution_level1.py` — Main prediction pipeline
- `data_exploration.py` — Data analysis utilities
- `submission.csv` — Output file (generated after running)

## How to Run
```bash
cd milestone1_eclass
python solution_level1.py
```

## Output Format
```csv
buyer_id,predicted_id
41165867,27050401
41165867,19030312
...
```
