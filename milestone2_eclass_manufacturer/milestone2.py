"""
Milestone 2 — Core Demand Prediction: E-Class + Manufacturer
=============================================================
Format:   legal_entity_id,cluster   (cluster = "eclass|Manufacturer")
Scoring:  Savings = 10% of matched spend  |  Fee = €10/item  |  1 month

Strategy:
  Warm-start: Select items that are (a) high-spend AND (b) recurring/recent.
              Skip one-off purchases and low-spend noise.
  Cold-start: Skip entirely — at L2 granularity the expected value per
              prediction is always negative for cold-start buyers.
  Missing warm buyer (61933687): Has no training data despite being labeled
              "predict future" → treated as cold-start (= skip).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
FEE_PER_ITEM   = 10.0
SAVINGS_RATE   = 0.10

OUTPUT_DIR = Path("/content/drive/MyDrive/Datamining")

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")

plis      = pd.read_csv("/content/drive/MyDrive/Datamining/plis_training.csv",  sep=None, engine='python')
customers = pd.read_csv("/content/drive/MyDrive/Datamining/customer_test.csv",  sep=None, engine='python')

# Normalize column names: strip whitespace + lowercase
plis.columns      = plis.columns.str.strip().str.lower()
customers.columns = customers.columns.str.strip().str.lower()

plis.rename(columns={plis.columns[0]: "orderdate"}, inplace=True)
customers.rename(columns={customers.columns[0]: "legal_entity_id"}, inplace=True)

print(f"plis columns: {list(plis.columns)}")
print(f"customers columns: {list(customers.columns)}")
print(f"plis shape: {plis.shape}")
print(f"customers shape: {customers.shape}")

# Parse dates
plis['orderdate'] = pd.to_datetime(plis['orderdate'])

# Clean eclass: keep only valid 8-digit codes
plis['eclass'] = plis['eclass'].astype(str).str.strip()
plis = plis[plis['eclass'].str.match(r'^\d{8}$', na=False)].copy()

# Original manufacturer names (scorer does EXACT matching)
plis['manufacturer_orig'] = plis['manufacturer'].fillna('UNKNOWN').astype(str).str.strip()

# Cluster key in scorer format — CRITICAL: eclass must have .0 float suffix
# The scorer builds ground truth with float eclass (e.g., "27050401.0|Lenovo")
plis['cluster'] = plis['eclass'] + '.0|' + plis['manufacturer_orig']

# Spend per line item
plis['spend'] = plis['quantityvalue'].fillna(0) * plis['vk_per_item'].fillna(0)

# Time features
plis['year_month'] = plis['orderdate'].dt.to_period('M')

DATA_CUTOFF = pd.Timestamp('2025-07-01')

print(f"\nTraining: {len(plis):,} rows, {plis['legal_entity_id'].nunique():,} buyers")
print(f"Clusters: {plis['cluster'].nunique():,}")
print(f"Dates: {plis['orderdate'].min().date()} to {plis['orderdate'].max().date()}")

# ============================================================
# WARM-START PREDICTIONS
# ============================================================

def predict_warm(buyer_id, buyer_data):
    """
    Select (eclass|manufacturer) pairs likely to recur in July 2025.

    Economics: 10% * monthly_spend must exceed €10 fee.
    So breakeven monthly_spend = €100.

    We use TWO lenses:
      A) Overall history: clusters with avg monthly spend >= threshold
         AND bought in multiple months (not one-off)
      B) Recent activity (last 6 months): clusters actively purchased
         recently with decent spend

    Final portfolio = union, sorted by expected value.
    """
    bd = buyer_data.copy()

    total_months = bd['year_month'].nunique()
    if total_months == 0:
        return []

    # --- Overall aggregation ---
    agg = bd.groupby('cluster').agg(
        total_spend=('spend',      'sum'),
        n_orders   =('set_id',     'nunique'),
        n_months   =('year_month', 'nunique'),
        last_purchase =('orderdate', 'max'),
        first_purchase=('orderdate', 'min'),
    ).reset_index()

    agg['monthly_spend'] = agg['total_spend'] / total_months
    agg['monthly_rate']  = agg['n_months']    / total_months

    # Recency: months since last purchase
    agg['months_since_last'] = (
        (DATA_CUTOFF - agg['last_purchase']).dt.days / 30.0
    ).clip(lower=0)

    # --- Recent aggregation (last 6 months: Jan-Jun 2025) ---
    recent_cutoff = pd.Timestamp('2025-01-01')
    recent = bd[bd['orderdate'] >= recent_cutoff]
    if len(recent) > 0:
        recent_agg = recent.groupby('cluster').agg(
            recent_spend  =('spend',      'sum'),
            recent_months =('year_month', 'nunique'),
            recent_orders =('set_id',     'nunique'),
        ).reset_index()
        agg = agg.merge(recent_agg, on='cluster', how='left')
    else:
        agg['recent_spend']  = 0.0
        agg['recent_months'] = 0
        agg['recent_orders'] = 0

    agg['recent_spend']        = agg['recent_spend'].fillna(0)
    agg['recent_months']       = agg['recent_months'].fillna(0).astype(int)
    agg['recent_orders']       = agg['recent_orders'].fillna(0).astype(int)
    agg['recent_monthly_spend'] = agg['recent_spend'] / 6.0

    # --- RECENCY DECAY (per-month granularity) ---
    # Each key = integer month bucket (1 = most recent, 12 = 12 months ago)
    # Tune any individual month without touching the rest of the logic.
    recency_decay_map = {
        1:  10.00,   # bought last month        → very likely to repeat
        2:  9.50,
        3:  9.00,
        4:  8.50,
        5:  8.00,
        6:  7.50,
        7:  7.00,
        8:  6.50,
        9:  6.00,
        10: 5.50,
        11: 5.00,
        12: 4.50,   # bought 12 months ago     → unlikely to repeat
    }
    DEFAULT_DECAY = 4.0  # older than 12 months → very unlikely

    # Convert continuous months_since_last to integer bucket, floor-clipped to 1
    months_bucket  = agg['months_since_last'].clip(lower=1).round().astype(int)
    recency_factor = months_bucket.map(recency_decay_map).fillna(DEFAULT_DECAY).values

    # --- EXPECTED MONTHLY SAVINGS ---
    # Use the HIGHER of overall vs recent monthly spend (recent is more predictive)
    agg['best_monthly_spend'] = np.maximum(
        agg['monthly_spend'], agg['recent_monthly_spend']
    )

    # Purchase probability: combines recurrence rate and recency
    agg['purchase_prob']   = (agg['monthly_rate'] * recency_factor).clip(upper=1.0)

    # Expected savings per prediction
    agg['expected_savings'] = SAVINGS_RATE * agg['best_monthly_spend'] * agg['purchase_prob']
    agg['expected_net']     = agg['expected_savings'] - FEE_PER_ITEM

    # --- SELECTION ---
    # Tier 1: High expected value items (expected_net > 0)
    tier1 = agg[agg['expected_net'] > 0].copy()

    # Tier 2: Recently active items with decent spend
    # (might have low overall rate but high recent activity → likely to continue)
    tier2 = agg[
        (agg['recent_months']       >= 1) &
        (agg['recent_monthly_spend'] >= 30) &
        (~agg['cluster'].isin(tier1['cluster']))
    ].copy()

    # Tier 3: Consistently recurring with meaningful spend
    tier3 = agg[
        (agg['monthly_rate']      >= 0.2) &
        (agg['monthly_spend']     >= 30)  &
        (agg['months_since_last'] <= 1)   &
        (~agg['cluster'].isin(set(tier1['cluster']) | set(tier2['cluster'])))
    ].copy()

    # Combine and sort by expected net value
    candidates = pd.concat([tier1, tier2, tier3])
    candidates = candidates.sort_values('expected_net', ascending=False)

    # Cap per buyer: limit risk exposure (€10/item × 150 = €1,500 max fee)
    MAX_PER_BUYER = 200
    candidates = candidates.head(MAX_PER_BUYER)

    return candidates['cluster'].tolist()


# ============================================================
# MAIN PIPELINE
# ============================================================
warm_buyers = set(
    customers[customers['task'] == 'predict future']['legal_entity_id']
)

print(f"\nWarm test buyers: {len(warm_buyers)}")

# Index warm buyer data
buyer_groups = {}
for bid in warm_buyers:
    mask = plis['legal_entity_id'] == bid
    if mask.any():
        buyer_groups[bid] = plis[mask]

found_warm   = len(buyer_groups)
missing_warm = warm_buyers - set(buyer_groups.keys())
print(f"Found in training: {found_warm}")
if missing_warm:
    print(f"Missing warm buyers (no training data): {missing_warm}")

# Generate predictions
results = []
print("\n--- Generating Level 2 predictions ---")

for buyer_id in sorted(warm_buyers):
    if buyer_id in buyer_groups:
        preds = predict_warm(buyer_id, buyer_groups[buyer_id])
        for cluster in preds:
            results.append({'legal_entity_id': buyer_id, 'cluster': cluster})
        print(f"  Buyer {buyer_id}: {len(preds)} predictions")
    else:
        print(f"  Buyer {buyer_id}: SKIPPED (no training data)")

# Cold-start buyers: skip (negative expected value at L2 level)
cold_buyers = set(
    customers[customers['task'] == 'cold start']['legal_entity_id']
)
print(f"\nCold-start buyers: {len(cold_buyers)} (all skipped — negative EV at L2)")

# ============================================================
# OUTPUT
# ============================================================
submission = pd.DataFrame(results)

if len(submission) == 0:
    print("\nWARNING: No predictions generated!")
else:
    output_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)

    n_preds  = len(submission)
    n_buyers = submission['legal_entity_id'].nunique()
    avg_per  = n_preds / max(n_buyers, 1)
    fee_exp  = n_preds * FEE_PER_ITEM

    print(f"\n{'='*60}")
    print(f"Level 2 Submission: {output_path}")
    print(f"  Total predictions:       {n_preds:,}")
    print(f"  Buyers with predictions: {n_buyers}")
    print(f"  Avg per buyer:           {avg_per:.1f}")
    print(f"  Fee exposure:            EUR {fee_exp:,.0f}")
    print(f"{'='*60}")

    # Per-buyer breakdown
    per_buyer = submission.groupby('legal_entity_id').size()
    print(f"\nPer-buyer stats:")
    print(f"  Min: {per_buyer.min()},  Max: {per_buyer.max()},  Median: {per_buyer.median():.0f}")

    # Sample output
    print(f"\nSample predictions:")
    print(submission.head(10).to_string(index=False))

    # Verify format
    print(f"\nFormat check:")
    print(f"  Columns: {list(submission.columns)}")
    has_pipe   = submission['cluster'].str.contains(r'\|',  regex=True).all()
    has_double = submission['cluster'].str.contains(r'\|\|', regex=True).any()
    print(f"  All have pipe separator: {has_pipe}")
    print(f"  Any double pipe:         {has_double}")
    no_nan = not submission.isna().any().any()
    print(f"  No NaN values:           {no_nan}")
