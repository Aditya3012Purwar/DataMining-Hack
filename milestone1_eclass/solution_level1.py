"""
Milestone 1 — Core Demand Prediction at E-Class Level
=====================================================
Predict recurring procurement needs at the E-Class (product category) level.
Submission: buyer_id, predicted_id (eclass_id)

Strategy (validated via temporal split: train < 2024-07, validate >= 2024-07):
- Warm-start buyers: Score each (buyer, eclass) pair using monetary-weighted
  recurrence scoring. Include items with 2+ quarters of history and positive
  estimated net benefit. No hard cap — let the economics decide.
  Achieves ~90% hit rate on validation with ~250 items/buyer.
- Cold-start buyers: Find similar buyers by NACE code + company size,
  aggregate their recurring patterns, recommend high-penetration categories.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG (optimized via validate_and_optimize.py sweep)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent / "Challenge2"
OUTPUT_DIR = Path(__file__).resolve().parent

MONTHLY_FEE = 3.0
PREDICTION_HORIZON = 12
ANNUAL_FEE_PER_ELEMENT = MONTHLY_FEE * PREDICTION_HORIZON
SAVINGS_MULTIPLIER = 3.0

# Recurrence score weights (monetary-focused — validated as best)
W_FREQ = 0.20
W_CONSISTENCY = 0.25
W_MONETARY = 0.35
W_RECENCY = 0.20

WARM_THRESHOLD = 0.50
MIN_QUARTERS = 2
COLD_MAX_ITEMS = 500

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
plis = pd.read_csv(
    BASE_DIR / "plis_training.csv",
    sep='\t',
    low_memory=False,
    dtype={'eclass': str, 'nace_code': str, 'secondary_nace_code': str}
)
customers = pd.read_csv(
    BASE_DIR / "customer_test.csv",
    sep='\t',
    dtype={'nace_code': str, 'secondary_nace_code': str}
)
nace = pd.read_csv(
    BASE_DIR / "nace_codes.csv",
    sep='\t',
    dtype={'nace_code': str}
)

plis['orderdate'] = pd.to_datetime(plis['orderdate'])
plis['eclass'] = plis['eclass'].astype(str).str.strip()
# Filter to valid 8-digit eclass codes (removes NaN/invalid → eliminates wasted predictions)
plis = plis[plis['eclass'].str.match(r'^\d{8}$', na=False)].copy()

print(f"Training data: {len(plis):,} rows, {plis['legal_entity_id'].nunique():,} buyers")
print(f"Test buyers: {len(customers)} ({(customers['task']=='cold start').sum()} cold, {(customers['task']=='predict future').sum()} warm)")

# ============================================================
# FEATURE ENGINEERING: Warm-start buyers
# ============================================================

def compute_recurrence_scores(buyer_data):
    """
    For a given buyer's transaction data, compute a recurrence score per eclass.
    Uses monetary-weighted scoring validated via temporal split.
    """
    buyer_data = buyer_data.copy()
    buyer_data['total_value'] = buyer_data['quantityvalue'].fillna(0) * buyer_data['vk_per_item'].fillna(0)
    buyer_data['quarter'] = buyer_data['orderdate'].dt.to_period('Q')
    buyer_data['year'] = buyer_data['orderdate'].dt.year

    max_date = buyer_data['orderdate'].max()
    total_quarters = buyer_data['quarter'].nunique()

    if total_quarters == 0:
        return pd.DataFrame()

    agg = buyer_data.groupby('eclass').agg(
        order_count=('set_id', 'nunique'),
        total_spend=('total_value', 'sum'),
        quarters_active=('quarter', 'nunique'),
        n_years=('year', 'nunique'),
        last_purchase=('orderdate', 'max'),
        avg_price=('vk_per_item', 'mean'),
    ).reset_index()

    agg['recency_days'] = (max_date - agg['last_purchase']).dt.days

    freq_norm = np.log1p(agg['order_count']) / np.log1p(agg['order_count'].max()) if agg['order_count'].max() > 0 else 0
    consistency = agg['quarters_active'] / max(total_quarters, 1)
    monetary_norm = np.log1p(agg['total_spend']) / np.log1p(agg['total_spend'].max()) if agg['total_spend'].max() > 0 else 0
    recency_score = 1 - (agg['recency_days'] / max(agg['recency_days'].max(), 1))

    agg['recurrence_score'] = (
        W_FREQ * freq_norm +
        W_CONSISTENCY * consistency +
        W_MONETARY * monetary_norm +
        W_RECENCY * recency_score
    )

    annual_freq_estimate = agg['order_count'] * (12 / max(total_quarters * 3, 1))
    agg['estimated_annual_savings'] = np.sqrt(agg['avg_price'].clip(lower=0.01)) * annual_freq_estimate * SAVINGS_MULTIPLIER
    agg['net_benefit'] = agg['estimated_annual_savings'] - ANNUAL_FEE_PER_ELEMENT

    return agg


def get_warm_predictions(buyer_id, buyer_data):
    """Get eclass predictions for a warm-start buyer."""
    scores = compute_recurrence_scores(buyer_data)
    if scores.empty:
        return []

    candidates = scores[
        (scores['recurrence_score'] >= WARM_THRESHOLD) &
        (scores['net_benefit'] > 0) &
        (scores['quarters_active'] >= MIN_QUARTERS)
    ].copy()

    candidates = candidates.sort_values('estimated_annual_savings', ascending=False)

    return candidates['eclass'].tolist()


# ============================================================
# FEATURE ENGINEERING: Cold-start buyers
# ============================================================

def get_nace_prefix(nace_code, level=2):
    """Get NACE code prefix at a given level (2-digit, 3-digit, etc.)"""
    if pd.isna(nace_code):
        return None
    nace_str = str(int(float(nace_code))) if not isinstance(nace_code, str) else nace_code
    return nace_str[:level] if len(nace_str) >= level else nace_str


def find_similar_buyers(target_customer, all_training_data, top_n=50):
    """
    Find similar buyers from training data based on NACE code and company size.
    """
    nace = str(target_customer['nace_code'])
    sec_nace = target_customer.get('secondary_nace_code', None)
    emp_count = target_customer['estimated_number_employees']
    
    # Get unique buyers in training with their attributes
    buyer_attrs = all_training_data.groupby('legal_entity_id').agg(
        nace_code=('nace_code', 'first'),
        emp=('estimated_number_employees', 'first'),
        n_orders=('set_id', 'nunique'),
    ).reset_index()
    
    buyer_attrs['nace_code'] = buyer_attrs['nace_code'].astype(str)
    
    # Score similarity
    # 1) Exact NACE match
    buyer_attrs['nace_match'] = 0
    buyer_attrs.loc[buyer_attrs['nace_code'] == nace, 'nace_match'] = 4
    
    # 2) 3-digit NACE match
    nace_3 = nace[:3] if len(nace) >= 3 else nace
    mask_3 = buyer_attrs['nace_code'].str[:3] == nace_3
    buyer_attrs.loc[mask_3 & (buyer_attrs['nace_match'] == 0), 'nace_match'] = 3
    
    # 3) 2-digit NACE match
    nace_2 = nace[:2] if len(nace) >= 2 else nace
    mask_2 = buyer_attrs['nace_code'].str[:2] == nace_2
    buyer_attrs.loc[mask_2 & (buyer_attrs['nace_match'] == 0), 'nace_match'] = 2
    
    # 4) Company size similarity (log scale)
    if pd.notna(emp_count) and emp_count > 0:
        buyer_attrs['size_sim'] = 1 / (1 + np.abs(np.log1p(buyer_attrs['emp']) - np.log1p(emp_count)))
    else:
        buyer_attrs['size_sim'] = 0.5
    
    # Combined similarity
    buyer_attrs['similarity'] = buyer_attrs['nace_match'] * 2 + buyer_attrs['size_sim']
    
    # Filter to at least 2-digit NACE match
    similar = buyer_attrs[buyer_attrs['nace_match'] >= 2].nlargest(top_n, 'similarity')
    
    # If not enough, fall back to broader match
    if len(similar) < 5:
        nace_1 = nace[:1] if len(nace) >= 1 else nace
        mask_1 = buyer_attrs['nace_code'].str[:1] == nace_1
        buyer_attrs.loc[mask_1 & (buyer_attrs['nace_match'] == 0), 'nace_match'] = 1
        buyer_attrs['similarity'] = buyer_attrs['nace_match'] * 2 + buyer_attrs['size_sim']
        similar = buyer_attrs[buyer_attrs['nace_match'] >= 1].nlargest(top_n, 'similarity')
    
    # Last resort: use all buyers
    if len(similar) < 5:
        similar = buyer_attrs.nlargest(top_n, 'n_orders')
    
    return similar['legal_entity_id'].tolist()


def get_cold_predictions(target_customer, all_training_data):
    """Get eclass predictions for a cold-start buyer."""
    similar_buyer_ids = find_similar_buyers(target_customer, all_training_data)
    
    if not similar_buyer_ids:
        return []
    
    # Get transactions of similar buyers
    sim_data = all_training_data[all_training_data['legal_entity_id'].isin(similar_buyer_ids)].copy()
    sim_data['total_value'] = sim_data['quantityvalue'].fillna(0) * sim_data['vk_per_item'].fillna(0)
    sim_data['quarter'] = sim_data['orderdate'].dt.to_period('Q')
    
    # Aggregate across similar buyers
    eclass_stats = sim_data.groupby('eclass').agg(
        n_buyers=('legal_entity_id', 'nunique'),
        total_orders=('set_id', 'nunique'),
        total_spend=('total_value', 'sum'),
        avg_price=('vk_per_item', 'mean'),
        quarters_active=('quarter', 'nunique'),
    ).reset_index()
    
    n_similar = len(similar_buyer_ids)
    
    # Buyer penetration: fraction of similar buyers who buy this eclass  
    eclass_stats['penetration'] = eclass_stats['n_buyers'] / max(n_similar, 1)
    
    # Average frequency per buyer
    eclass_stats['avg_freq'] = eclass_stats['total_orders'] / max(n_similar, 1)
    
    eclass_stats['cold_score'] = (
        eclass_stats['penetration'] * 0.5 +
        np.log1p(eclass_stats['avg_freq']) / np.log1p(eclass_stats['avg_freq'].max()) * 0.3 +
        np.log1p(eclass_stats['total_spend']) / np.log1p(eclass_stats['total_spend'].max()) * 0.2
    )

    candidates = eclass_stats[eclass_stats['penetration'] >= 0.05]
    candidates = candidates.sort_values('cold_score', ascending=False)
    
    # Be conservative for cold start
    candidates = candidates.head(COLD_MAX_ITEMS)
    
    return candidates['eclass'].tolist()


# ============================================================
# MAIN PREDICTION PIPELINE
# ============================================================
print("\nPrecomputing buyer data index...")
# Pre-index training data by buyer for efficiency
buyer_groups = {}
warm_buyer_ids = set(customers[customers['task'] == 'predict future']['legal_entity_id'])

# Only load warm buyer groups
for buyer_id in warm_buyer_ids:
    mask = plis['legal_entity_id'] == buyer_id
    if mask.any():
        buyer_groups[buyer_id] = plis[mask]

print(f"Indexed {len(buyer_groups)} warm buyers")

# Also precompute a summary for cold-start similarity search
# (random sample to make it tractable)
print("Preparing cold-start buyer profiles...")
cold_start_sample = plis.copy()  # use all data for better accuracy

results = []

print("\nGenerating predictions...")
for idx, row in customers.iterrows():
    buyer_id = row['legal_entity_id']
    task = row['task']
    
    if task == 'predict future' and buyer_id in buyer_groups:
        # Warm start
        predictions = get_warm_predictions(buyer_id, buyer_groups[buyer_id])
        source = 'warm'
    else:
        # Cold start
        predictions = get_cold_predictions(row, cold_start_sample)
        source = 'cold'
    
    for eclass in predictions:
        results.append({
            'legal_entity_id': buyer_id,
            'eclass': eclass
        })
    
    print(f"  Buyer {buyer_id} ({source}): {len(predictions)} predictions")

# ============================================================
# OUTPUT
# ============================================================
submission = pd.DataFrame(results)
# Ensure eclass is clean string (no .0 float suffix)
submission['eclass'] = submission['eclass'].astype(str).str.strip()
# Remove any remaining invalid/NaN predictions
submission = submission[submission['eclass'].str.match(r'^\d{8}$', na=False)].copy()
output_path = OUTPUT_DIR / "submission.csv"
submission.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(f"Level 1 Submission saved to: {output_path}")
print(f"Total predictions: {len(submission)}")
print(f"Unique buyers with predictions: {submission['legal_entity_id'].nunique()}")
print(f"Avg predictions per buyer: {len(submission)/max(submission['legal_entity_id'].nunique(),1):.1f}")
print(f"{'='*60}")

# Summary statistics
warm_sub = submission[submission['legal_entity_id'].isin(warm_buyer_ids)]
cold_sub = submission[~submission['legal_entity_id'].isin(warm_buyer_ids)]
print(f"\nWarm-start: {warm_sub['legal_entity_id'].nunique()} buyers, {len(warm_sub)} predictions ({len(warm_sub)/max(warm_sub['legal_entity_id'].nunique(),1):.1f} avg)")
print(f"Cold-start: {cold_sub['legal_entity_id'].nunique()} buyers, {len(cold_sub)} predictions ({len(cold_sub)/max(cold_sub['legal_entity_id'].nunique(),1):.1f} avg)")
