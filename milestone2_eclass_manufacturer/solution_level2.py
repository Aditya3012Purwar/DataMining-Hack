"""
Milestone 2 — Core Demand Prediction at E-Class + Manufacturer Level
====================================================================
Predict recurring procurement needs at E-Class + Manufacturer level.
Submission: buyer_id, predicted_id (eclass_manufacturer_id)

Strategy:
- Higher precision by adding manufacturer/brand specificity.
- Warm-start: Identify recurring (eclass, manufacturer) pairs from history.
  De-duplicate manufacturer names (basic normalization).
  Apply economic threshold: more selective because fees are per pair.
- Cold-start: Similar buyer matching + aggregate (eclass, manufacturer) patterns.
  Be extra conservative — more granular = more noise.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent / "Challenge2"
OUTPUT_DIR = Path(__file__).resolve().parent

FEE_PER_ITEM = 10.0      # €10 fixed fee per predicted item (from scoring page)
SCORING_MONTHS = 1        # 1 month scoring window
ANNUAL_FEE_PER_ELEMENT = FEE_PER_ITEM  # fee is per item, not per month

WARM_THRESHOLD = 0.50  # high threshold — need strong evidence at this granularity
COLD_MAX_ITEMS = 5     # very conservative for cold-start at L2 granularity

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
plis = pd.read_csv(
    BASE_DIR / "plis_training.csv" / "plis_training.csv",
    sep='\t',
    low_memory=False,
    dtype={'eclass': str, 'nace_code': str, 'secondary_nace_code': str}
)
customers = pd.read_csv(
    BASE_DIR / "customer_test.csv" / "customer_test.csv",
    sep='\t',
    dtype={'nace_code': str, 'secondary_nace_code': str}
)

plis['orderdate'] = pd.to_datetime(plis['orderdate'])
plis['eclass'] = plis['eclass'].astype(str).str.strip()

print(f"Training data: {len(plis):,} rows, {plis['legal_entity_id'].nunique():,} buyers")
print(f"Test buyers: {len(customers)}")

# ============================================================
# MANUFACTURER NORMALIZATION
# ============================================================
def normalize_manufacturer(name):
    """Normalize manufacturer for GROUPING (lowercase comparison)."""
    if pd.isna(name) or str(name).strip() == '':
        return 'UNKNOWN'
    name = str(name).strip()
    name_low = name.lower()
    for suffix in [' gmbh', ' ag', ' co.', ' inc.', ' ltd.', ' corp.', ' se', ' kg']:
        name_low = name_low.replace(suffix, '')
    name_low = re.sub(r'[^a-z0-9\s]', '', name_low)
    name_low = re.sub(r'\s+', ' ', name_low).strip()
    return name_low if name_low else 'UNKNOWN'

print("Normalizing manufacturer names...")
plis['manufacturer_norm'] = plis['manufacturer'].apply(normalize_manufacturer)
# Keep original manufacturer name for output (scorer matches on original names)
plis['manufacturer_orig'] = plis['manufacturer'].fillna('UNKNOWN').str.strip()

# Create combined key for GROUPING (normalized)
plis['eclass_mfr'] = plis['eclass'] + '|' + plis['manufacturer_norm']
# Create output key with ORIGINAL manufacturer name (for scorer matching)
plis['eclass_mfr_orig'] = plis['eclass'] + '|' + plis['manufacturer_orig']

print(f"Unique eclass_mfr combinations: {plis['eclass_mfr'].nunique():,}")

# ============================================================
# FEATURE ENGINEERING: Warm-start buyers
# ============================================================

def compute_recurrence_scores_l2(buyer_data):
    """
    Compute recurrence scores per (eclass, manufacturer) pair.
    """
    buyer_data = buyer_data.copy()
    buyer_data['total_value'] = buyer_data['quantityvalue'].fillna(0) * buyer_data['vk_per_item'].fillna(0)
    buyer_data['quarter'] = buyer_data['orderdate'].dt.to_period('Q')
    
    total_quarters = buyer_data['quarter'].nunique()
    if total_quarters == 0:
        return pd.DataFrame()
    
    agg = buyer_data.groupby('eclass_mfr').agg(
        eclass=('eclass', 'first'),
        manufacturer_norm=('manufacturer_norm', 'first'),
        eclass_mfr_orig=('eclass_mfr_orig', 'first'),  # keep original name for output
        order_count=('set_id', 'nunique'),
        line_count=('sku', 'count'),
        total_spend=('total_value', 'sum'),
        total_qty=('quantityvalue', 'sum'),
        quarters_active=('quarter', 'nunique'),
        last_purchase=('orderdate', 'max'),
        avg_price=('vk_per_item', 'mean'),
    ).reset_index()
    
    max_date = buyer_data['orderdate'].max()
    
    # Normalize
    agg['frequency_score'] = np.log1p(agg['order_count']) / np.log1p(max(agg['order_count'].max(), 1))
    agg['consistency_score'] = agg['quarters_active'] / max(total_quarters, 1)
    agg['monetary_score'] = np.log1p(agg['total_spend']) / np.log1p(max(agg['total_spend'].max(), 1))
    agg['recency_days'] = (max_date - agg['last_purchase']).dt.days
    agg['recency_score'] = 1 - (agg['recency_days'] / max(agg['recency_days'].max(), 1))
    
    # Combined recurrence score
    agg['recurrence_score'] = (
        0.30 * agg['frequency_score'] +
        0.35 * agg['consistency_score'] +
        0.20 * agg['monetary_score'] +
        0.15 * agg['recency_score']
    )
    
    # Economic benefit estimation
    annual_freq_estimate = agg['order_count'] * (12 / max(total_quarters * 3, 1))
    agg['estimated_annual_savings'] = np.sqrt(agg['avg_price'].clip(lower=0.01)) * annual_freq_estimate * 0.5
    agg['net_benefit'] = agg['estimated_annual_savings'] - ANNUAL_FEE_PER_ELEMENT
    
    return agg


def get_warm_predictions_l2(buyer_id, buyer_data):
    """Get (eclass, manufacturer) predictions for a warm-start buyer."""
    scores = compute_recurrence_scores_l2(buyer_data)
    if scores.empty:
        return []
    
    # STRICT: Require BOTH good recurrence AND positive net benefit
    # At L2 granularity, every wrong prediction costs a fee with no savings
    candidates = scores[
        (scores['recurrence_score'] >= WARM_THRESHOLD) &
        (scores['net_benefit'] > 0) &
        (scores['order_count'] >= 2) &            # at least 2 distinct orders
        (scores['quarters_active'] >= 2)           # appeared in at least 2 quarters
    ].copy()
    
    # NO fallback — if nothing qualifies, predict nothing rather than guess
    if candidates.empty:
        return []
    
    # Portfolio discipline: STRICT — only 1 manufacturer per eclass
    # Keep the strongest manufacturer per eclass
    deduped = []
    for eclass, group in candidates.groupby('eclass'):
        group_sorted = group.sort_values('recurrence_score', ascending=False)
        # Keep only the top manufacturer per eclass
        deduped.append(group_sorted.iloc[0])
    
    if not deduped:
        return []
    
    result = pd.DataFrame(deduped)
    # Cap at 25 — be disciplined
    result = result.sort_values('recurrence_score', ascending=False).head(25)
    
    # Return ORIGINAL manufacturer name format for scorer
    return result['eclass_mfr_orig'].tolist()


# ============================================================
# FEATURE ENGINEERING: Cold-start buyers
# ============================================================

def find_similar_buyers(target_customer, all_training_data, top_n=50):
    """Find similar buyers from training data based on NACE code and company size."""
    nace = str(target_customer['nace_code'])
    emp_count = target_customer['estimated_number_employees']
    
    buyer_attrs = all_training_data.groupby('legal_entity_id').agg(
        nace_code=('nace_code', 'first'),
        emp=('estimated_number_employees', 'first'),
        n_orders=('set_id', 'nunique'),
    ).reset_index()
    
    buyer_attrs['nace_code'] = buyer_attrs['nace_code'].astype(str)
    
    # NACE matching
    buyer_attrs['nace_match'] = 0
    buyer_attrs.loc[buyer_attrs['nace_code'] == nace, 'nace_match'] = 4
    
    nace_3 = nace[:3] if len(nace) >= 3 else nace
    mask_3 = buyer_attrs['nace_code'].str[:3] == nace_3
    buyer_attrs.loc[mask_3 & (buyer_attrs['nace_match'] == 0), 'nace_match'] = 3
    
    nace_2 = nace[:2] if len(nace) >= 2 else nace
    mask_2 = buyer_attrs['nace_code'].str[:2] == nace_2
    buyer_attrs.loc[mask_2 & (buyer_attrs['nace_match'] == 0), 'nace_match'] = 2
    
    if pd.notna(emp_count) and emp_count > 0:
        buyer_attrs['size_sim'] = 1 / (1 + np.abs(np.log1p(buyer_attrs['emp']) - np.log1p(emp_count)))
    else:
        buyer_attrs['size_sim'] = 0.5
    
    buyer_attrs['similarity'] = buyer_attrs['nace_match'] * 2 + buyer_attrs['size_sim']
    
    similar = buyer_attrs[buyer_attrs['nace_match'] >= 2].nlargest(top_n, 'similarity')
    
    if len(similar) < 5:
        nace_1 = nace[:1]
        mask_1 = buyer_attrs['nace_code'].str[:1] == nace_1
        buyer_attrs.loc[mask_1 & (buyer_attrs['nace_match'] == 0), 'nace_match'] = 1
        buyer_attrs['similarity'] = buyer_attrs['nace_match'] * 2 + buyer_attrs['size_sim']
        similar = buyer_attrs[buyer_attrs['nace_match'] >= 1].nlargest(top_n, 'similarity')
    
    if len(similar) < 5:
        similar = buyer_attrs.nlargest(top_n, 'n_orders')
    
    return similar['legal_entity_id'].tolist()


def get_cold_predictions_l2(target_customer, all_training_data):
    """Get (eclass, manufacturer) predictions for a cold-start buyer."""
    similar_buyer_ids = find_similar_buyers(target_customer, all_training_data)
    
    if not similar_buyer_ids:
        return []
    
    sim_data = all_training_data[all_training_data['legal_entity_id'].isin(similar_buyer_ids)].copy()
    sim_data['total_value'] = sim_data['quantityvalue'].fillna(0) * sim_data['vk_per_item'].fillna(0)
    sim_data['quarter'] = sim_data['orderdate'].dt.to_period('Q')
    
    eclass_mfr_stats = sim_data.groupby('eclass_mfr').agg(
        eclass=('eclass', 'first'),
        manufacturer_norm=('manufacturer_norm', 'first'),
        eclass_mfr_orig=('eclass_mfr_orig', 'first'),  # keep original name
        n_buyers=('legal_entity_id', 'nunique'),
        total_orders=('set_id', 'nunique'),
        total_spend=('total_value', 'sum'),
        avg_price=('vk_per_item', 'mean'),
    ).reset_index()
    
    n_similar = len(similar_buyer_ids)
    eclass_mfr_stats['penetration'] = eclass_mfr_stats['n_buyers'] / max(n_similar, 1)
    eclass_mfr_stats['avg_freq'] = eclass_mfr_stats['total_orders'] / max(n_similar, 1)
    
    eclass_mfr_stats['cold_score'] = (
        eclass_mfr_stats['penetration'] * 0.5 +
        np.log1p(eclass_mfr_stats['avg_freq']) / np.log1p(max(eclass_mfr_stats['avg_freq'].max(), 1)) * 0.3 +
        np.log1p(eclass_mfr_stats['total_spend']) / np.log1p(max(eclass_mfr_stats['total_spend'].max(), 1)) * 0.2
    )
    
    # VERY strict penetration for L2 cold-start — need high confidence
    candidates = eclass_mfr_stats[eclass_mfr_stats['penetration'] >= 0.40]
    candidates = candidates.sort_values('cold_score', ascending=False)
    
    # Portfolio discipline: strictly 1 manufacturer per eclass
    deduped = []
    seen_eclass = set()
    for _, row in candidates.iterrows():
        ec = row['eclass']
        if ec not in seen_eclass:
            deduped.append(row)
            seen_eclass.add(ec)
    
    result = pd.DataFrame(deduped).head(COLD_MAX_ITEMS)
    
    return result['eclass_mfr_orig'].tolist() if not result.empty else []


# ============================================================
# MAIN PREDICTION PIPELINE
# ============================================================
print("\nPrecomputing buyer data index...")
warm_buyer_ids = set(customers[customers['task'] == 'predict future']['legal_entity_id'])

buyer_groups = {}
for buyer_id in warm_buyer_ids:
    mask = plis['legal_entity_id'] == buyer_id
    if mask.any():
        buyer_groups[buyer_id] = plis[mask]

print(f"Indexed {len(buyer_groups)} warm buyers")

results = []

print("\nGenerating Level 2 predictions...")
for idx, row in customers.iterrows():
    buyer_id = row['legal_entity_id']
    task = row['task']
    
    if task == 'predict future' and buyer_id in buyer_groups:
        predictions = get_warm_predictions_l2(buyer_id, buyer_groups[buyer_id])
        source = 'warm'
    else:
        predictions = get_cold_predictions_l2(row, plis)
        source = 'cold'
    
    for eclass_mfr in predictions:
        results.append({
            'legal_entity_id': buyer_id,
            'cluster': eclass_mfr
        })
    
    print(f"  Buyer {buyer_id} ({source}): {len(predictions)} predictions")

# ============================================================
# OUTPUT
# ============================================================
submission = pd.DataFrame(results)
output_path = OUTPUT_DIR / "submission.csv"
submission.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(f"Level 2 Submission saved to: {output_path}")
print(f"Total predictions: {len(submission)}")
print(f"Unique buyers with predictions: {submission['legal_entity_id'].nunique()}")
print(f"Avg predictions per buyer: {len(submission)/max(submission['legal_entity_id'].nunique(),1):.1f}")
print(f"{'='*60}")

warm_sub = submission[submission['legal_entity_id'].isin(warm_buyer_ids)]
cold_sub = submission[~submission['legal_entity_id'].isin(warm_buyer_ids)]
print(f"\nWarm-start: {warm_sub['legal_entity_id'].nunique()} buyers, {len(warm_sub)} predictions ({len(warm_sub)/max(warm_sub['legal_entity_id'].nunique(),1):.1f} avg)")
print(f"Cold-start: {cold_sub['legal_entity_id'].nunique()} buyers, {len(cold_sub)} predictions ({len(cold_sub)/max(cold_sub['legal_entity_id'].nunique(),1):.1f} avg)")
