"""
Milestone 3 — Core Demand Prediction at E-Class + Feature Combination Level
============================================================================
Predict recurring procurement needs at the E-Class + clustered features level.
Submission: buyer_id, predicted_id (cluster_id)

Strategy:
- Load features_per_sku.csv to get product feature attributes per SKU.
- For each eclass, cluster SKUs by their feature vectors into stable groups.
- Map historical purchases to feature clusters.
- Warm-start: Identify recurring (eclass, feature_cluster) pairs from history.
- Cold-start: Similar-buyer matching + aggregate patterns on feature clusters.
- Portfolio optimization with economic thresholds.

This level focuses on methodological quality: feature engineering, clustering,
and economic reasoning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re
import hashlib
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent / "Challenge2"
OUTPUT_DIR = Path(__file__).resolve().parent

MONTHLY_FEE = 5.0
PREDICTION_HORIZON = 12
ANNUAL_FEE_PER_ELEMENT = MONTHLY_FEE * PREDICTION_HORIZON

WARM_THRESHOLD = 0.40
COLD_MAX_ITEMS = 12
MAX_FEATURES_PER_CLUSTER = 5  # top N features to define a cluster

# ============================================================
# LOAD DATA
# ============================================================
print("Loading training data...")
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
# LOAD & PROCESS FEATURES
# ============================================================
print("\nLoading features data (this may take a moment)...")
features = pd.read_csv(
    BASE_DIR / "features_per_sku.csv" / "features_per_sku.csv",
    sep='\t',
    low_memory=False
)
print(f"Features data: {len(features):,} rows, {features['sku'].nunique():,} unique SKUs")

# Get only the SKUs that appear in our training data
relevant_skus = set(plis['sku'].unique())
features_relevant = features[features['sku'].isin(relevant_skus)].copy()
print(f"Relevant features (in training SKUs): {len(features_relevant):,} rows, {features_relevant['sku'].nunique():,} SKUs")

del features  # free memory

# ============================================================
# FEATURE CLUSTERING
# ============================================================
print("\nBuilding feature-based product clusters...")

def build_feature_signature(sku_features):
    """
    Build a feature signature for a SKU from its feature rows.
    Select the top features (by frequency across SKUs) and create a hashable key.
    """
    # Sort features by key for deterministic signatures
    sig_parts = []
    for _, row in sku_features.iterrows():
        key = str(row['key']).strip()
        # Use fvalue_set if available, otherwise fvalue
        val = str(row['fvalue_set']).strip() if pd.notna(row['fvalue_set']) and str(row['fvalue_set']).strip() else str(row['fvalue']).strip()
        sig_parts.append(f"{key}={val}")
    
    sig_parts.sort()
    return '|'.join(sig_parts[:MAX_FEATURES_PER_CLUSTER])


# Step 1: Identify the most common/useful feature keys per eclass
# Map SKUs to eclasses
sku_to_eclass = plis[['sku', 'eclass']].drop_duplicates().set_index('sku')['eclass'].to_dict()

# Add eclass to features
features_relevant['eclass'] = features_relevant['sku'].map(sku_to_eclass)
features_relevant = features_relevant.dropna(subset=['eclass'])

print(f"Features with eclass mapping: {len(features_relevant):,} rows")

# Step 2: For each eclass, find the top feature keys (most discriminative)
print("Computing top feature keys per eclass...")
eclass_feature_key_counts = features_relevant.groupby(['eclass', 'key']).agg(
    n_skus=('sku', 'nunique')
).reset_index()

# For each eclass, keep top N most common feature keys
top_keys_per_eclass = {}
for eclass, group in eclass_feature_key_counts.groupby('eclass'):
    top = group.nlargest(MAX_FEATURES_PER_CLUSTER, 'n_skus')['key'].tolist()
    top_keys_per_eclass[eclass] = set(top)

# Step 3: Build feature signatures for each SKU using top keys
print("Building feature signatures for SKUs...")
# Filter features to only top keys per eclass — VECTORIZED approach
# Build a set of valid (eclass, key) pairs
valid_eclass_key_pairs = set()
for ec, keys in top_keys_per_eclass.items():
    for k in keys:
        valid_eclass_key_pairs.add((ec, k))

# Convert to fast lookup via merge
valid_pairs_df = pd.DataFrame(list(valid_eclass_key_pairs), columns=['eclass', 'key'])
filtered_df = features_relevant.merge(valid_pairs_df, on=['eclass', 'key'], how='inner')

print(f"Filtered features (top keys only): {len(filtered_df):,} rows")

# Group by SKU and build signatures — optimized
sku_signatures = {}
if not filtered_df.empty:
    # Pre-compute: for each SKU, sort feature key-value pairs and join
    filtered_df['fval_clean'] = filtered_df['fvalue_set'].fillna(filtered_df['fvalue']).astype(str).str.strip()
    filtered_df['kv_pair'] = filtered_df['key'].astype(str).str.strip() + '=' + filtered_df['fval_clean']
    
    # Sort within each SKU and take top N features
    sig_df = (filtered_df.sort_values(['sku', 'kv_pair'])
              .groupby('sku')['kv_pair']
              .apply(lambda x: '|'.join(x.head(MAX_FEATURES_PER_CLUSTER)))
              .reset_index())
    sig_df.columns = ['sku', 'signature']
    sku_signatures = dict(zip(sig_df['sku'], sig_df['signature']))

print(f"Built signatures for {len(sku_signatures):,} SKUs")

# Step 4: Cluster SKUs within each eclass by their feature signatures
# (Feature signatures that are identical = same cluster)
# For signatures that are similar, we use a hash-based grouping
print("Clustering SKUs by feature signatures within eclasses...")

# Create cluster IDs: eclass + truncated signature hash
def make_cluster_id(eclass, signature):
    """Create a stable cluster ID from eclass and feature signature."""
    if not signature:
        return f"{eclass}__default"
    sig_hash = hashlib.md5(signature.encode()).hexdigest()[:8]
    return f"{eclass}__{sig_hash}"

sku_to_cluster = {}
cluster_info = {}  # cluster_id -> {eclass, signature, sku_count}

for sku, sig in sku_signatures.items():
    ec = sku_to_eclass.get(sku, 'unknown')
    cluster_id = make_cluster_id(ec, sig)
    sku_to_cluster[sku] = cluster_id
    if cluster_id not in cluster_info:
        cluster_info[cluster_id] = {'eclass': ec, 'signature': sig, 'sku_count': 0}
    cluster_info[cluster_id]['sku_count'] += 1

# SKUs without features get a default cluster per eclass
for sku in relevant_skus:
    if sku not in sku_to_cluster:
        ec = sku_to_eclass.get(sku, 'unknown')
        cluster_id = f"{ec}__default"
        sku_to_cluster[sku] = cluster_id
        if cluster_id not in cluster_info:
            cluster_info[cluster_id] = {'eclass': ec, 'signature': 'default', 'sku_count': 0}
        cluster_info[cluster_id]['sku_count'] += 1

print(f"Total clusters: {len(cluster_info):,}")
print(f"SKUs mapped to clusters: {len(sku_to_cluster):,}")

# Add cluster_id to training data
plis['cluster_id'] = plis['sku'].map(sku_to_cluster)
plis['cluster_id'] = plis['cluster_id'].fillna(plis['eclass'] + '__default')

# ============================================================
# FEATURE ENGINEERING: Warm-start buyers
# ============================================================

def compute_recurrence_scores_l3(buyer_data):
    """Compute recurrence scores per cluster_id."""
    buyer_data = buyer_data.copy()
    buyer_data['total_value'] = buyer_data['quantityvalue'].fillna(0) * buyer_data['vk_per_item'].fillna(0)
    buyer_data['quarter'] = buyer_data['orderdate'].dt.to_period('Q')
    
    total_quarters = buyer_data['quarter'].nunique()
    if total_quarters == 0:
        return pd.DataFrame()
    
    agg = buyer_data.groupby('cluster_id').agg(
        eclass=('eclass', 'first'),
        order_count=('set_id', 'nunique'),
        line_count=('sku', 'count'),
        total_spend=('total_value', 'sum'),
        total_qty=('quantityvalue', 'sum'),
        quarters_active=('quarter', 'nunique'),
        last_purchase=('orderdate', 'max'),
        avg_price=('vk_per_item', 'mean'),
        unique_skus=('sku', 'nunique'),
    ).reset_index()
    
    max_date = buyer_data['orderdate'].max()
    
    agg['frequency_score'] = np.log1p(agg['order_count']) / np.log1p(max(agg['order_count'].max(), 1))
    agg['consistency_score'] = agg['quarters_active'] / max(total_quarters, 1)
    agg['monetary_score'] = np.log1p(agg['total_spend']) / np.log1p(max(agg['total_spend'].max(), 1))
    agg['recency_days'] = (max_date - agg['last_purchase']).dt.days
    agg['recency_score'] = 1 - (agg['recency_days'] / max(agg['recency_days'].max(), 1))
    
    agg['recurrence_score'] = (
        0.30 * agg['frequency_score'] +
        0.35 * agg['consistency_score'] +
        0.20 * agg['monetary_score'] +
        0.15 * agg['recency_score']
    )
    
    annual_freq_estimate = agg['order_count'] * (12 / max(total_quarters * 3, 1))
    agg['estimated_annual_savings'] = np.sqrt(agg['avg_price'].clip(lower=0.01)) * annual_freq_estimate * 0.5
    agg['net_benefit'] = agg['estimated_annual_savings'] - ANNUAL_FEE_PER_ELEMENT
    
    return agg


def get_warm_predictions_l3(buyer_id, buyer_data):
    """Get cluster-based predictions for a warm-start buyer."""
    scores = compute_recurrence_scores_l3(buyer_data)
    if scores.empty:
        return []
    
    candidates = scores[
        (scores['recurrence_score'] >= WARM_THRESHOLD) |
        (scores['net_benefit'] > 0)
    ].copy()
    
    if len(candidates) < 2:
        candidates = scores.nlargest(min(5, len(scores)), 'recurrence_score')
    
    # Portfolio discipline: prefer non-default clusters, deduplicate per eclass
    deduped = []
    for eclass, group in candidates.groupby('eclass'):
        group_sorted = group.sort_values('recurrence_score', ascending=False)
        # Prefer non-default cluster
        non_default = group_sorted[~group_sorted['cluster_id'].str.endswith('__default')]
        if not non_default.empty:
            deduped.append(non_default.iloc[0])
            # Keep additional clusters only if strong
            for i in range(1, len(non_default)):
                row = non_default.iloc[i]
                if row['recurrence_score'] >= 0.55 and row['order_count'] >= 3:
                    deduped.append(row)
        else:
            deduped.append(group_sorted.iloc[0])
    
    if not deduped:
        return []
    
    result = pd.DataFrame(deduped)
    result = result.sort_values('recurrence_score', ascending=False).head(50)
    
    return result['cluster_id'].tolist()


# ============================================================
# FEATURE ENGINEERING: Cold-start buyers
# ============================================================

def find_similar_buyers(target_customer, all_training_data, top_n=50):
    """Find similar buyers from training data."""
    nace = str(target_customer['nace_code'])
    emp_count = target_customer['estimated_number_employees']
    
    buyer_attrs = all_training_data.groupby('legal_entity_id').agg(
        nace_code=('nace_code', 'first'),
        emp=('estimated_number_employees', 'first'),
        n_orders=('set_id', 'nunique'),
    ).reset_index()
    
    buyer_attrs['nace_code'] = buyer_attrs['nace_code'].astype(str)
    
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


def get_cold_predictions_l3(target_customer, all_training_data):
    """Get cluster-based predictions for a cold-start buyer."""
    similar_buyer_ids = find_similar_buyers(target_customer, all_training_data)
    
    if not similar_buyer_ids:
        return []
    
    sim_data = all_training_data[all_training_data['legal_entity_id'].isin(similar_buyer_ids)].copy()
    sim_data['total_value'] = sim_data['quantityvalue'].fillna(0) * sim_data['vk_per_item'].fillna(0)
    
    cluster_stats = sim_data.groupby('cluster_id').agg(
        eclass=('eclass', 'first'),
        n_buyers=('legal_entity_id', 'nunique'),
        total_orders=('set_id', 'nunique'),
        total_spend=('total_value', 'sum'),
        avg_price=('vk_per_item', 'mean'),
    ).reset_index()
    
    n_similar = len(similar_buyer_ids)
    cluster_stats['penetration'] = cluster_stats['n_buyers'] / max(n_similar, 1)
    cluster_stats['avg_freq'] = cluster_stats['total_orders'] / max(n_similar, 1)
    
    cluster_stats['cold_score'] = (
        cluster_stats['penetration'] * 0.5 +
        np.log1p(cluster_stats['avg_freq']) / np.log1p(max(cluster_stats['avg_freq'].max(), 1)) * 0.3 +
        np.log1p(cluster_stats['total_spend']) / np.log1p(max(cluster_stats['total_spend'].max(), 1)) * 0.2
    )
    
    candidates = cluster_stats[cluster_stats['penetration'] >= 0.25]
    candidates = candidates.sort_values('cold_score', ascending=False)
    
    # Portfolio discipline: top cluster per eclass
    deduped = []
    seen_eclass = set()
    for _, row in candidates.iterrows():
        ec = row['eclass']
        if ec not in seen_eclass:
            deduped.append(row)
            seen_eclass.add(ec)
        elif row['penetration'] >= 0.4:
            deduped.append(row)
    
    result = pd.DataFrame(deduped).head(COLD_MAX_ITEMS)
    return result['cluster_id'].tolist() if not result.empty else []


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

print("\nGenerating Level 3 predictions...")
for idx, row in customers.iterrows():
    buyer_id = row['legal_entity_id']
    task = row['task']
    
    if task == 'predict future' and buyer_id in buyer_groups:
        predictions = get_warm_predictions_l3(buyer_id, buyer_groups[buyer_id])
        source = 'warm'
    else:
        predictions = get_cold_predictions_l3(row, plis)
        source = 'cold'
    
    for cid in predictions:
        results.append({
            'legal_entity_id': buyer_id,
            'cluster': cid
        })
    
    print(f"  Buyer {buyer_id} ({source}): {len(predictions)} predictions")

# ============================================================
# OUTPUT
# ============================================================
submission = pd.DataFrame(results)
output_path = OUTPUT_DIR / "submission.csv"
submission.to_csv(output_path, index=False)

# Also save cluster mapping for documentation
cluster_map = pd.DataFrame([
    {'cluster_id': cid, 'eclass': info['eclass'], 'feature_signature': info['signature'], 'sku_count': info['sku_count']}
    for cid, info in cluster_info.items()
])
cluster_map.to_csv(OUTPUT_DIR / "cluster_mapping.csv", index=False)

print(f"\n{'='*60}")
print(f"Level 3 Submission saved to: {output_path}")
print(f"Cluster mapping saved to: {OUTPUT_DIR / 'cluster_mapping.csv'}")
print(f"Total predictions: {len(submission)}")
print(f"Unique buyers with predictions: {submission['legal_entity_id'].nunique()}")
print(f"Avg predictions per buyer: {len(submission)/max(submission['legal_entity_id'].nunique(),1):.1f}")
print(f"Total clusters used: {submission['cluster'].nunique()}")
print(f"{'='*60}")

warm_sub = submission[submission['legal_entity_id'].isin(warm_buyer_ids)]
cold_sub = submission[~submission['legal_entity_id'].isin(warm_buyer_ids)]
print(f"\nWarm-start: {warm_sub['legal_entity_id'].nunique()} buyers, {len(warm_sub)} predictions ({len(warm_sub)/max(warm_sub['legal_entity_id'].nunique(),1):.1f} avg)")
print(f"Cold-start: {cold_sub['legal_entity_id'].nunique()} buyers, {len(cold_sub)} predictions ({len(cold_sub)/max(cold_sub['legal_entity_id'].nunique(),1):.1f} avg)")
