"""
Fast Validation & Parameter Optimization for Level 1
=====================================================
Pre-computes features once, then sweeps selection thresholds.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).resolve().parent.parent / "Challenge2"

# ============================================================
# LOAD DATA
# ============================================================
print("Loading data...")
plis = pd.read_csv(
    BASE_DIR / "plis_training.csv",
    sep='\t', low_memory=False,
    dtype={'eclass': str, 'nace_code': str, 'secondary_nace_code': str}
)
plis['orderdate'] = pd.to_datetime(plis['orderdate'])
plis['eclass'] = plis['eclass'].astype(str).str.strip()
plis = plis[plis['eclass'].str.match(r'^\d{8}$', na=False)].copy()
plis['total_value'] = plis['quantityvalue'].fillna(0) * plis['vk_per_item'].fillna(0)

customers = pd.read_csv(
    BASE_DIR / "customer_test.csv", sep='\t',
    dtype={'nace_code': str, 'secondary_nace_code': str}
)
warm_buyer_ids = set(customers[customers['task'] == 'predict future']['legal_entity_id'])

warm_data = plis[plis['legal_entity_id'].isin(warm_buyer_ids)].copy()

# ============================================================
# TEMPORAL SPLIT
# ============================================================
SPLIT_DATE = pd.Timestamp('2024-07-01')
train = warm_data[warm_data['orderdate'] < SPLIT_DATE].copy()
val = warm_data[warm_data['orderdate'] >= SPLIT_DATE].copy()

print(f"Train: {train['orderdate'].min()} to {train['orderdate'].max()} ({len(train):,} rows)")
print(f"Val:   {val['orderdate'].min()} to {val['orderdate'].max()} ({len(val):,} rows)")

# Ground truth
val['quarter'] = val['orderdate'].dt.to_period('Q')
val_econ = val.groupby(['legal_entity_id', 'eclass']).agg(
    val_orders=('set_id', 'nunique'),
    val_spend=('total_value', 'sum'),
    val_avg_price=('vk_per_item', 'mean'),
    val_quarters=('quarter', 'nunique'),
).reset_index()

val_total_quarters = val['quarter'].nunique()
val_econ['annual_freq'] = val_econ['val_orders'] * (12 / max(val_total_quarters * 3, 1))

truth_pairs = set(zip(val_econ['legal_entity_id'], val_econ['eclass']))
print(f"Ground truth: {len(truth_pairs):,} (buyer, eclass) pairs")

# ============================================================
# PRE-COMPUTE FEATURES FOR ALL WARM BUYERS (done once)
# ============================================================
print("\nPre-computing features for all warm buyers...")
train['quarter'] = train['orderdate'].dt.to_period('Q')
train['year'] = train['orderdate'].dt.year

all_features = []
for bid in warm_buyer_ids:
    bd = train[train['legal_entity_id'] == bid]
    if len(bd) == 0:
        continue

    max_date = bd['orderdate'].max()
    total_quarters = bd['quarter'].nunique()
    if total_quarters == 0:
        continue

    agg = bd.groupby('eclass').agg(
        order_count=('set_id', 'nunique'),
        total_spend=('total_value', 'sum'),
        quarters_active=('quarter', 'nunique'),
        n_years=('year', 'nunique'),
        last_purchase=('orderdate', 'max'),
        avg_price=('vk_per_item', 'mean'),
    ).reset_index()

    agg['legal_entity_id'] = bid
    agg['recency_days'] = (max_date - agg['last_purchase']).dt.days
    agg['recent'] = (agg['recency_days'] <= 180).astype(float)
    agg['total_quarters'] = total_quarters

    max_orders = agg['order_count'].max()
    max_spend = agg['total_spend'].max()
    max_recency = agg['recency_days'].max()

    agg['freq_norm'] = np.log1p(agg['order_count']) / np.log1p(max_orders) if max_orders > 0 else 0
    agg['consistency'] = agg['quarters_active'] / max(total_quarters, 1)
    agg['monetary_norm'] = np.log1p(agg['total_spend']) / np.log1p(max_spend) if max_spend > 0 else 0
    agg['recency_norm'] = 1 - (agg['recency_days'] / max(max_recency, 1))

    agg['annual_freq'] = agg['order_count'] * (12 / max(total_quarters * 3, 1))
    agg['est_savings_raw'] = np.sqrt(agg['avg_price'].clip(lower=0.01)) * agg['annual_freq']

    all_features.append(agg)

features_df = pd.concat(all_features, ignore_index=True)
print(f"Pre-computed features: {len(features_df):,} (buyer, eclass) pairs from {features_df['legal_entity_id'].nunique()} buyers")

# ============================================================
# FAST SCORING
# ============================================================
def fast_score(pred_mask, features, val_econ_df, monthly_fee, savings_mult):
    preds = features[pred_mask][['legal_entity_id', 'eclass']]
    n_preds = len(preds)
    if n_preds == 0:
        return 0, 0, 0, 0.0

    total_fees = n_preds * monthly_fee * 12
    hits = preds.merge(val_econ_df, on=['legal_entity_id', 'eclass'], how='inner')

    if len(hits) == 0:
        return -total_fees, 0, total_fees, 0.0

    savings = (np.sqrt(hits['val_avg_price'].clip(lower=0.01)) * hits['annual_freq'] * savings_mult).sum()
    hit_rate = len(hits) / n_preds
    return savings - total_fees, savings, total_fees, hit_rate


# ============================================================
# PARAMETER SWEEP — only varies selection logic on precomputed features
# ============================================================
print("\n" + "="*70)
print("PARAMETER SWEEP")
print("="*70)

scoring_scenarios = [
    (3, 1.0), (3, 2.0), (3, 5.0), (3, 10.0),
    (5, 1.0), (5, 2.0), (5, 5.0), (5, 10.0),
    (8, 2.0), (8, 5.0), (8, 10.0),
    (10, 2.0), (10, 5.0), (10, 10.0),
]

results = []
config_id = 0

weight_sets = [
    ('balanced', 0.30, 0.35, 0.20, 0.15),
    ('consistency', 0.20, 0.45, 0.15, 0.20),
    ('frequency', 0.40, 0.30, 0.15, 0.15),
    ('monetary', 0.20, 0.25, 0.35, 0.20),
]

for wname, w1, w2, w3, w4 in weight_sets:
    features_df[f'score_{wname}'] = (
        w1 * features_df['freq_norm'] +
        w2 * features_df['consistency'] +
        w3 * features_df['monetary_norm'] +
        w4 * features_df['recency_norm']
    )

mult_range = [0.5, 1.0, 2.0, 3.0, 5.0]
fee_range = [3, 5, 8, 10, 15]
threshold_range = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
cap_range = [30, 50, 100, 200, 500, 9999]
min_q_range = [2, 3, 4]

total = len(weight_sets) * len(threshold_range) * len(mult_range) * len(fee_range) * len(cap_range) * len(min_q_range)
print(f"Configs to test: {total}")

for wname, w1, w2, w3, w4 in weight_sets:
    score_col = f'score_{wname}'
    for thresh in threshold_range:
        for s_mult in mult_range:
            for a_fee in fee_range:
                annual_fee = a_fee * 12
                features_df['_net'] = features_df['est_savings_raw'] * s_mult - annual_fee

                for min_q in min_q_range:
                    base_mask = (
                        (features_df[score_col] >= thresh) &
                        (features_df['_net'] > 0) &
                        (features_df['quarters_active'] >= min_q)
                    ).values

                    for cap in cap_range:
                        if cap < 9999:
                            mask = base_mask.copy()
                            selected = features_df[mask].copy()
                            selected['_rank'] = selected.groupby('legal_entity_id')['_net'].rank(ascending=False)
                            drop_idx = selected[selected['_rank'] > cap].index
                            mask[drop_idx] = False
                        else:
                            mask = base_mask

                        config_id += 1

                        avg_net = 0
                        for sf, sm in scoring_scenarios:
                            net, _, _, _ = fast_score(mask, features_df, val_econ, sf, sm)
                            avg_net += net
                        avg_net /= len(scoring_scenarios)

                        results.append({
                            'id': config_id,
                            'weights': wname,
                            'thresh': thresh,
                            's_mult': s_mult,
                            'a_fee': a_fee,
                            'min_q': min_q,
                            'cap': cap,
                            'n_preds': int(mask.sum()) if isinstance(mask, np.ndarray) else int(mask.sum()),
                            'avg_net': avg_net,
                        })

                        if config_id % 2000 == 0:
                            best_so_far = max(results, key=lambda x: x['avg_net'])
                            print(f"  [{config_id}/{total}] best avg_net: €{best_so_far['avg_net']:,.0f} "
                                  f"(preds={best_so_far['n_preds']})")

res_df = pd.DataFrame(results).sort_values('avg_net', ascending=False)

print(f"\nSweep complete! Tested {config_id} configs.")
print(f"\n{'='*70}")
print("TOP 20 CONFIGURATIONS")
print(f"{'='*70}")
for _, row in res_df.head(20).iterrows():
    print(f"  avg_net=€{row['avg_net']:>10,.0f} | preds={row['n_preds']:>5} | "
          f"w={row['weights']:>12} thresh={row['thresh']:.2f} mult={row['s_mult']:.1f} "
          f"fee={row['a_fee']:>2} minq={row['min_q']} cap={row['cap']}")

# ============================================================
# DETAILED ANALYSIS OF BEST CONFIG
# ============================================================
best = res_df.iloc[0]
print(f"\n{'='*70}")
print("BEST CONFIG — DETAILED SCORING")
print(f"{'='*70}")
print(f"Weights: {best['weights']}, Threshold: {best['thresh']}, "
      f"Mult: {best['s_mult']}, Fee: {best['a_fee']}, MinQ: {best['min_q']}, Cap: {best['cap']}")

score_col = f"score_{best['weights']}"
annual_fee = best['a_fee'] * 12
features_df['_net_best'] = features_df['est_savings_raw'] * best['s_mult'] - annual_fee

best_mask = (
    (features_df[score_col] >= best['thresh']) &
    (features_df['_net_best'] > 0) &
    (features_df['quarters_active'] >= best['min_q'])
).values

if best['cap'] < 9999:
    selected = features_df[best_mask].copy()
    selected['_rank'] = selected.groupby('legal_entity_id')['_net_best'].rank(ascending=False)
    drop_idx = selected[selected['_rank'] > best['cap']].index
    best_mask[drop_idx] = False

best_preds = features_df[best_mask]
print(f"\nTotal predictions: {len(best_preds)}")
print(f"Buyers: {best_preds['legal_entity_id'].nunique()}")
print(f"Avg per buyer: {len(best_preds)/max(best_preds['legal_entity_id'].nunique(),1):.1f}")

print(f"\n{'Fee/mo':>8} {'Mult':>6} {'Hits':>6} {'HitRate':>8} {'Savings':>12} {'Fees':>12} {'Net':>12}")
print("-" * 70)
for sf in [3, 5, 8, 10, 15]:
    for sm in [1.0, 2.0, 5.0, 10.0]:
        net, sav, fees, hr = fast_score(best_mask, features_df, val_econ, sf, sm)
        n_hits = int(hr * int(best_mask.sum()))
        print(f"  €{sf:>5}/mo  {sm:>5.1f}  {n_hits:>5}  {hr:>7.1%}  €{sav:>10,.0f}  €{fees:>10,.0f}  €{net:>10,.0f}")

# Show the optimal parameter recommendation for the actual solution
print(f"\n{'='*70}")
print("RECOMMENDED PARAMETERS FOR solution_level1.py")
print(f"{'='*70}")
wname = best['weights']
for n, w1, w2, w3, w4 in weight_sets:
    if n == wname:
        print(f"  WARM_WEIGHTS = ({w1}, {w2}, {w3}, {w4})  # '{wname}'")
        break
print(f"  WARM_THRESHOLD = {best['thresh']}")
print(f"  SAVINGS_MULTIPLIER = {best['s_mult']}")
print(f"  MONTHLY_FEE = {best['a_fee']}")
print(f"  WARM_CAP = {best['cap']}")
print(f"  MIN_QUARTERS = {best['min_q']}")
