"""Create L2 submission derived from L1's working predictions by adding manufacturer.
Uses vectorized pandas merge instead of row-by-row loop."""
import pandas as pd

print("Loading data...")
l1 = pd.read_csv('../milestone1_eclass/submission.csv')
l1['eclass'] = l1['eclass'].astype(str)
print(f'L1: {len(l1)} predictions')

plis = pd.read_csv('../Challenge2/plis_training.csv/plis_training.csv', sep='\t',
                    low_memory=False, dtype={'eclass': str})
plis['eclass'] = plis['eclass'].astype(str).str.strip()
plis = plis[plis['eclass'].str.match(r'^\d{8}$', na=False)].copy()
plis['mfr'] = plis['manufacturer'].fillna('UNKNOWN').astype(str).str.strip()
plis['spend'] = plis['quantityvalue'].fillna(0) * plis['vk_per_item'].fillna(0)

customers = pd.read_csv('../Challenge2/customer_test.csv/customer_test.csv', sep='\t')
warm_ids = set(customers[customers['task'] == 'predict future']['legal_entity_id'])

# --- WARM BUYERS: top manufacturer per (buyer, eclass) by spend ---
print("Computing top manufacturers per (buyer, eclass)...")
warm_plis = plis[plis['legal_entity_id'].isin(warm_ids)]
warm_mfr = warm_plis.groupby(['legal_entity_id', 'eclass', 'mfr'])['spend'].sum().reset_index()
warm_mfr['rank'] = warm_mfr.groupby(['legal_entity_id', 'eclass'])['spend'].rank(
    ascending=False, method='first')
top_mfr = warm_mfr[warm_mfr['rank'] == 1][['legal_entity_id', 'eclass', 'mfr', 'spend']]

# Merge with L1 warm predictions
l1_warm = l1[l1['legal_entity_id'].isin(warm_ids)]
warm_result = l1_warm.merge(top_mfr, on=['legal_entity_id', 'eclass'], how='left')
warm_result = warm_result.dropna(subset=['mfr'])
warm_result['cluster'] = warm_result['eclass'] + '|' + warm_result['mfr']

# --- COLD BUYERS: global top manufacturer per eclass ---
print("Computing global top manufacturers per eclass...")
global_mfr = plis.groupby(['eclass', 'mfr'])['spend'].sum().reset_index()
global_mfr['rank'] = global_mfr.groupby('eclass')['spend'].rank(
    ascending=False, method='first')
global_top = global_mfr[global_mfr['rank'] == 1][['eclass', 'mfr']].rename(
    columns={'mfr': 'mfr_global'})

l1_cold = l1[~l1['legal_entity_id'].isin(warm_ids)]
cold_result = l1_cold.merge(global_top, on='eclass', how='left')
cold_result = cold_result.dropna(subset=['mfr_global'])
cold_result['cluster'] = cold_result['eclass'] + '|' + cold_result['mfr_global']

# --- COMBINE ---
submission = pd.concat([
    warm_result[['legal_entity_id', 'cluster']],
    cold_result[['legal_entity_id', 'cluster']],
], ignore_index=True)

submission.to_csv('submission_from_l1.csv', index=False)

n = len(submission)
nb = submission['legal_entity_id'].nunique()
print(f'\nL2-from-L1: {n} predictions, {nb} buyers')
print(f'Fee: EUR {n * 10:,}')
print(f'Warm: {len(warm_result)} preds')
print(f'Cold: {len(cold_result)} preds')
print()
print('Sample warm:')
print(warm_result[['legal_entity_id', 'cluster']].head(10).to_string(index=False))
print()
print('Sample cold:')
print(cold_result[['legal_entity_id', 'cluster']].head(10).to_string(index=False))

# --- MINI TEST: 10 most certain predictions ---
print("\n--- Mini test submission ---")
warm_sorted = warm_result.sort_values('spend', ascending=False)
mini = warm_sorted[['legal_entity_id', 'cluster']].head(10)
mini.to_csv('submission_mini_test.csv', index=False)
print(mini.to_string(index=False))
print(f'\nMini test fee: EUR {len(mini) * 10}')
