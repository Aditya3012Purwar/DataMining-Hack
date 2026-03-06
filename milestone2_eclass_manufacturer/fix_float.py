"""Fix L2 submissions: add .0 float suffix to eclass part of cluster."""
import pandas as pd

# Fix main submission
sub = pd.read_csv('submission.csv')
parts = sub['cluster'].str.split('|', n=1)
sub['cluster'] = parts.str[0] + '.0|' + parts.str[1]
sub.to_csv('submission.csv', index=False)
n = len(sub)
nb = sub['legal_entity_id'].nunique()
print(f'Fixed submission.csv: {n} rows, {nb} buyers, EUR {n*10:,} fee')
print('Sample:')
print(sub.head(10).to_string(index=False))

# Fix L1-derived submission
l1sub = pd.read_csv('submission_from_l1.csv')
parts2 = l1sub['cluster'].str.split('|', n=1)
l1sub['cluster'] = parts2.str[0] + '.0|' + parts2.str[1]
l1sub.to_csv('submission_from_l1.csv', index=False)
print(f'\nFixed submission_from_l1.csv: {len(l1sub)} rows')
print(l1sub.head(5).to_string(index=False))
