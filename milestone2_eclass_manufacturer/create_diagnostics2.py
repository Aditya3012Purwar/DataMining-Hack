"""Create diagnostic L2 submissions testing various eclass format hypotheses."""
import pandas as pd

sub = pd.read_csv('submission.csv')

# HYPOTHESIS 1: Ground truth uses float eclass: "27050401.0|Lenovo"
h1 = sub.copy()
parts = h1['cluster'].str.split('|', n=1)
h1['cluster'] = parts.str[0] + '.0|' + parts.str[1]
h1.head(10).to_csv('diag_float_eclass.csv', index=False)
print('diag_float_eclass.csv (10 rows):')
print(h1.head(5).to_string(index=False))
print()

# HYPOTHESIS 2: Space after pipe: "27050401| Lenovo"  
h2 = sub.copy()
parts = h2['cluster'].str.split('|', n=1)
h2['cluster'] = parts.str[0] + '| ' + parts.str[1]
h2.head(10).to_csv('diag_space_after_pipe.csv', index=False)
print('diag_space_after_pipe.csv:')
print(h2.head(3).to_string(index=False))
print()

# HYPOTHESIS 3: Space before pipe: "27050401 |Lenovo"
h3 = sub.copy()
parts = h3['cluster'].str.split('|', n=1)
h3['cluster'] = parts.str[0] + ' |' + parts.str[1]
h3.head(10).to_csv('diag_space_before_pipe.csv', index=False)
print('diag_space_before_pipe.csv:')
print(h3.head(3).to_string(index=False))
print()

# HYPOTHESIS 4: Double pipe: "27050401||Lenovo"
h4 = sub.copy()
parts = h4['cluster'].str.split('|', n=1)
h4['cluster'] = parts.str[0] + '||' + parts.str[1]
h4.head(10).to_csv('diag_double_pipe.csv', index=False)
print('diag_double_pipe.csv:')
print(h4.head(3).to_string(index=False))
print()

# HYPOTHESIS 5: Manufacturer lowercased in ground truth: "27050401|lenovo"
h5 = sub.copy()
parts = h5['cluster'].str.split('|', n=1)
h5['cluster'] = parts.str[0] + '|' + parts.str[1].str.lower()
h5.head(10).to_csv('diag_lower_mfr.csv', index=False)
print('diag_lower_mfr.csv:')
print(h5.head(3).to_string(index=False))
print()

# HYPOTHESIS 6: Tab separator (TSV) since training data is TSV
with open('diag_tsv.csv', 'w', newline='') as f:
    f.write('legal_entity_id\tcluster\n')
    for _, row in sub.head(10).iterrows():
        f.write(f'{row["legal_entity_id"]}\t{row["cluster"]}\n')
print('diag_tsv.csv (tab-separated):')
with open('diag_tsv.csv') as f:
    print(f.read()[:200])

# HYPOTHESIS 7: sku-based cluster instead of manufacturer name?
# Load training to check if there's a different field
plis = pd.read_csv('../Challenge2/plis_training.csv/plis_training.csv', sep='\t',
                    low_memory=False, dtype={'eclass': str}, nrows=10)
print('\nTraining columns:', list(plis.columns))
print('Sample row:')
print(plis.iloc[0].to_string())

# HYPOTHESIS 8: Maybe manufacturer has extra spaces/encoding
print('\nChecking for hidden chars in manufacturer names...')
plis_full = pd.read_csv('../Challenge2/plis_training.csv/plis_training.csv', sep='\t',
                         low_memory=False, dtype={'eclass': str},
                         usecols=['manufacturer'])
mfrs = plis_full['manufacturer'].dropna().unique()
# Check for leading/trailing whitespace
has_space = [m for m in mfrs if m != m.strip()]
print(f'Manufacturers with leading/trailing spaces: {len(has_space)}')
if has_space:
    print(f'  Examples: {has_space[:5]}')

# Check for non-ASCII characters
import re
non_ascii = [m for m in mfrs if re.search(r'[^\x00-\x7F]', str(m))]
print(f'Manufacturers with non-ASCII chars: {len(non_ascii)}')
if non_ascii:
    print(f'  Examples: {non_ascii[:10]}')

print('\n=== FILES TO TRY ===')
print('1. diag_float_eclass.csv     -> "27050401.0|Lenovo" (MOST LIKELY)')
print('2. diag_lower_mfr.csv        -> "27050401|lenovo"')
print('3. diag_space_after_pipe.csv  -> "27050401| Lenovo"')
print('4. diag_double_pipe.csv       -> "27050401||Lenovo"')
print('5. diag_tsv.csv               -> tab-separated')
