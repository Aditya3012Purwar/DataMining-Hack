"""Create diagnostic submissions to isolate the L2 matching problem."""
import pandas as pd
import csv

# Test 1: L1 eclass values renamed as 'cluster' (NO manufacturer)
l1 = pd.read_csv('../milestone1_eclass/submission.csv')
test1 = pd.DataFrame({
    'legal_entity_id': l1['legal_entity_id'],
    'cluster': l1['eclass'].astype(str)
})
test1.to_csv('diag_eclass_only.csv', index=False)
print('diag_eclass_only.csv - L1 eclasses as cluster column')
print(test1.head(3).to_string(index=False))
print(f'  Rows: {len(test1)}')
print()

# Test 2: Scorer's exact example for a warm buyer
test2 = pd.DataFrame({
    'legal_entity_id': [41165867],
    'cluster': ['30020903|Bissell']
})
test2.to_csv('diag_scorer_example.csv', index=False)
print('diag_scorer_example.csv - scorer\'s own example format')
print(test2.to_string(index=False))
print()

# Test 3: Quoted pipe values
with open('diag_quoted.csv', 'w', newline='') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(['legal_entity_id', 'cluster'])
    writer.writerow([61457883, '24200201|HP'])
    writer.writerow([61349064, '19060106|Samsung'])
    writer.writerow([60922385, '24360105|OtterBox'])
    writer.writerow([60326103, '19010107|Lenovo'])
    writer.writerow([61020458, '19069203|Jabra'])
print('diag_quoted.csv - all values quoted')
with open('diag_quoted.csv') as f:
    print(f.read())

# Test 4: Full submission with quoting on cluster only
sub = pd.read_csv('submission.csv')
with open('diag_full_quoted.csv', 'w', newline='') as f:
    f.write('legal_entity_id,cluster\n')
    for _, row in sub.iterrows():
        f.write(f'{row["legal_entity_id"]},"{row["cluster"]}"\n')
print(f'diag_full_quoted.csv - full submission with quoted clusters ({len(sub)} rows)')

# Test 5: Check raw bytes of our mini test
print('\n--- Raw bytes of mini test ---')
with open('submission_mini_test.csv', 'rb') as f:
    content = f.read()
print(content.decode('utf-8'))
print(repr(content[:300]))

# Show all diagnostic files
print('\n=== DIAGNOSTIC FILES ===')
print('Priority 1: diag_eclass_only.csv')
print('  -> If hits > 0, L2 scorer matches eclass-only too, and issue is manufacturer')
print('  -> If hits = 0, scorer is stricter about format/requires pipe')
print()
print('Priority 2: diag_scorer_example.csv')
print('  -> Tests scorer\'s own example (30020903|Bissell) for buyer 41165867')
print('  -> If hits = 0, either scorer is broken or buyer has no matching ground truth')
print()
print('Priority 3: diag_quoted.csv')
print('  -> Tests if pipe needs to be inside quotes')
