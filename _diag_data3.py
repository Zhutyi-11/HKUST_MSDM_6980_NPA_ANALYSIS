"""Check test scored accounts and report - encoding safe."""
import os, json, csv, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

out = r'c:\Users\marcozhu\Desktop\6980\agent_outputs\baseline_comparison_run'

ac_path = os.path.join(out,'test_scored_accounts.csv')
print('=== TEST_SCORED_ACCOUNTS.CSV ===')
with open(ac_path,encoding='utf-8-sig',errors='replace') as fh:
    reader = csv.DictReader(fh)
    rows = list(reader)
print(f'  Rows: {len(rows)}')
if rows:
    print(f'  Cols ({len(rows[0])}): {list(rows[0].keys())}')
    r = rows[0]
    for k in list(r.keys())[:20]:
        v = r[k]
        sv = str(v).encode('ascii','replace').decode('ascii')
        if len(sv) > 60:
            print(f'  [{k}]={sv[:60]}...')
        else:
            print(f'  [{k}]={sv}')

rp_path = os.path.join(out,'collection_strategy_report.md')
print('\n=== REPORT (first 2000 chars) ===')
with open(rp_path,encoding='utf-8',errors='replace') as fh:
    content = fh.read()
print(f'Total length: {len(content)} chars')
# Safe print - replace non-ascii
safe = content[:2000].encode('ascii','replace').decode('ascii')
print(safe)
