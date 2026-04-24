"""Deep diagnosis: extract actual data structures for dashboard."""
import os, json, csv

out = r'c:\Users\marcozhu\Desktop\6980\agent_outputs\baseline_comparison_run'

# === metrics.json deep dive ===
print('=== CHAMPION_CHALLENGER DETAIL ===')
with open(os.path.join(out,'metrics.json')) as fh:
    m = json.load(fh)

for item in m.get('champion_challenger',[]):
    print(f'\n--- {item["model_name"]} ({item["model_role"]}) ---')
    for k,v in item.items():
        if isinstance(v,(list,dict)):
            print(f'  {k}: {type(v).__name__} len={len(v)}')
        else:
            print(f'  {k}: {v}')

print('\n=== MODEL_SELECTION DETAIL ===')
for model_name, mdata in m.get('model_selection',{}).items():
    print(f'\n--- {model_name} ---')
    if isinstance(mdata,dict):
        for k,v in mdata.items():
            print(f'  {k}: {v}')
    else:
        print(f'  {mdata}')

print('\n=== ALL_FEATURE_IMPORTANCE ===')
for model_name, fi_list in m.get('all_feature_importance',{}).items():
    print(f'\n--- {model_name} ---')
    if isinstance(fi_list,list):
        for fi in fi_list[:5]:
            print(f'  {fi}')
        if len(fi_list) > 5:
            print(f'  ... +{len(fi_list)-5} more')

print('\n=== TOP_FEATURES ===')
for tf in m.get('top_features',[])[:10]:
    print(f'  {tf}')

print('\n=== DESCRIPTIVE_STATS ===')
ds = m.get('descriptive_stats',{})
for k,v in ds.items():
    if isinstance(v,dict):
        print(f'  {k}: {json.dumps(v)[:200]}')
    else:
        print(f'  {k}: {v}')

print('\n=== QUEUE_SUMMARY ===')
for qs in m.get('queue_summary',[]):
    print(f'  {qs}')

print('\n=== CONCENTRATION ===')
c = m.get('concentration',{})
for k,v in c.items():
    if isinstance(v,list):
        print(f'  {k}: list[{len(v)}]')
        if v and len(v)>0:
            print(f'    [0]={str(v[0])[:150]}')
    elif isinstance(v,(dict)):
        print(f'  {k}: dict keys={list(v.keys())[:10]}')
    else:
        print(f'  {k}: {v}')

print('\n=== POLICY_SUMMARY ===')
ps = m.get('policy_summary',{})
for k,v in ps.items():
    print(f'  {k}: {v}')

print('\n=== BASELINE_POLICY_SUMMARY ===')
bps = m.get('baseline_policy_summary',{})
for k,v in bps.items():
    print(f'  {k}: {v}')

print('\n=== TUNING_SUMMARY ===')
ts = m.get('tuning_summary',{})
for k,v in ts.items():
    if isinstance(v,(dict)):
        print(f'  {k}: dict keys={list(v.keys())[:10]}')
        for k2,v2 in v.items():
            if isinstance(v2,(list)):
                print(f'  .{k2}: list[{len(v2)}]')
            else:
                print(f'  .{k2}: {v2}')
    else:
        print(f'  {k}: {v}')

print('\n=== TEST_METRICS ===')
tm = m.get('test_metrics',{})
for k,v in tm.items():
    print(f'  {k}: {v}')

print('\n=== BASELINE_TEST_METRICS ===')
btm = m.get('baseline_test_metrics',{})
for k,v in btm.items():
    print(f'  {k}: {v}')

print('\n=== PAYER_RATE BY BALANCE ===')
prb = m.get('payer_rate_by_balance',[])
for pr in prb:
    print(f'  {pr}')

print('\n=== DATA_OVERVIEW ===')
do = m.get('data_overview',{})
for k,v in do.items():
    print(f'  {k}: {v}')
