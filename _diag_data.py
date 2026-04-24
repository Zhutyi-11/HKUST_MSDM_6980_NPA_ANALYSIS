"""Diagnose all data files for dashboard."""
import os, json, csv

out = r'c:\Users\marcozhu\Desktop\6980\agent_outputs\baseline_comparison_run'

print('=== FILES ===')
for f in sorted(os.listdir(out)):
    fp = os.path.join(out, f)
    sz = os.path.getsize(fp)
    print(f'  {f}: {sz:,} bytes')

# metrics.json
print('\n=== METRICS.JSON ===')
with open(os.path.join(out,'metrics.json')) as fh:
    m = json.load(fh)
    for k,v in m.items():
        if isinstance(v,(dict,list)):
            print(f'  {k}: type={type(v).__name__}, len={len(v)}')
            if isinstance(v,dict):
                for k2 in list(v.keys())[:10]:
                    print(f'    .{k2}')
            elif isinstance(v,list) and v:
                print(f'    [0]={str(v[0])[:100]}')
        else:
            print(f'  {k}: {v}')

# model_comparison.json
print('\n=== MODEL_COMPARISON ===')
mc_path = os.path.join(out,'model_comparison.json')
if os.path.exists(mc_path):
    with open(mc_path) as fh:
        mc = json.load(fh)
        if isinstance(mc,list):
            print(f'  List of {len(mc)} models:')
            for mi,mdata in enumerate(mc):
                if isinstance(mdata,dict):
                    keys = list(mdata.keys())[:15]
                    name = mdata.get('model_name', mdata.get('name','?'))
                    print(f'  [{mi}] {name}')
                    print(f'       keys={keys}')
                else:
                    print(f'  [{mi}] {type(mdata).__name__} = {str(mdata)[:100]}')
        elif isinstance(mc,dict):
            for k,v in mc.items():
                if isinstance(v,(list,dict)):
                    print(f'  {k}: {type(v).__name__} len={len(v)}')
                else:
                    print(f'  {k}: {v}')

# feature_importance.json
print('\n=== FEATURE_IMPORTANCE ===')
fi_path = os.path.join(out,'feature_importance.json')
if os.path.exists(fi_path):
    with open(fi_path) as fh:
        fi = json.load(fh)
        if isinstance(fi,dict):
            for k,v in fi.items():
                if isinstance(v,list):
                    print(f'  {k}: list[{len(v)}]')
                    if v and isinstance(v[0],(dict)):
                        print(f'    [0] keys={list(v[0].keys())[:8]}')
                        for item in v[:2]:
                            print(f'    sample: {item}')
                else:
                    print(f'  {k}: {v}')
        elif isinstance(fi,list):
            print(f'  list[{len(fi)}]')
            if fi:
                if isinstance(fi[0],dict):
                    print(f'  [0] keys={list(fi[0].keys())[:8]}')

# accounts_scored.csv
print('\n=== ACCOUNTS_SCORED.CSV ===')
ac_path = os.path.join(out,'accounts_scored.csv')
if os.path.exists(ac_path):
    with open(ac_path) as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        print(f'  Rows: {len(rows)}, Cols: {len(rows[0]) if rows else 0}')
        if rows:
            print(f'  Cols: {list(rows[0].keys())}')
            r = rows[0]
            for k in ['id','prob','pred','label','action','expected_recovery','cost']:
                if k in r:
                    print(f'  [{k}]={r[k]}')

# tuning_results
print('\n=== TUNING_RESULTS ===')
tune_path = os.path.join(out,'tuning_results.json')
if os.path.exists(tune_path):
    with open(tune_path) as fh:
        tr = json.load(fh)
        if isinstance(tr,dict):
            for k,v in tr.items():
                if isinstance(v,list):
                    print(f'  {k}: list[{len(v)}]')
                    if v and isinstance(v[0],dict):
                        print(f'    keys={list(v[0].keys())[:10]}')
                else:
                    print(f'  {k}: {type(v).__name__} = {v}')
