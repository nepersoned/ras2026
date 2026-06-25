import sys, json
sys.path.insert(0, 'ras_release_v1.1/ras_release_v1.1/scoring')
from fast_validator_v1_1 import fast_validate_payload

od = 'ras_release_v1.1/ras_release_v1.1/scoring/od_distance_matrix.csv'

files = [
    ('ras_release_v1.1/ras_release_v1.1/scoring/sample_solutions/solution_result_l1_10.json', 'SAMPLE'),
    ('ras_release_v1.1/ras_release_v1.1/scoring/solutions/solution_l1_10.json', 'OURS'),
]

for fname, label in files:
    with open(fname) as f:
        data = json.load(f)
    ok, res = fast_validate_payload(data, od_distance_matrix_path=od)
    c = res.get('cost', {})
    s = res.get('stress_metrics', {})
    bd = data['outputs']['1 Block Design']
    sq = data['outputs']['2 Blocking Sequence']
    print(f'--- {label} ---')
    print(f'  PASS={ok}  blocks={len(bd)}  seqs={len(sq)}')
    print(f'  fixed={c.get("fixed",0):,.0f}')
    print(f'  transport={c.get("transport",0):,.0f}')
    print(f'  handling={c.get("handling",0):,.0f}')
    print(f'  interchange={c.get("interchange",0):,.0f}')
    print(f'  total_cost={c.get("total",0):,.0f}')
    print(f'  stress={s.get("stress_score",0):,.0f}  loaded={s.get("loaded_demand_ratio",0):.4f}')
    for name, info in res.get('checks', {}).items():
        if not info.get('pass'):
            print(f'  [FAIL] {name}: {info}')
    print()
