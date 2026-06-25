"""
run.py — 9개 시나리오 순서대로 풀고 submission.csv 생성

Usage:
    # L1만 (빠른 테스트)
    python run.py --layers l1 --iters 50000 --restarts 3

    # 전체 9개 시나리오
    python run.py --layers l1 l2 l3 --iters 80000 --restarts 4

    # 이미 풀린 건 cache 재사용
    python run.py --layers l1 l2 l3 --iters 120000 --restarts 5
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

# kaggle 채점 case ID 순서
CASE_ORDER = [
    ("l1", 0.5), ("l1", 1.0), ("l1", 2.0),
    ("l2", 0.5), ("l2", 1.0), ("l2", 2.0),
    ("l3", 0.5), ("l3", 1.0), ("l3", 2.0),
]
CASE_ID_MAP = {v: i for i, v in enumerate(CASE_ORDER)}

BASE     = Path(__file__).parent / "ras_release_v1.1" / "ras_release_v1.1"
SOL_DIR  = BASE / "scoring" / "solutions"
SUB_PATH = BASE / "scoring" / "submission.csv"


def sol_path(layer: str, mult: float) -> Path:
    return SOL_DIR / f"solution_{layer}_{int(mult * 10):02d}.json"


def load_or_solve(layer, mult, iters, restarts, force):
    p = sol_path(layer, mult)
    if p.exists() and not force:
        print(f"[cache]  {p.name}")
        return p.read_text(encoding="utf-8")

    from solver import solve
    result = solve(layer, mult, sa_iters=iters, n_ils_restarts=restarts)
    text   = json.dumps(result, separators=(",", ":"), default=str)
    SOL_DIR.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    print(f"[saved]  {p.name}")
    return text


def make_submission(solutions: dict[int, str]) -> None:
    rows = [["ID", "data"]]
    for case_id in range(9):
        rows.append([case_id, solutions.get(case_id, "{}")])
    with open(SUB_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    solved = sum(1 for _, d in rows[1:] if d != "{}")
    print(f"\nsubmission.csv  →  {SUB_PATH}")
    print(f"Solved: {solved}/9 scenarios")


def validate_one(json_text: str, layer: str, mult: float) -> bool:
    """Run fast_validator_v1_1 in-process."""
    try:
        sys.path.insert(0, str(BASE / "scoring"))
        from fast_validator_v1_1 import fast_validate_payload
        od_path = BASE / "scoring" / "od_distance_matrix.csv"
        data    = json.loads(json_text)
        ok, res = fast_validate_payload(
            data,
            od_distance_matrix_path=od_path if od_path.exists() else None,
        )
        s = res.get("stress_metrics", {})
        print(f"  validate {layer}×{mult}: {'PASS' if ok else 'FAIL'}"
              f"  stress={s.get('stress_score', '?'):,.0f}"
              f"  loaded={s.get('loaded_demand_ratio', 0):.3f}")
        return ok
    except Exception as e:
        print(f"  validate {layer}×{mult}: ERROR — {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers",   nargs="+", default=["l1"],
                    help="Layers to solve: l1 l2 l3")
    ap.add_argument("--iters",    type=int,  default=80_000,
                    help="SA iterations per run")
    ap.add_argument("--restarts", type=int,  default=4,
                    help="ILS restarts (total SA runs = restarts+1)")
    ap.add_argument("--mults",    nargs="+", type=float, default=[0.5, 1.0, 2.0],
                    help="Demand multipliers to solve")
    ap.add_argument("--force",    action="store_true",
                    help="Re-solve even if cached JSON exists")
    ap.add_argument("--validate", action="store_true",
                    help="Run local validator after each solve")
    ap.add_argument("--no-sub",   action="store_true",
                    help="Skip writing submission.csv")
    args = ap.parse_args()

    t0        = time.time()
    solutions = {}

    # Load existing cache for cases we're not re-solving
    for (layer, mult), case_id in CASE_ID_MAP.items():
        p = sol_path(layer, mult)
        if p.exists():
            solutions[case_id] = p.read_text(encoding="utf-8")

    # Solve requested cases
    for layer in args.layers:
        for mult in args.mults:
            case_id = CASE_ID_MAP.get((layer, mult))
            if case_id is None:
                print(f"Unknown case ({layer}, {mult}) — skipped")
                continue

            print(f"\n{'─'*60}")
            print(f"Case {case_id}: {layer} ×{mult}")
            text = load_or_solve(layer, mult, args.iters, args.restarts, args.force)
            solutions[case_id] = text

            if args.validate:
                validate_one(text, layer, mult)

    if not args.no_sub:
        make_submission(solutions)

    print(f"\nTotal elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
