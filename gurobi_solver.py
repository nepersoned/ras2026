"""
gurobi_solver.py — MILP for the Railroad Blocking Problem
==========================================================

Decision variables:
  y[k,p]   binary  demand k uses path p  (direct or one-hop via hub)
  u[k]     binary  demand k is unserved
  x[b]     binary  block b is opened
  vol_b[b] cont    total railcar-volume through block b

Objective:
  min  Σ block_fixed·x[b]
     + Σ (dist·tc + ic_count·ic_rate)·vol_b[b]
     + Σ handling_cost[hub]·flow_at_hub
     + M · Σ unserved_carmiles

Constraints (hard):
  C1-assign   exactly one path or unserved per demand
  C2          outbound manifest/bulk blocks ≤ num_tracks per yard
  C3          classification volume ≤ handling_capacity per yard
  C4          vol_b[b] ≥ min_vol(dist_b) · x[b]
  C8          (implicit: blocks indexed by commodity_type)
  C9          (implicit: u[k]=1 for zero-volume demands)
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from solver import (
    COMMODITY_TO_BLOCK_TYPE, CLASSIFICATION, DIRECT_ONLY,
    Demand, Solution,
    load_layer, load_od_matrix, Network,
    build_json,
)

BASE    = Path(__file__).parent / "ras_release_v1.1" / "ras_release_v1.1"
SOL_DIR = BASE / "scoring" / "solutions"


def gurobi_solve(
    layer:             str,
    demand_multiplier: float = 1.0,
    time_limit:        float = 300.0,
    mip_gap:           float = 0.005,
    max_hops:          int   = 1,     # 0=direct only, 1=one intermediate hub
    verbose:           bool  = True,
) -> dict:
    print(f"\n{'='*64}")
    print(f"  [Gurobi MIP]  layer={layer}  x{demand_multiplier}")
    print(f"{'='*64}")
    t0 = time.time()

    nodes_df, links_df, demands_scaled, demands_raw, settings = \
        load_layer(layer, demand_multiplier)

    yard_rows = nodes_df[nodes_df["node_type"] == "yard"]
    yard_info = {
        int(r["node_id"]): {
            "num_tracks":        float(r.get("num_tracks",        9999) or 9999),
            "handling_capacity": float(r.get("handling_capacity", 1e9)  or 1e9),
            "handling_cost":     float(r.get("handling_cost",     0)    or 0),
        }
        for _, r in yard_rows.iterrows()
    }

    block_fixed = float(settings.get("block_fixed_cost",           1500.0))
    tc          = float(settings.get("transport_cost_coefficient",  1.0))
    ic_rate     = float(settings.get("interchange_cost",            100.0))
    M_pen       = float(settings.get("stress_penalty_M",            5.0))

    origin_ids = set(demands_scaled["origin_yard_id"].astype(int))
    dest_ids   = set(demands_scaled["dest_yard_id"].astype(int))
    all_yards  = sorted(origin_ids | dest_ids)

    # Load OD matrix + build network
    all_od_pairs = {(o, d) for o in all_yards for d in all_yards if o != d}
    od_matrix    = load_od_matrix(all_od_pairs)
    net = Network(nodes_df, links_df, origin_ids, dest_ids, settings, verbose)

    def c4_dist(o: int, d: int) -> float:
        v = od_matrix.get((o, d))
        return v if (v and v > 0) else net.dist(o, d)

    def min_vol_thr(o: int, d: int) -> float:
        dist = c4_dist(o, d)
        if not math.isfinite(dist): return 1e9
        s = settings
        if dist < 100:  return float(s.get("min_block_vol_short(<100mi)", 5))
        if dist <= 500: return float(s.get("min_block_vol_med(100-500mi)", 10))
        return float(s.get("min_block_vol_long(>500mi)", 15))

    # ── Enumerate demands ──────────────────────────────────────────────────────
    dem_list = []
    for _, row in demands_scaled.iterrows():
        dem_list.append({
            "id":    int(row["demand_id"]),
            "ct":    str(row["block_type"]),
            "orig":  int(row["origin_yard_id"]),
            "dest":  int(row["dest_yard_id"]),
            "vol":   int(row["volume"]),
            "sp":    od_matrix.get(
                (int(row["origin_yard_id"]), int(row["dest_yard_id"])),
                net.dist(int(row["origin_yard_id"]), int(row["dest_yard_id"]))
            ),
        })

    # ── Enumerate paths ────────────────────────────────────────────────────────
    dem_paths: list[list[list[tuple[int,int]]]] = []
    for dem in dem_list:
        paths: list[list[tuple[int,int]]] = []
        o, d, ct = dem["orig"], dem["dest"], dem["ct"]
        if dem["vol"] <= 0:
            dem_paths.append(paths)
            continue
        if net.has_path(o, d):
            paths.append([(o, d)])
        if ct not in DIRECT_ONLY and max_hops >= 1:
            for h in all_yards:
                if h != o and h != d and net.has_path(o, h) and net.has_path(h, d):
                    paths.append([(o, h), (h, d)])
        dem_paths.append(paths)

    # ── Block catalogue ────────────────────────────────────────────────────────
    block_set: set[tuple[int,int,str]] = set()
    for dem, paths in zip(dem_list, dem_paths):
        for path in paths:
            for seg in path:
                block_set.add((seg[0], seg[1], dem["ct"]))
    blocks = sorted(block_set)
    b2i    = {b: i for i, b in enumerate(blocks)}

    total_vol = sum(d["vol"] for d in dem_list if d["vol"] > 0)
    n_paths   = sum(len(ps) for ps in dem_paths)
    print(f"  Demands={len(dem_list)}  Blocks={len(blocks)}  Paths={n_paths}")

    # ── Gurobi model ───────────────────────────────────────────────────────────
    m = gp.Model("RAS_blocking")
    m.setParam("OutputFlag",  1 if verbose else 0)
    m.setParam("TimeLimit",   time_limit)
    m.setParam("MIPGap",      mip_gap)
    m.setParam("Threads",     0)          # use all cores
    m.setParam("MIPFocus",    1)          # favour feasible solutions

    # Variables
    y    = [[m.addVar(vtype=GRB.BINARY) for _ in dem_paths[k]]
             for k in range(len(dem_list))]
    u    = [m.addVar(vtype=GRB.BINARY) for _ in dem_list]
    x    = [m.addVar(vtype=GRB.BINARY) for _ in blocks]
    vb   = [m.addVar(lb=0.0)           for _ in blocks]   # block volumes (cont)
    m.update()

    # ── Constraints ──────────────────────────────────────────────────────────

    # C1-assign
    for k, (dem, paths) in enumerate(zip(dem_list, dem_paths)):
        if dem["vol"] <= 0 or not paths:
            m.addConstr(u[k] == 1)
        else:
            m.addConstr(gp.quicksum(y[k]) + u[k] == 1)

    # Block volume definition:  vb[b] = Σ vol_k · y[k][p]  for k,p using block b
    block_flow: dict[int, list[tuple[int,int,float]]] = defaultdict(list)
    for k, (dem, paths) in enumerate(zip(dem_list, dem_paths)):
        for p_idx, path in enumerate(paths):
            for seg in path:
                bi = b2i[(seg[0], seg[1], dem["ct"])]
                block_flow[bi].append((k, p_idx, float(dem["vol"])))

    for bi in range(len(blocks)):
        items = block_flow.get(bi, [])
        if not items:
            m.addConstr(vb[bi] == 0)
        else:
            m.addConstr(vb[bi] == gp.quicksum(v * y[k][p] for k, p, v in items))

    # C4 + block-open coupling
    for bi, blk in enumerate(blocks):
        mv = min_vol_thr(blk[0], blk[1])
        m.addConstr(vb[bi] >= mv * x[bi])            # C4
        m.addConstr(vb[bi] <= total_vol * x[bi])     # open only if flow

    # C2 — track limits (Manifest/Bulk only)
    yard_mf_blocks: dict[int, list[int]] = defaultdict(list)
    for bi, (o, d, ct) in enumerate(blocks):
        if COMMODITY_TO_BLOCK_TYPE.get(ct, ct) in CLASSIFICATION:
            yard_mf_blocks[o].append(bi)
    for yard, bis in yard_mf_blocks.items():
        tracks = int(yard_info.get(yard, {}).get("num_tracks", 9999))
        if tracks < 9999:
            m.addConstr(gp.quicksum(x[bi] for bi in bis) <= tracks)

    # C3 — handling capacity
    hub_flows: dict[int, list[tuple[int,int,float]]] = defaultdict(list)
    for k, (dem, paths) in enumerate(zip(dem_list, dem_paths)):
        for p_idx, path in enumerate(paths):
            if len(path) > 1:
                hub = path[0][1]
                hub_flows[hub].append((k, p_idx, float(dem["vol"])))
    for hub, items in hub_flows.items():
        cap = yard_info.get(hub, {}).get("handling_capacity", 1e9)
        if cap < 1e8:
            m.addConstr(gp.quicksum(v * y[k][p] for k, p, v in items) <= cap)

    # ── Objective ────────────────────────────────────────────────────────────
    fixed_expr = block_fixed * gp.quicksum(x)

    # Per-block: (dist·tc + interchanges·ic_rate) · vb[b]
    trans_ic = gp.quicksum(
        (net.dist(o, d) * tc + net.interchanges(o, d) * ic_rate) * vb[bi]
        for bi, (o, d, ct) in enumerate(blocks)
        if math.isfinite(net.dist(o, d))
    )

    # Handling at intermediate hubs
    hdl = gp.LinExpr()
    for hub, items in hub_flows.items():
        hcost = yard_info.get(hub, {}).get("handling_cost", 0.0)
        if hcost > 0:
            for k, p, v in items:
                hdl += hcost * v * y[k][p]

    # Unserved car-mile penalty
    unserved = gp.quicksum(
        M_pen * dem["vol"] * (dem["sp"] if math.isfinite(dem.get("sp", math.inf)) else 0)
        * u[k]
        for k, dem in enumerate(dem_list)
        if dem["vol"] > 0 and math.isfinite(dem.get("sp", math.inf))
    )

    m.setObjective(fixed_expr + trans_ic + hdl + unserved, GRB.MINIMIZE)

    print(f"  Model built ({time.time()-t0:.1f}s)  optimising …")
    m.optimize()

    status = m.Status
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.SUBOPTIMAL):
        print(f"  Gurobi: no solution (status={status})")
        return {}

    print(f"  Obj={m.ObjVal:,.0f}  gap={m.MIPGap*100:.3f}%  "
          f"time={time.time()-t0:.1f}s")

    # ── Extract solution ──────────────────────────────────────────────────────
    sol_demands, routings = [], []
    for k, (dem, paths) in enumerate(zip(dem_list, dem_paths)):
        d_obj = Demand(
            idx            = k,
            demand_id      = dem["id"],
            commodity_type = dem["ct"],
            block_type     = COMMODITY_TO_BLOCK_TYPE.get(dem["ct"], "Manifest"),
            origin         = dem["orig"],
            dest           = dem["dest"],
            volume         = dem["vol"],
            sp_dist        = dem["sp"],
        )
        sol_demands.append(d_obj)
        route = None
        for p_idx, path in enumerate(paths):
            if y[k][p_idx].X > 0.5:
                route = path
                break
        routings.append(route)

    sol = Solution(sol_demands, routings)
    return build_json(sol, net, od_matrix, settings, nodes_df, links_df, demands_raw)


def main():
    import argparse, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers",   nargs="+", default=["l1"])
    ap.add_argument("--mults",    nargs="+", type=float, default=[0.5, 1.0, 2.0])
    ap.add_argument("--timelimit", type=float, default=300.0)
    ap.add_argument("--gap",      type=float, default=0.005)
    ap.add_argument("--hops",     type=int,   default=1)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--no-sub",   action="store_true")
    args = ap.parse_args()

    CASE_ORDER = [
        ("l1",0.5),("l1",1.0),("l1",2.0),
        ("l2",0.5),("l2",1.0),("l2",2.0),
        ("l3",0.5),("l3",1.0),("l3",2.0),
    ]
    CASE_ID_MAP = {v: i for i, v in enumerate(CASE_ORDER)}

    import sys
    sys.path.insert(0, str(BASE / "scoring"))

    SOL_DIR.mkdir(parents=True, exist_ok=True)
    solutions = {}

    # Load existing cached solutions
    for (layer, mult), cid in CASE_ID_MAP.items():
        p = SOL_DIR / f"solution_{layer}_{int(mult*10):02d}.json"
        if p.exists():
            solutions[cid] = p.read_text(encoding="utf-8")

    for layer in args.layers:
        for mult in args.mults:
            cid = CASE_ID_MAP.get((layer, mult))
            if cid is None:
                continue
            print(f"\n{'─'*60}\nCase {cid}: {layer} ×{mult}")
            result = gurobi_solve(layer, mult,
                                  time_limit=args.timelimit,
                                  mip_gap=args.gap,
                                  max_hops=args.hops)
            if not result:
                print("  No solution.")
                continue
            text = json.dumps(result, separators=(",", ":"), default=str)
            p    = SOL_DIR / f"solution_{layer}_{int(mult*10):02d}.json"
            p.write_text(text, encoding="utf-8")
            solutions[cid] = text
            print(f"[saved] {p.name}")

            if args.validate:
                from fast_validator_v1_1 import fast_validate_payload
                od = BASE / "scoring" / "od_distance_matrix.csv"
                ok, res = fast_validate_payload(
                    result, od_distance_matrix_path=od if od.exists() else None)
                s = res.get("stress_metrics", {})
                print(f"  Validate: {'PASS' if ok else 'FAIL'}"
                      f"  stress={s.get('stress_score','?'):,.0f}"
                      f"  loaded={s.get('loaded_demand_ratio',0):.3f}")
                if not ok:
                    for k, v in res.get("checks", {}).items():
                        if not v.get("pass"):
                            print(f"    FAIL {k}: {v}")

    if not args.no_sub:
        sub_path = BASE / "scoring" / "submission.csv"
        rows = [["ID", "data"]]
        for cid in range(9):
            rows.append([cid, solutions.get(cid, "{}")])
        with open(sub_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)
        solved = sum(1 for _, d in rows[1:] if d != "{}")
        print(f"\nsubmission.csv → {sub_path}  ({solved}/9 scenarios)")


if __name__ == "__main__":
    main()
