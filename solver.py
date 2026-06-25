"""
RAS 2026 PSC Solver  —  Greedy + SA + ILS   (v2)
=================================================

Key design choices (validated against fast_validator_v1_1):

  1. INTERCHANGE-PENALIZED DIJKSTRA
     Edge weight = link.length + (interchange_cost / transport_coeff) × is_rr_change
     ≡  link.length + 100 miles-equivalent per railroad boundary
     Finds paths that minimise total_cost (transport + interchange), not just miles.

  2. ACCURATE COST MODEL (mirrors validator exactly)
     fixed + transport + handling + interchange + M × unserved_carmiles
     + LARGE × C2_overuse + LARGE × C3_overuse

  3. OD-MATRIX DISTANCES FOR C4 / C6 CHECKS
     Loaded from od_distance_matrix.csv if available; Dijkstra fallback.
     Prevents borderline violations when validator uses different distance source.

  4. METAHEURISTIC STACK
     Greedy constructor → SA (4 neighbourhood types) → ILS (SA + perturbation)
"""

from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as sp_dijkstra

# ─────────────────────────────────────────────────────────────────────────────
# Domain constants
# ─────────────────────────────────────────────────────────────────────────────

COMMODITY_TO_BLOCK_TYPE: dict[str, str] = {
    "Merchandise": "Manifest",
    "Coal":        "Bulk",
    "Grain":       "Bulk",
    "Intermodal":  "Intermodal",
    "Automobile":  "Multilevel",
}
DIRECT_ONLY    = {"Intermodal", "Automobile"}
CLASSIFICATION = {"Manifest", "Bulk"}
LARGE_PENALTY  = 1e10

BASE     = Path(__file__).parent / "ras_release_v1.1" / "ras_release_v1.1"
OD_FILE  = BASE / "scoring" / "od_distance_matrix.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_layer(layer: str, demand_multiplier: float = 1.0):
    d             = BASE / "datasets" / layer
    nodes_df      = pd.read_csv(d / "node.csv")
    links_df      = pd.read_csv(d / "link.csv")
    demands_raw   = pd.read_csv(d / "demand.csv")
    settings_df   = pd.read_csv(d / "setting.csv")

    settings: dict = {}
    for _, row in settings_df.iterrows():
        try:    settings[row["parameter"]] = float(row["value"])
        except: settings[row["parameter"]] = row["value"]
    settings["demand_multiplier"] = demand_multiplier

    demands_scaled = demands_raw.copy()
    demands_scaled["volume"] = (
        demands_raw["volume"].astype(float) * demand_multiplier
    ).apply(lambda x: max(0, int(x)))

    return nodes_df, links_df, demands_scaled, demands_raw, settings


def load_od_matrix(pairs: set[tuple[int, int]]) -> dict[tuple[int, int], float]:
    """Load only the OD pairs we need from od_distance_matrix.csv."""
    if not OD_FILE.exists():
        return {}
    try:
        df  = pd.read_csv(OD_FILE)
        df  = df[["from_yard_id", "to_yard_id", "min_distance_mile"]].dropna()
        req = pd.DataFrame(list(pairs), columns=["from_yard_id", "to_yard_id"])
        merged = df.merge(req, on=["from_yard_id", "to_yard_id"])
        return {
            (int(r["from_yard_id"]), int(r["to_yard_id"])): float(r["min_distance_mile"])
            for _, r in merged.iterrows()
        }
    except Exception as e:
        print(f"  [OD matrix] load failed: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _norm_rr(v) -> str:
    s = str(v).strip()
    return "" if s in {"", "-1", "nan", "None", "NONE", "null", "NULL"} else s


def _count_interchanges(node_path: list[int], rr_map: dict[int, str]) -> int:
    """Count railroad boundary crossings — mirrors fast_validator_v1_1 exactly."""
    n = 0
    cur = _norm_rr(rr_map.get(node_path[0], ""))
    for nid in node_path[1:]:
        nxt = _norm_rr(rr_map.get(nid, ""))
        if not nxt:
            continue
        if not cur:
            cur = nxt
            continue
        if cur != nxt:
            n  += 1
            cur = nxt
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Network (graph + Dijkstra precomputation)
# ─────────────────────────────────────────────────────────────────────────────

class Network:
    """
    Directed graph with interchange-penalized edge weights.
    Precomputes yard-to-yard shortest paths, physical paths, and
    railroad-change counts.
    """

    def __init__(
        self,
        nodes_df:       pd.DataFrame,
        links_df:       pd.DataFrame,
        origin_ids:     set[int],
        dest_ids:       set[int],
        settings:       dict,
        verbose:        bool = True,
    ):
        ic_rate   = float(settings.get("interchange_cost",           100.0))
        tc_rate   = float(settings.get("transport_cost_coefficient", 1.0))
        # 1 railroad change ≡ ic_rate / tc_rate miles of extra weight
        rr_penalty = ic_rate / tc_rate if tc_rate > 0 else 100.0

        # Node railroad map
        self.rr_map: dict[int, str] = {
            int(r["node_id"]): _norm_rr(r.get("railroad_id", ""))
            for _, r in nodes_df.iterrows()
        }

        # Build graph
        node_set = (
            set(nodes_df["node_id"].astype(int))
            | set(links_df["from_node_id"].astype(int))
            | set(links_df["to_node_id"].astype(int))
        )
        self._nl  = sorted(node_set)
        self._n2i = {n: i for i, n in enumerate(self._nl)}
        N         = len(self._nl)

        rows, cols, data_w, data_w_unpen = [], [], [], []
        self._lid_lookup: dict[tuple[int, int], int]   = {}
        self._len_lookup: dict[tuple[int, int], float] = {}

        for _, lk in links_df.iterrows():
            u   = int(lk["from_node_id"])
            v   = int(lk["to_node_id"])
            w   = float(lk["length"])
            lid = int(lk["link_id"])
            ui  = self._n2i.get(u)
            vi  = self._n2i.get(v)
            if ui is None or vi is None:
                continue

            # Interchange penalty: charge if railroad changes
            rr_u = self.rr_map.get(u, "")
            rr_v = self.rr_map.get(v, "")
            penalty = rr_penalty if (rr_u and rr_v and rr_u != rr_v) else 0.0

            rows.append(ui); cols.append(vi)
            data_w.append(w + penalty)   # penalized
            data_w_unpen.append(w)       # unpenalized (for C6 fallback)

            # Keep minimum-weight link for same (u, v) pair
            if (u, v) not in self._lid_lookup or w < self._len_lookup.get((u, v), math.inf):
                self._lid_lookup[(u, v)] = lid
                self._len_lookup[(u, v)] = w

        self._mat = csr_matrix((data_w, (rows, cols)), shape=(N, N))

        # Unpenalized matrix (for C6 fallback — pure shortest path)
        self._mat_unpen = csr_matrix((data_w_unpen, (rows, cols)), shape=(N, N))

        # Precompute paths
        self.od_dist:         dict[tuple[int, int], float] = {}
        self.od_path_n:       dict[tuple[int, int], str]   = {}
        self.od_path_l:       dict[tuple[int, int], str]   = {}
        self.od_interchanges: dict[tuple[int, int], int]   = {}

        self._run_dijkstra(origin_ids, dest_ids, verbose)

    # ── Dijkstra ─────────────────────────────────────────────────────────────

    def _run_dijkstra(self, origins: set[int], dests: set[int], verbose: bool):
        src_list  = sorted(o for o in origins if o in self._n2i)
        dest_set  = {d for d in dests if d in self._n2i}
        if verbose:
            print(f"  Dijkstra (IC-penalised + unpen): {len(src_list)} sources", flush=True)
        t0 = time.time()

        BATCH = 250
        if len(src_list) <= BATCH:
            idxs          = [self._n2i[o] for o in src_list]
            d_pen, p_pen  = sp_dijkstra(self._mat,       directed=True, indices=idxs,
                                        return_predecessors=True)
            d_unp, p_unp  = sp_dijkstra(self._mat_unpen, directed=True, indices=idxs,
                                        return_predecessors=True)
            for ri, orig in enumerate(src_list):
                self._fill(orig, d_pen[ri], p_pen[ri], d_unp[ri], p_unp[ri], dest_set)
        else:
            for orig in src_list:
                si = self._n2i[orig]
                d_pen, p_pen = sp_dijkstra(self._mat,       directed=True, indices=si,
                                           return_predecessors=True)
                d_unp, p_unp = sp_dijkstra(self._mat_unpen, directed=True, indices=si,
                                           return_predecessors=True)
                self._fill(orig, d_pen, p_pen, d_unp, p_unp, dest_set)

        if verbose:
            print(f"  done in {time.time()-t0:.1f}s  ({len(self.od_dist)} paths)")

    MAX_CIRCUITOUS = 1.28   # slightly under validator's 1.30 for safety

    def _fill(self, orig: int,
              dist_arr_pen, pred_arr_pen,
              dist_arr_unp, pred_arr_unp,
              dest_set: set[int]):
        si = self._n2i[orig]
        for dest in dest_set:
            ti = self._n2i.get(dest)
            if ti is None:
                continue

            # Always compute unpenalized (true shortest) path for C6 baseline
            dv_unp = float(dist_arr_unp[ti])
            if not math.isfinite(dv_unp) or dv_unp < 0:
                continue
            idx_unp = self._extract(si, ti, pred_arr_unp)
            if idx_unp is None:
                continue
            pn_unp = [self._nl[idx] for idx in idx_unp]
            unp_miles = sum(
                self._len_lookup.get((pn_unp[j], pn_unp[j+1]), 0.0)
                for j in range(len(pn_unp)-1)
            )

            # Try penalized path; accept if it satisfies C6
            chosen_pn = pn_unp
            chosen_miles = unp_miles
            dv_pen = float(dist_arr_pen[ti])
            if math.isfinite(dv_pen) and dv_pen >= 0:
                idx_pen = self._extract(si, ti, pred_arr_pen)
                if idx_pen is not None:
                    pn_pen = [self._nl[idx] for idx in idx_pen]
                    pen_miles = sum(
                        self._len_lookup.get((pn_pen[j], pn_pen[j+1]), 0.0)
                        for j in range(len(pn_pen)-1)
                    )
                    # Use penalized path only if it doesn't exceed C6 ratio
                    if unp_miles <= 0 or pen_miles <= self.MAX_CIRCUITOUS * unp_miles:
                        chosen_pn = pn_pen
                        chosen_miles = pen_miles

            self.od_dist[(orig, dest)] = chosen_miles
            self.od_path_n[(orig, dest)] = " -> ".join(str(x) for x in chosen_pn)
            self.od_path_l[(orig, dest)] = " -> ".join(
                str(self._lid_lookup.get((chosen_pn[j], chosen_pn[j+1]), -1))
                for j in range(len(chosen_pn)-1)
            )
            self.od_interchanges[(orig, dest)] = _count_interchanges(chosen_pn, self.rr_map)

    def _extract(self, si: int, ti: int, pred_arr) -> list[int] | None:
        """Return list of node *indices* (not node IDs)."""
        if si == ti:
            return [si]
        path, cur, seen = [], ti, set()
        while cur != si:
            if cur < 0 or cur in seen:
                return None
            seen.add(cur); path.append(cur); cur = pred_arr[cur]
        path.append(si); path.reverse()
        return path

    # ── Accessors ─────────────────────────────────────────────────────────────

    def dist(self, o: int, d: int) -> float:
        return self.od_dist.get((o, d), math.inf)

    def interchanges(self, o: int, d: int) -> int:
        return self.od_interchanges.get((o, d), 0)

    def has_path(self, o: int, d: int) -> bool:
        return (o, d) in self.od_dist


# ─────────────────────────────────────────────────────────────────────────────
# Domain objects
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Demand:
    idx:            int
    demand_id:      int
    commodity_type: str   # original type
    block_type:     str   # block category
    origin:         int
    dest:           int
    volume:         int
    sp_dist:        float   # shortest-path miles (for unserved car-miles)


Route = Optional[list[tuple[int, int]]]   # list of (from_yard, to_yard) segments


class Solution:
    __slots__ = ("demands", "routings")

    def __init__(self, demands: list[Demand], routings: list[Route]):
        self.demands  = demands
        self.routings = routings

    def copy(self) -> "Solution":
        return Solution(self.demands, [r[:] if r else None for r in self.routings])

    def block_volumes(self) -> dict[tuple[int, int, str], float]:
        bv: dict[tuple[int, int, str], float] = defaultdict(float)
        for dem, route in zip(self.demands, self.routings):
            if route:
                for seg in route:
                    bv[(seg[0], seg[1], dem.commodity_type)] += dem.volume
        return bv


# ─────────────────────────────────────────────────────────────────────────────
# Cost / Objective
# ─────────────────────────────────────────────────────────────────────────────

def _min_vol_threshold(dist: float, s: dict) -> float:
    # +5 buffer: OD-matrix vs Dijkstra can differ by up to one tier (5 cars)
    if dist < 100:
        return float(s.get("min_block_vol_short(<100mi)", 5))   + 5
    if dist <= 500:
        return float(s.get("min_block_vol_med(100-500mi)", 10)) + 5
    return float(s.get("min_block_vol_long(>500mi)", 15))        + 5


def evaluate(
    sol:         Solution,
    net:         Network,
    od_matrix:   dict[tuple[int, int], float],   # from OD CSV
    settings:    dict,
    yard_info:   dict,
) -> tuple[float, dict]:
    """Full objective matching fast_validator_v1_1 cost model."""
    block_fixed = float(settings.get("block_fixed_cost",           1500.0))
    tc          = float(settings.get("transport_cost_coefficient",  1.0))
    ic_rate     = float(settings.get("interchange_cost",           100.0))
    M           = float(settings.get("stress_penalty_M",           5.0))

    bv = sol.block_volumes()

    # Prefer OD-matrix distance for C4 (same as validator)
    def c4_dist(o, d):
        v = od_matrix.get((o, d))
        if v is not None and v > 0:
            return v
        return net.dist(o, d)

    # Feasible blocks (C4: min volume, with buffer)
    feasible: set[tuple[int, int, str]] = set()
    for (o, d, ct), vol in bv.items():
        dist = c4_dist(o, d)
        if math.isfinite(dist) and vol >= _min_vol_threshold(dist, settings):
            feasible.add((o, d, ct))

    # Fixed
    fixed_cost = len(feasible) * block_fixed

    # Transport
    transport_cost = sum(
        bv[(o, d, ct)] * net.dist(o, d) * tc
        for (o, d, ct) in feasible
    )

    # Handling
    handling_vol: dict[int, float] = defaultdict(float)
    for dem, route in zip(sol.demands, sol.routings):
        if not route or len(route) <= 1:
            continue
        for seg in route[:-1]:
            k = (seg[0], seg[1], dem.commodity_type)
            if k in feasible:
                handling_vol[seg[1]] += dem.volume
    handling_cost = sum(
        vol * yard_info.get(hub, {}).get("handling_cost", 0.0)
        for hub, vol in handling_vol.items()
    )

    # Interchange (exact: count per block from physical paths)
    interchange_cost = sum(
        net.interchanges(o, d) * bv[(o, d, ct)] * ic_rate
        for (o, d, ct) in feasible
    )

    total_cost = fixed_cost + transport_cost + handling_cost + interchange_cost

    # Served / unserved
    unserved_cm = 0.0
    for dem, route in zip(sol.demands, sol.routings):
        if not route:
            if math.isfinite(dem.sp_dist):
                unserved_cm += dem.volume * dem.sp_dist
            continue
        if not all((s[0], s[1], dem.commodity_type) in feasible for s in route):
            if math.isfinite(dem.sp_dist):
                unserved_cm += dem.volume * dem.sp_dist

    # C2 penalty
    track_usage: dict[int, int] = defaultdict(int)
    for (o, d, ct) in feasible:
        if COMMODITY_TO_BLOCK_TYPE.get(ct, ct) in CLASSIFICATION:
            track_usage[o] += 1
    c2_pen = sum(
        LARGE_PENALTY * max(0, u - int(yard_info.get(y, {}).get("num_tracks", 9999)))
        for y, u in track_usage.items()
    )

    # C3 penalty
    c3_pen = sum(
        LARGE_PENALTY * max(0.0, v - yard_info.get(h, {}).get("handling_capacity", 1e9))
        for h, v in handling_vol.items()
    )

    obj = total_cost + M * unserved_cm + c2_pen + c3_pen

    return obj, {
        "objective":    obj,
        "fixed":        fixed_cost,
        "transport":    transport_cost,
        "handling":     handling_cost,
        "interchange":  interchange_cost,
        "total_cost":   total_cost,
        "unserved_cm":  unserved_cm,
        "stress_score": total_cost + M * unserved_cm,
        "n_blocks":     len(feasible),
        "c2_penalty":   c2_pen,
        "c3_penalty":   c3_pen,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Greedy Constructor
# ─────────────────────────────────────────────────────────────────────────────

def greedy_construct(
    demands:   list[Demand],
    net:       Network,
    od_matrix: dict[tuple[int, int], float],
    settings:  dict,
    yard_info: dict,
) -> Solution:
    def c4_dist(o, d):
        v = od_matrix.get((o, d))
        return v if (v and v > 0) else net.dist(o, d)

    groups: dict[tuple[int, int, str], list[int]] = defaultdict(list)
    for i, dem in enumerate(demands):
        groups[(dem.origin, dem.dest, dem.commodity_type)].append(i)

    sorted_groups = sorted(
        groups.items(),
        key=lambda x: sum(demands[i].volume for i in x[1]),
        reverse=True,
    )

    track_budget = {
        yid: int(info.get("num_tracks", 9999))
        for yid, info in yard_info.items()
    }
    routings: list[Route] = [None] * len(demands)

    for (orig, dest, ctype), indices in sorted_groups:
        total_vol = sum(demands[i].volume for i in indices)
        if total_vol <= 0 or not net.has_path(orig, dest):
            continue
        dist = c4_dist(orig, dest)
        if total_vol < _min_vol_threshold(dist, settings):
            continue

        bt = COMMODITY_TO_BLOCK_TYPE.get(ctype, "Manifest")
        if bt in CLASSIFICATION:
            if track_budget.get(orig, 9999) <= 0:
                continue
            track_budget[orig] = track_budget.get(orig, 9999) - 1

        for i in indices:
            if demands[i].volume > 0:
                routings[i] = [(orig, dest)]

    return Solution(demands, routings)


# ─────────────────────────────────────────────────────────────────────────────
# Neighbourhood Moves
# ─────────────────────────────────────────────────────────────────────────────

def _feasible_hubs(dem: Demand, net: Network, all_yards: list[int]) -> list[int]:
    if dem.commodity_type in DIRECT_ONLY:
        return []
    return [
        h for h in all_yards
        if h != dem.origin and h != dem.dest
        and net.has_path(dem.origin, h) and net.has_path(h, dem.dest)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Simulated Annealing
# ─────────────────────────────────────────────────────────────────────────────

def simulated_annealing(
    sol:        Solution,
    net:        Network,
    od_matrix:  dict,
    settings:   dict,
    yard_info:  dict,
    hubs:       list[int],
    max_iter:   int   = 80_000,
    T_init:     float = None,
    T_min:      float = 500.0,
    alpha:      float = None,   # if None, computed dynamically
    verbose:    bool  = True,
    log_every:  int   = 10_000,
) -> tuple[Solution, float, dict]:

    obj, stats = evaluate(sol, net, od_matrix, settings, yard_info)
    if T_init is None:
        # Low initial T: typical |Δ| ≈ 50k, want ~30% acceptance at start
        T_init = max(obj * 0.0001, 1_000)

    # Dynamic alpha: T reaches T_min exactly at max_iter
    if alpha is None:
        alpha = (T_min / T_init) ** (1.0 / max(max_iter, 1))

    best_sol, best_obj, best_stats = sol.copy(), obj, stats
    T   = T_init
    n   = len(sol.demands)
    rng = random.Random()

    if verbose:
        print(f"    SA  T0={T_init:.2e}  stress={stats['stress_score']:,.0f}  blocks={stats['n_blocks']}")

    for it in range(1, max_iter + 1):
        T = max(T * alpha, T_min)

        i   = rng.randrange(n)
        dem = sol.demands[i]
        if dem.volume <= 0:
            continue
        cur = sol.routings[i]
        mv  = rng.random()

        new_r = sol.routings[:]

        if mv < 0.30:                        # MAKE_HUB
            if dem.commodity_type in DIRECT_ONLY:
                continue
            h_list = _feasible_hubs(dem, net, hubs)
            if not h_list:
                continue
            h = rng.choice(h_list)
            new_r[i] = [(dem.origin, h), (h, dem.dest)]

        elif mv < 0.60:                      # MAKE_DIRECT
            if not net.has_path(dem.origin, dem.dest):
                continue
            new_r[i] = [(dem.origin, dem.dest)]

        elif mv < 0.78:                      # CHANGE_HUB
            if not cur or len(cur) == 1 or dem.commodity_type in DIRECT_ONLY:
                continue
            h_list = _feasible_hubs(dem, net, hubs)
            old_h  = cur[0][1]
            others = [h for h in h_list if h != old_h]
            if not others:
                continue
            h = rng.choice(others)
            new_r[i] = [(dem.origin, h), (h, dem.dest)]

        else:                                # TOGGLE serve ↔ unserve
            if cur is None:
                if net.has_path(dem.origin, dem.dest):
                    new_r[i] = [(dem.origin, dem.dest)]
                else:
                    continue
            else:
                new_r[i] = None

        cand = Solution(sol.demands, new_r)
        new_obj, new_stats = evaluate(cand, net, od_matrix, settings, yard_info)
        delta = new_obj - obj

        if delta < 0 or rng.random() < math.exp(min(0, -delta / T)):
            sol = cand
            obj = new_obj
            if obj < best_obj:
                best_obj   = obj
                best_sol   = sol.copy()
                best_stats = new_stats

        if verbose and it % log_every == 0:
            print(f"    it={it:>7}  T={T:.2e}  best_stress={best_stats['stress_score']:,.0f}"
                  f"  blocks={best_stats['n_blocks']}"
                  f"  ic={best_stats['interchange']:,.0f}"
                  f"  c2={best_stats['c2_penalty']:.0f}")

    return best_sol, best_obj, best_stats


# ─────────────────────────────────────────────────────────────────────────────
# ILS  (Iterated Local Search)
# ─────────────────────────────────────────────────────────────────────────────

def _perturb(sol: Solution, net: Network, hubs: list[int], k: int) -> Solution:
    new_sol = sol.copy()
    n       = len(new_sol.demands)
    for _ in range(k):
        i   = random.randrange(n)
        dem = new_sol.demands[i]
        mv  = random.random()
        if mv < 0.4 and dem.commodity_type not in DIRECT_ONLY:
            h_list = _feasible_hubs(dem, net, hubs)
            if h_list:
                h = random.choice(h_list)
                new_sol.routings[i] = [(dem.origin, h), (h, dem.dest)]
        elif mv < 0.75 and net.has_path(dem.origin, dem.dest):
            new_sol.routings[i] = [(dem.origin, dem.dest)]
        else:
            new_sol.routings[i] = None
    return new_sol


def iterated_local_search(
    init_sol:   Solution,
    net:        Network,
    od_matrix:  dict,
    settings:   dict,
    yard_info:  dict,
    hubs:       list[int],
    n_restarts: int   = 4,
    sa_iters:   int   = 60_000,
    T_init:     float = None,
    alpha:      float = None,   # None → computed dynamically per SA run
    verbose:    bool  = True,
) -> tuple[Solution, float, dict]:

    gb_sol, gb_obj, gb_stats = simulated_annealing(
        init_sol, net, od_matrix, settings, yard_info, hubs,
        max_iter=sa_iters, T_init=T_init, alpha=alpha, verbose=verbose,
    )
    k = max(5, len(init_sol.demands) // 30)

    for restart in range(1, n_restarts):
        if verbose:
            print(f"\n  [ILS restart {restart+1}/{n_restarts+1}]")
        curr     = _perturb(gb_sol, net, hubs, k=k)
        s, o, st = simulated_annealing(
            curr, net, od_matrix, settings, yard_info, hubs,
            max_iter=sa_iters, T_init=T_init, alpha=alpha, verbose=verbose,
        )
        if o < gb_obj:
            gb_sol, gb_obj, gb_stats = s, o, st

    return gb_sol, gb_obj, gb_stats


# ─────────────────────────────────────────────────────────────────────────────
# JSON Assembly
# ─────────────────────────────────────────────────────────────────────────────

def _to_py(v):
    if isinstance(v, (np.integer,)):  return int(v)
    if isinstance(v, (np.floating,)): return None if math.isnan(float(v)) else float(v)
    if isinstance(v, float) and math.isnan(v): return None
    return v


def df_records(df: pd.DataFrame) -> list[dict]:
    return [{k: _to_py(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def build_json(
    sol:         Solution,
    net:         Network,
    od_matrix:   dict,
    settings:    dict,
    nodes_df:    pd.DataFrame,
    links_df:    pd.DataFrame,
    demands_raw: pd.DataFrame,
) -> dict:

    def c4_dist(o, d):
        v = od_matrix.get((o, d))
        return v if (v and v > 0) else net.dist(o, d)

    # Two-pass: volumes from ALL routings may differ from what sequences produce
    # after filtering infeasible blocks. Iterate to convergence.
    routings = sol.routings
    for _ in range(5):  # converges in ≤2 passes in practice
        bv: dict[tuple, float] = defaultdict(float)
        for dem, route in zip(sol.demands, routings):
            if route:
                for seg in route:
                    bv[(seg[0], seg[1], dem.commodity_type)] += dem.volume

        feasible = {
            (o, d, ct)
            for (o, d, ct), vol in bv.items()
            if math.isfinite(c4_dist(o, d)) and vol >= _min_vol_threshold(c4_dist(o, d), settings)
        }

        # Only keep routings where every segment is feasible
        new_routings = [
            r if (r and all((s[0], s[1], d.commodity_type) in feasible for s in r)) else None
            for d, r in zip(sol.demands, routings)
        ]
        if new_routings == routings:
            break
        routings = new_routings

    # Assign block IDs to feasible blocks
    block_id_map: dict[tuple[int, int, str], int] = {}
    block_list, route_list = [], []

    for bid, (o, d, ct) in enumerate(sorted(feasible), start=1):
        block_id_map[(o, d, ct)] = bid
        bt = COMMODITY_TO_BLOCK_TYPE.get(ct, ct)
        block_list.append({
            "block_id":     bid, "from_yard_id": o, "to_yard_id": d,
            "commodity_type": bt, "block_type": bt,
            "block_volume": int(bv[(o, d, ct)]),
        })
        route_list.append({
            "block_id": bid, "from_yard_id": o, "to_yard_id": d,
            "physical_path_nodes": net.od_path_n.get((o, d), ""),
            "physical_path_links": net.od_path_l.get((o, d), ""),
        })

    seq_list = []
    for dem, route in zip(sol.demands, routings):
        if not route or dem.volume <= 0:   # skip zero-volume (v1.1 rule)
            continue
        bids = [block_id_map[(s[0], s[1], dem.commodity_type)] for s in route]
        seq_list.append({
            "commodity_id":      dem.demand_id,
            "commodity_type":    dem.commodity_type,
            "origin_yard_id":    dem.origin,
            "dest_yard_id":      dem.dest,
            "volume":            dem.volume,
            "blocking_sequence": " -> ".join(str(b) for b in bids),
        })

    demand_records = [
        {
            "commodity_id":   int(r["demand_id"]),
            "commodity_type": str(r["block_type"]),
            "origin_yard_id": int(r["origin_yard_id"]),
            "dest_yard_id":   int(r["dest_yard_id"]),
            "volume":         int(r["volume"]),
        }
        for _, r in demands_raw.iterrows()
    ]

    settings_out = {
        k: (float(v) if isinstance(v, (int, float, np.number)) else v)
        for k, v in settings.items()
    }

    return {
        "inputs": {
            "settings": settings_out,
            "nodes":    df_records(nodes_df),
            "links":    df_records(links_df),
            "demands":  demand_records,
        },
        "outputs": {
            "1 Block Design":      block_list,
            "2 Blocking Sequence": seq_list,
            "3 Block Route":       route_list,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Top-level Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def solve(
    layer:             str,
    demand_multiplier: float = 1.0,
    sa_iters:          int   = 80_000,
    n_ils_restarts:    int   = 4,
    verbose:           bool  = True,
) -> dict:
    print(f"\n{'='*64}")
    print(f"  layer={layer}  x{demand_multiplier}")
    print(f"{'='*64}")
    t0 = time.time()

    nodes_df, links_df, demands_scaled, demands_raw, settings = \
        load_layer(layer, demand_multiplier)

    yard_rows = nodes_df[nodes_df["node_type"] == "yard"]
    yard_info: dict[int, dict] = {
        int(r["node_id"]): {
            "num_tracks":        float(r.get("num_tracks",        9999) or 9999),
            "handling_capacity": float(r.get("handling_capacity", 1e9)  or 1e9),
            "handling_cost":     float(r.get("handling_cost",     0)    or 0),
            "railroad_id":       str(r.get("railroad_id", "")),
        }
        for _, r in yard_rows.iterrows()
    }

    origin_ids   = set(demands_scaled["origin_yard_id"].astype(int))
    dest_ids     = set(demands_scaled["dest_yard_id"].astype(int))
    all_yard_ids = sorted(origin_ids | dest_ids)

    # OD matrix (for accurate C4/C6 distances)
    all_od_pairs = {
        (o, d)
        for o in all_yard_ids for d in all_yard_ids
        if o != d
    }
    od_matrix = load_od_matrix(all_od_pairs)
    if verbose:
        print(f"  OD matrix: {len(od_matrix)} pairs loaded")

    # Build network (IC-penalised Dijkstra)
    net = Network(nodes_df, links_df, origin_ids, dest_ids, settings, verbose)

    # Demand objects
    demands: list[Demand] = [
        Demand(
            idx            = idx,
            demand_id      = int(row["demand_id"]),
            commodity_type = str(row["block_type"]),
            block_type     = COMMODITY_TO_BLOCK_TYPE.get(str(row["block_type"]), "Manifest"),
            origin         = int(row["origin_yard_id"]),
            dest           = int(row["dest_yard_id"]),
            volume         = int(row["volume"]),
            sp_dist        = od_matrix.get(
                (int(row["origin_yard_id"]), int(row["dest_yard_id"])),
                net.dist(int(row["origin_yard_id"]), int(row["dest_yard_id"]))
            ),
        )
        for idx, (_, row) in enumerate(demands_scaled.iterrows())
    ]

    # Phase 1: Greedy
    print(f"\n  [1/2] Greedy")
    init_sol      = greedy_construct(demands, net, od_matrix, settings, yard_info)
    g_obj, g_stat = evaluate(init_sol, net, od_matrix, settings, yard_info)
    print(f"  Greedy  stress={g_stat['stress_score']:,.0f}"
          f"  blocks={g_stat['n_blocks']}"
          f"  ic={g_stat['interchange']:,.0f}")

    # Phase 2: ILS + SA
    print(f"\n  [2/2] ILS+SA  ({n_ils_restarts+1} runs x {sa_iters:,})")
    best_sol, best_obj, best_stat = iterated_local_search(
        init_sol, net, od_matrix, settings, yard_info, all_yard_ids,
        n_restarts=n_ils_restarts,
        sa_iters=sa_iters,
        T_init=g_obj * 0.02,
        verbose=verbose,
    )

    print(f"\n  [DONE] stress={best_stat['stress_score']:,.0f}"
          f"  blocks={best_stat['n_blocks']}"
          f"  ic={best_stat['interchange']:,.0f}"
          f"  elapsed={time.time()-t0:.1f}s")

    return build_json(best_sol, net, od_matrix, settings, nodes_df, links_df, demands_raw)
