"""
OR-Tools CP-SAT repair for Neural Destroy + OR-Tools Repair (NDOR).

Given K demands removed from current solution, find the optimal
re-assignment of those K demands using CP-SAT.
"""

from ortools.sat.python import cp_model
import math


def ortools_repair(destroyed_indices, dd, env, current_routings, scale=1000):
    """
    Repair K destroyed demands optimally via CP-SAT.

    Args:
        destroyed_indices: list of demand indices to re-assign
        dd: precomputed per-demand data (feat, mask, hubs, candidates)
        env: environment dict
        current_routings: current full routing list (fixed demands intact)
        scale: cost scaling for integer CP-SAT variables

    Returns:
        repaired_routings: new routings with K demands re-assigned
    """
    model  = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 2.0  # 2s per repair
    solver.parameters.num_search_workers  = 1

    demands    = env['demands']
    net        = env['net']
    settings   = env['settings']
    yard_info  = env['yard_info']
    od_matrix  = env['od_matrix']

    block_fc   = int(settings.get('block_fixed_cost', 1500))
    unserved_M = int(settings.get('stress_penalty_M', 5))
    ic_cost    = int(settings.get('interchange_cost', 100))
    tc_coeff   = float(settings.get('transport_cost_coefficient', 1.0))

    # ── Variables: one integer per destroyed demand ───────────────────────────
    # x[i] = route_idx ∈ {0 .. len(candidates[i])-1}
    x_vars = {}
    for i in destroyed_indices:
        d = dd[i]
        if d is None:
            continue
        n_cands = len(d['candidates'])
        x_vars[i] = model.NewIntVar(0, n_cands - 1, f'x_{i}')

    if not x_vars:
        return list(current_routings)

    # ── Objective: minimise sum of marginal costs ─────────────────────────────
    cost_terms = []

    for i, x in x_vars.items():
        d    = dd[i]
        dem  = demands[i]
        cands = d['candidates']  # list of (route, cost_int)

        # Build cost table: cost[route_idx] = int cost
        cost_table = []
        for ri, (route, cost_f) in enumerate(cands):
            cost_table.append(int(cost_f * scale))

        # Add element constraint: cost_var = cost_table[x]
        cost_var = model.NewIntVar(
            min(cost_table), max(cost_table), f'cost_{i}'
        )
        model.AddElement(x, cost_table, cost_var)
        cost_terms.append(cost_var)

    model.Minimize(sum(cost_terms))

    # ── Solve ─────────────────────────────────────────────────────────────────
    status = solver.Solve(model)

    repaired = list(current_routings)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i, x in x_vars.items():
            ri = solver.Value(x)
            repaired[i] = dd[i]['candidates'][ri][0]  # (route, cost) → route
    # else: keep original routes for destroyed demands

    return repaired


def build_candidates(i, dd, env):
    """
    Build (route, marginal_cost) candidate list for demand i.
    Used to precompute cost tables for OR-Tools.
    """
    d      = dd[i]
    dem    = env['demands'][i]
    net    = env['net']
    s      = env['settings']

    if d is None:
        return []

    tc_coeff   = float(s.get('transport_cost_coefficient', 1.0))
    ic_cost    = float(s.get('interchange_cost', 100))
    unserved_M = float(s.get('stress_penalty_M', 5))

    cands = []

    # Unserved
    unserved_cost = unserved_M * dem.volume * dem.sp_dist
    cands.append((None, unserved_cost))

    # Direct
    od_dist = net.dist(dem.origin, dem.dest)
    if not math.isinf(od_dist):
        od_ic  = net.interchanges(dem.origin, dem.dest)
        cost   = tc_coeff * dem.volume * od_dist + ic_cost * dem.volume * od_ic
        cands.append(([(dem.origin, dem.dest)], cost))

    # Via hubs
    if not d.get('direct_only', False):
        for hub in d['hubs']:
            d1 = net.dist(dem.origin, hub)
            d2 = net.dist(hub, dem.dest)
            if math.isinf(d1) or math.isinf(d2):
                continue
            ic1 = net.interchanges(dem.origin, hub)
            ic2 = net.interchanges(hub, dem.dest)
            hc  = env['yard_info'].get(hub, {}).get('handling_cost', 0.0)
            cost = (tc_coeff * dem.volume * (d1 + d2)
                    + ic_cost * dem.volume * (ic1 + ic2)
                    + hc * dem.volume)
            cands.append(([(dem.origin, hub), (hub, dem.dest)], cost))

    return cands
