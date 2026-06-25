# env.jl — RAS 2026 MDP Environment
# Handles data loading, cost evaluation, and LNS step

using CSV, DataFrames, SparseArrays, Statistics, Random

# ── Constants ─────────────────────────────────────────────────────────────────
const DIRECT_ONLY = Set(["Intermodal", "Multilevel"])
const COMMODITY_TO_BLOCK_TYPE = Dict(
    "merchandise" => "Manifest",
    "coal"        => "Bulk",
    "grain"       => "Bulk",
    "intermodal"  => "Intermodal",
    "automobile"  => "Multilevel",
    "Merchandise" => "Manifest",
    "Coal"        => "Bulk",
    "Grain"       => "Bulk",
    "Intermodal"  => "Intermodal",
    "Automobile"  => "Multilevel",
)
const CLASSIFICATION_TYPES = Set(["Manifest", "Bulk"])  # C2 applies

# ── Data Structures ───────────────────────────────────────────────────────────
struct Settings
    min_vol_short::Float64   # <100 mi
    min_vol_med::Float64     # 100-500 mi
    min_vol_long::Float64    # >500 mi
    max_circuitous::Float64
    block_fixed::Float64
    transport_coef::Float64
    interchange_cost::Float64
    penalty_M::Float64
    demand_multiplier::Float64
end

struct YardInfo
    node_id::Int
    num_tracks::Int
    handling_capacity::Float64
    handling_cost::Float64
    railroad_id::String
end

struct Demand
    id::Int
    origin::Int
    dest::Int
    volume::Float64          # already multiplied
    block_type::String       # Manifest/Bulk/Intermodal/Multilevel
    direct_only::Bool
end

# Route: nothing=unserved, [o,d]=direct, [o,h,d]=via hub
const Route = Union{Nothing, Vector{Int}}

mutable struct RASEnv
    layer::String
    settings::Settings
    demands::Vector{Demand}
    yards::Dict{Int, YardInfo}          # yard_id → info
    all_yard_ids::Vector{Int}           # sorted unique yard ids in this layer
    od_dist_pen::Dict{Tuple{Int,Int}, Float64}   # IC-penalized dist
    od_dist_unp::Dict{Tuple{Int,Int}, Float64}   # shortest dist
    od_ic::Dict{Tuple{Int,Int}, Int}             # interchange count
    # current solution
    routings::Vector{Route}
end

# ── Data Loading ──────────────────────────────────────────────────────────────
function load_settings(path::String, multiplier::Float64)::Settings
    df = CSV.read(path, DataFrame)
    get_val(k) = parse(Float64, string(df[df.parameter .== k, :value][1]))
    Settings(
        get_val("min_block_vol_short(<100mi)"),
        get_val("min_block_vol_med(100-500mi)"),
        get_val("min_block_vol_long(>500mi)"),
        get_val("max_circuitous_ratio"),
        get_val("block_fixed_cost"),
        get_val("transport_cost_coefficient"),
        get_val("interchange_cost"),
        get_val("stress_penalty_M"),
        multiplier,
    )
end

function load_yards(node_path::String)::Dict{Int, YardInfo}
    df = CSV.read(node_path, DataFrame)
    yards = Dict{Int, YardInfo}()
    for row in eachrow(df)
        row.node_type == "yard" || continue
        nid = Int(row.node_id)
        yards[nid] = YardInfo(
            nid,
            ismissing(row.num_tracks)        ? 9999 : Int(row.num_tracks),
            ismissing(row.handling_capacity) ? 1e9  : Float64(row.handling_capacity),
            ismissing(row.handling_cost)     ? 0.0  : Float64(row.handling_cost),
            ismissing(row.railroad_id)       ? ""   : string(row.railroad_id),
        )
    end
    yards
end

function load_demands(demand_path::String, multiplier::Float64)::Vector{Demand}
    df = CSV.read(demand_path, DataFrame)
    demands = Demand[]
    for row in eachrow(df)
        bt = string(row.block_type)
        vol = Float64(row.volume) * multiplier
        push!(demands, Demand(
            Int(row.demand_id),
            Int(row.origin_yard_id),
            Int(row.dest_yard_id),
            vol,
            bt,
            bt in DIRECT_ONLY,
        ))
    end
    demands
end

# ── Dijkstra on Physical Network ──────────────────────────────────────────────
function build_graph(link_path::String, node_path::String)
    links = CSV.read(link_path, DataFrame)
    nodes = CSV.read(node_path, DataFrame)

    all_node_ids = sort(unique(nodes.node_id))
    n2i = Dict(id => i for (i, id) in enumerate(all_node_ids))
    N = length(all_node_ids)

    # Build CSR-style adjacency: penalized (IC as 100-mile equiv) + unpenalized
    # railroad_id for each node
    rr_map = Dict{Int, String}()
    for row in eachrow(nodes)
        ismissing(row.railroad_id) || (rr_map[Int(row.node_id)] = string(row.railroad_id))
    end

    # edges: (from_idx, to_idx, length, penalized_length)
    from_v, to_v, pen_w, unp_w = Int[], Int[], Float64[], Float64[]
    for row in eachrow(links)
        fi = get(n2i, Int(row.from_node_id), 0)
        ti = get(n2i, Int(row.to_node_id), 0)
        (fi == 0 || ti == 0) && continue
        len = Float64(row.length)
        rr_from = get(rr_map, Int(row.from_node_id), "")
        rr_to   = get(rr_map, Int(row.to_node_id),   "")
        pen = (rr_from != "" && rr_to != "" && rr_from != rr_to) ? len + 100.0 : len
        push!(from_v, fi); push!(to_v, ti)
        push!(pen_w, pen); push!(unp_w, len)
        # undirected
        push!(from_v, ti); push!(to_v, fi)
        push!(pen_w, pen); push!(unp_w, len)
    end

    mat_pen = sparse(from_v, to_v, pen_w, N, N)
    mat_unp = sparse(from_v, to_v, unp_w, N, N)
    return mat_pen, mat_unp, n2i, all_node_ids, rr_map
end

function dijkstra_from(mat::SparseMatrixCSC{Float64,Int}, src_idx::Int)
    N = size(mat, 1)
    dist = fill(Inf, N)
    pred = fill(-1, N)
    dist[src_idx] = 0.0
    visited = falses(N)
    # priority queue via simple heap (adequate for N~50k)
    # Use a simple array-based approach
    pq = [(0.0, src_idx)]  # (dist, node_idx)

    while !isempty(pq)
        d, u = popfirst!(pq)  # not optimal but works
        visited[u] && continue
        visited[u] = true
        rows = mat.rowval[mat.colptr[u]:mat.colptr[u+1]-1]
        vals = mat.nzval[mat.colptr[u]:mat.colptr[u+1]-1]
        for k in eachindex(rows)
            v = rows[k]
            nd = d + vals[k]
            if nd < dist[v]
                dist[v] = nd
                pred[v] = u
                push!(pq, (nd, v))
            end
        end
        sort!(pq)  # keep sorted — simple O(n log n) per step
    end
    dist, pred
end

function extract_path(pred::Vector{Int}, src::Int, dst::Int)::Vector{Int}
    src == dst && return [src]
    path = Int[]
    cur = dst
    for _ in 1:100000
        cur == -1 && return Int[]
        pushfirst!(path, cur)
        cur == src && return path
        cur = pred[cur]
    end
    Int[]
end

function count_ic(path_node_ids::Vector{Int}, rr_map::Dict{Int,String})::Int
    ic = 0
    prev_rr = ""
    for nid in path_node_ids
        rr = get(rr_map, nid, "")
        if rr != "" && prev_rr != "" && rr != prev_rr
            ic += 1
        end
        rr != "" && (prev_rr = rr)
    end
    ic
end

function compute_od_tables(
    layer::String, multiplier::Float64,
    base_path::String
)
    node_path   = joinpath(base_path, "datasets", layer, "node.csv")
    link_path   = joinpath(base_path, "datasets", layer, "link.csv")
    demand_path = joinpath(base_path, "datasets", layer, "demand.csv")

    demands_raw = load_demands(demand_path, multiplier)
    origin_ids  = Set(d.origin for d in demands_raw)
    dest_ids    = Set(d.dest   for d in demands_raw)

    mat_pen, mat_unp, n2i, nl, rr_map = build_graph(link_path, node_path)

    od_dist_pen = Dict{Tuple{Int,Int}, Float64}()
    od_dist_unp = Dict{Tuple{Int,Int}, Float64}()
    od_ic       = Dict{Tuple{Int,Int}, Int}()

    max_circ = 1.28  # slightly below 1.30 for safety

    dest_idxs = [get(n2i, d, -1) for d in dest_ids]
    filter!(x -> x > 0, dest_idxs)

    println("  Dijkstra: $(length(origin_ids)) sources")
    t0 = time()
    for (cnt, oid) in enumerate(origin_ids)
        si = get(n2i, oid, -1)
        si == -1 && continue

        dist_pen, pred_pen = dijkstra_from(mat_pen, si)
        dist_unp, pred_unp = dijkstra_from(mat_unp, si)

        for did in dest_ids
            ti = get(n2i, did, -1)
            ti == -1 && continue

            d_unp = dist_unp[ti]
            d_pen = dist_pen[ti]
            isinf(d_unp) && isinf(d_pen) && continue

            # C6: use penalized path only if within max_circuitous of unpenalized
            if !isinf(d_pen) && (isinf(d_unp) || d_pen <= max_circ * d_unp)
                path_idx = extract_path(pred_pen, si, ti)
                path_nid = [nl[i] for i in path_idx]
                od_dist_pen[(oid, did)] = d_pen
                od_ic[(oid, did)]       = count_ic(path_nid, rr_map)
            else
                path_idx = extract_path(pred_unp, si, ti)
                path_nid = [nl[i] for i in path_idx]
                od_dist_pen[(oid, did)] = d_unp
                od_ic[(oid, did)]       = count_ic(path_nid, rr_map)
            end

            od_dist_unp[(oid, did)] = isinf(d_unp) ? d_pen : d_unp
        end

        cnt % 10 == 0 && print("\r  $(cnt)/$(length(origin_ids))")
    end
    println("\r  done in $(round(time()-t0, digits=1))s")

    od_dist_pen, od_dist_unp, od_ic
end

# ── Min Volume Threshold ──────────────────────────────────────────────────────
function min_vol_threshold(dist::Float64, s::Settings)::Float64
    dist < 100.0  && return s.min_vol_short
    dist <= 500.0 && return s.min_vol_med
    return s.min_vol_long
end

# ── Cost Evaluation ───────────────────────────────────────────────────────────
function evaluate(env::RASEnv)::Float64
    s = env.settings

    # block volumes: (orig, dest, block_type) → volume
    bvol = Dict{Tuple{Int,Int,String}, Float64}()
    for (dem, route) in zip(env.demands, env.routings)
        dem.volume <= 0 && continue
        route === nothing && continue
        segs = length(route) == 2 ? [(route[1], route[2])] :
                                    [(route[1], route[2]), (route[2], route[3])]
        for (o, d) in segs
            key = (o, d, dem.block_type)
            bvol[key] = get(bvol, key, 0.0) + dem.volume
        end
    end

    # C4: drop blocks below min volume
    feasible_blocks = Set{Tuple{Int,Int,String}}()
    for (key, vol) in bvol
        o, d, bt = key
        dist = get(env.od_dist_pen, (o, d), Inf)
        isinf(dist) && continue
        vol >= min_vol_threshold(dist, s) && push!(feasible_blocks, key)
    end

    # recompute served demands
    fixed_cost    = 0.0
    transport     = 0.0
    handling      = 0.0
    interchange   = 0.0
    unserved_cm   = 0.0

    seen_blocks = Set{Tuple{Int,Int,String}}()

    for (dem, route) in zip(env.demands, env.routings)
        dem.volume <= 0 && continue
        if route === nothing
            sp = get(env.od_dist_unp, (dem.origin, dem.dest), Inf)
            isinf(sp) || (unserved_cm += dem.volume * sp)
            continue
        end
        segs = length(route) == 2 ? [(route[1], route[2])] :
                                    [(route[1], route[2]), (route[2], route[3])]

        # check all segments feasible
        all_ok = all((o, d, dem.block_type) in feasible_blocks for (o, d) in segs)
        if !all_ok
            sp = get(env.od_dist_unp, (dem.origin, dem.dest), Inf)
            isinf(sp) || (unserved_cm += dem.volume * sp)
            continue
        end

        for (o, d) in segs
            key = (o, d, dem.block_type)
            if !(key in seen_blocks)
                push!(seen_blocks, key)
                fixed_cost += s.block_fixed
            end
            dist = get(env.od_dist_pen, (o, d), 0.0)
            ic   = get(env.od_ic, (o, d), 0)
            transport   += dem.volume * dist * s.transport_coef
            interchange += dem.volume * ic  * s.interchange_cost
        end

        # handling at intermediate hub
        if length(route) == 3
            hub = route[2]
            hcost = haskey(env.yards, hub) ? env.yards[hub].handling_cost : 0.0
            handling += dem.volume * hcost
        end
    end

    fixed_cost + transport + handling + interchange + s.penalty_M * unserved_cm
end

# ── Greedy Initial Solution ───────────────────────────────────────────────────
function greedy!(env::RASEnv)
    s = env.settings
    env.routings = Vector{Route}(undef, length(env.demands))

    for (i, dem) in enumerate(env.demands)
        dem.volume <= 0 && (env.routings[i] = nothing; continue)

        best_cost = Inf
        best_route = nothing

        # direct
        dist_d = get(env.od_dist_pen, (dem.origin, dem.dest), Inf)
        if !isinf(dist_d)
            ic   = get(env.od_ic, (dem.origin, dem.dest), 0)
            cost = dist_d * s.transport_coef + ic * s.interchange_cost
            if cost < best_cost
                best_cost  = cost
                best_route = [dem.origin, dem.dest]
            end
        end

        # via hub (not for direct-only commodities)
        if !dem.direct_only
            for hub in env.all_yard_ids
                (hub == dem.origin || hub == dem.dest) && continue
                d1 = get(env.od_dist_pen, (dem.origin, hub), Inf)
                d2 = get(env.od_dist_pen, (hub, dem.dest),   Inf)
                isinf(d1) || isinf(d2) && continue
                ic1 = get(env.od_ic, (dem.origin, hub), 0)
                ic2 = get(env.od_ic, (hub, dem.dest),   0)
                hcost = haskey(env.yards, hub) ? env.yards[hub].handling_cost : 0.0
                cost = (d1 + d2) * s.transport_coef +
                       (ic1 + ic2) * s.interchange_cost +
                       hcost
                if cost < best_cost
                    best_cost  = cost
                    best_route = [dem.origin, hub, dem.dest]
                end
            end
        end

        env.routings[i] = best_route
    end
end

# ── LNS Step ──────────────────────────────────────────────────────────────────
# Randomly select k demands, try all their route options, pick best delta
function lns_step!(env::RASEnv, k::Int = 10)::Float64
    before = evaluate(env)
    idxs = randperm(length(env.demands))[1:min(k, length(env.demands))]

    for i in idxs
        dem = env.demands[i]
        dem.volume <= 0 && continue

        best_cost = Inf
        best_route = nothing
        old_route  = env.routings[i]

        # try direct
        dist_d = get(env.od_dist_pen, (dem.origin, dem.dest), Inf)
        if !isinf(dist_d)
            env.routings[i] = [dem.origin, dem.dest]
            c = evaluate(env)
            if c < best_cost; best_cost = c; best_route = [dem.origin, dem.dest]; end
        end

        # try via hub
        if !dem.direct_only
            for hub in env.all_yard_ids
                (hub == dem.origin || hub == dem.dest) && continue
                d1 = get(env.od_dist_pen, (dem.origin, hub), Inf)
                d2 = get(env.od_dist_pen, (hub, dem.dest),   Inf)
                (isinf(d1) || isinf(d2)) && continue
                env.routings[i] = [dem.origin, hub, dem.dest]
                c = evaluate(env)
                if c < best_cost; best_cost = c; best_route = copy(env.routings[i]); end
            end
        end

        # try unserved
        env.routings[i] = nothing
        c = evaluate(env)
        if c < best_cost; best_cost = c; best_route = nothing; end

        env.routings[i] = best_route
    end

    evaluate(env) - before
end

# ── Environment Constructor ───────────────────────────────────────────────────
function make_env(layer::String, multiplier::Float64, base_path::String)::RASEnv
    println("\n=== Loading $layer ×$multiplier ===")
    node_path    = joinpath(base_path, "datasets", layer, "node.csv")
    demand_path  = joinpath(base_path, "datasets", layer, "demand.csv")
    setting_path = joinpath(base_path, "datasets", layer, "setting.csv")

    s       = load_settings(setting_path, multiplier)
    yards   = load_yards(node_path)
    demands = load_demands(demand_path, multiplier)

    origins = Set(d.origin for d in demands)
    dests   = Set(d.dest   for d in demands)
    all_yards = sort(collect(origins ∪ dests))

    od_pen, od_unp, od_ic = compute_od_tables(layer, multiplier, base_path)

    env = RASEnv(layer, s, demands, yards, all_yards,
                 od_pen, od_unp, od_ic,
                 Vector{Route}(undef, length(demands)))
    env
end
