# gurobi_l1.jl — JuMP + Gurobi optimal solution for L1
# Used as training target / benchmark for RL policy

include("../src/env.jl")

using JuMP, Gurobi, Printf

const BASE      = joinpath(@__DIR__, "..", "..", "ras_release_v1.1", "ras_release_v1.1")
const SCORE_DIR = joinpath(@__DIR__, "..", "..", "ras_release_v1.1", "ras_release_v1.1", "scoring", "solutions")

function gurobi_solve(layer::String, mult::Float64; timelimit=300.0, gap=0.005)
    env_ras = make_env(layer, mult, BASE)
    s = env_ras.settings
    demands = env_ras.demands
    yards   = env_ras.all_yard_ids

    # enumerate paths per demand: direct + 1-hop hubs
    paths_per_dem = Vector{Vector{Vector{Int}}}(undef, length(demands))
    for (i, dem) in enumerate(demands)
        ps = Vector{Int}[]
        # direct
        !isinf(get(env_ras.od_dist_pen, (dem.origin, dem.dest), Inf)) &&
            push!(ps, [dem.origin, dem.dest])
        # via hub
        if !dem.direct_only
            for hub in yards
                (hub == dem.origin || hub == dem.dest) && continue
                d1 = get(env_ras.od_dist_pen, (dem.origin, hub), Inf)
                d2 = get(env_ras.od_dist_pen, (hub, dem.dest),   Inf)
                (!isinf(d1) && !isinf(d2)) && push!(ps, [dem.origin, hub, dem.dest])
            end
        end
        paths_per_dem[i] = ps
    end

    # block catalogue: (orig, dest, block_type)
    block_set = Set{Tuple{Int,Int,String}}()
    for (i, dem) in enumerate(demands)
        for path in paths_per_dem[i]
            segs = length(path) == 2 ? [(path[1],path[2])] : [(path[1],path[2]),(path[2],path[3])]
            for (o, d) in segs; push!(block_set, (o, d, dem.block_type)); end
        end
    end
    blocks = sort(collect(block_set))
    b2i    = Dict(b => i for (i, b) in enumerate(blocks))
    total_vol = sum(d.volume for d in demands if d.volume > 0)

    n_dem = length(demands)
    n_blk = length(blocks)
    @printf("  Demands=%d  Blocks=%d  Paths=%d\n",
            n_dem, n_blk, sum(length.(paths_per_dem)))

    model = Model(Gurobi.Optimizer)
    set_optimizer_attribute(model, "TimeLimit",  timelimit)
    set_optimizer_attribute(model, "MIPGap",     gap)
    set_optimizer_attribute(model, "MIPFocus",   1)
    set_optimizer_attribute(model, "OutputFlag", 1)

    # variables
    y  = [@variable(model, [1:length(paths_per_dem[i])], Bin) for i in 1:n_dem]
    u  = @variable(model, [1:n_dem], Bin)
    x  = @variable(model, [1:n_blk], Bin)
    vb = @variable(model, [1:n_blk] .>= 0)

    # C1: assign
    for i in 1:n_dem
        if demands[i].volume <= 0 || isempty(paths_per_dem[i])
            @constraint(model, u[i] == 1)
        else
            @constraint(model, sum(y[i]) + u[i] == 1)
        end
    end

    # block volumes
    block_flows = [Tuple{Int,Int,Float64}[] for _ in 1:n_blk]
    for (i, dem) in enumerate(demands)
        for (pi, path) in enumerate(paths_per_dem[i])
            segs = length(path) == 2 ? [(path[1],path[2])] : [(path[1],path[2]),(path[2],path[3])]
            for (o, d) in segs
                bi = b2i[(o, d, dem.block_type)]
                push!(block_flows[bi], (i, pi, dem.volume))
            end
        end
    end
    for bi in 1:n_blk
        if isempty(block_flows[bi])
            @constraint(model, vb[bi] == 0)
        else
            @constraint(model, vb[bi] == sum(v * y[k][p] for (k,p,v) in block_flows[bi]))
        end
    end

    # C4 + open coupling
    for (bi, blk) in enumerate(blocks)
        dist = get(env_ras.od_dist_pen, (blk[1], blk[2]), Inf)
        mv   = isinf(dist) ? 1e9 : min_vol_threshold(dist, s)
        @constraint(model, vb[bi] >= mv * x[bi])
        @constraint(model, vb[bi] <= total_vol * x[bi])
    end

    # C2: track limits
    for yard in yards
        blk_idxs = [bi for (bi, (o,d,bt)) in enumerate(blocks)
                    if o == yard && bt in CLASSIFICATION_TYPES]
        isempty(blk_idxs) && continue
        tracks = haskey(env_ras.yards, yard) ? env_ras.yards[yard].num_tracks : 9999
        tracks < 9999 && @constraint(model, sum(x[bi] for bi in blk_idxs) <= tracks)
    end

    # C3: handling capacity
    for yard in yards
        items = [(i, pi, dem.volume)
                 for (i, dem) in enumerate(demands)
                 for (pi, path) in enumerate(paths_per_dem[i])
                 if length(path) == 3 && path[2] == yard]
        isempty(items) && continue
        cap = haskey(env_ras.yards, yard) ? env_ras.yards[yard].handling_capacity : 1e9
        cap < 1e8 && @constraint(model, sum(v * y[k][p] for (k,p,v) in items) <= cap)
    end

    # Objective
    fixed_expr    = s.block_fixed * sum(x)
    trans_ic_expr = sum(
        (get(env_ras.od_dist_pen, (o,d), 0.0) * s.transport_coef +
         get(env_ras.od_ic,       (o,d), 0)   * s.interchange_cost) * vb[bi]
        for (bi, (o,d,bt)) in enumerate(blocks)
        if !isinf(get(env_ras.od_dist_pen, (o,d), Inf))
    )
    hdl_expr = sum(
        env_ras.yards[path[2]].handling_cost * dem.volume * y[i][pi]
        for (i, dem) in enumerate(demands)
        for (pi, path) in enumerate(paths_per_dem[i])
        if length(path) == 3 && haskey(env_ras.yards, path[2])
    )
    unserved_expr = sum(
        s.penalty_M * dem.volume * get(env_ras.od_dist_unp, (dem.origin, dem.dest), 0.0) * u[i]
        for (i, dem) in enumerate(demands)
        if dem.volume > 0
    )

    @objective(model, Min, fixed_expr + trans_ic_expr + hdl_expr + unserved_expr)

    t0 = time()
    optimize!(model)

    status = termination_status(model)
    @printf("  Status: %s  gap=%.3f%%  time=%.1fs\n", status, relative_gap(model)*100, time()-t0)
    println("  Obj: $(round(Int, objective_value(model)))")

    # extract routings
    for (i, dem) in enumerate(demands)
        route = nothing
        for (pi, path) in enumerate(paths_per_dem[i])
            value(y[i][pi]) > 0.5 && (route = path; break)
        end
        env_ras.routings[i] = route
    end

    score = evaluate(env_ras)
    println("  Eval score: $(round(Int, score))")
    env_ras, score
end

# Run all L1 scenarios
for mult in [0.5, 1.0, 2.0]
    println("\n" * "="^60)
    println("Gurobi L1 × $mult")
    println("="^60)
    env_ras, score = gurobi_solve("l1", mult; timelimit=300.0, gap=0.005)
    println("  Final stress: $(round(Int, score))")
end
