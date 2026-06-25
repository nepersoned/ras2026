# policy.jl — Flux policy network (batched)

using Flux, Statistics, Random

const BLOCK_TYPES  = ["Manifest", "Bulk", "Intermodal", "Multilevel"]
const MAX_HUBS     = 50
const FEAT_DIM     = 7 + MAX_HUBS * 4   # 207
const N_ACTIONS    = MAX_HUBS + 2        # direct + hubs + unserved

# ── Feature vector for one demand ────────────────────────────────────────────
function block_type_onehot(bt::String)::Vector{Float32}
    v = zeros(Float32, 4)
    idx = findfirst(==(bt), BLOCK_TYPES)
    idx !== nothing && (v[idx] = 1.0f0)
    v
end

function demand_features(dem, hub_list::Vector{Int}, env)::Vector{Float32}
    od_d  = get(env.od_dist_pen, (dem.origin, dem.dest), 1e6)
    od_ic = get(env.od_ic,       (dem.origin, dem.dest), 0)
    base  = Float32[log1p(dem.volume), log1p(od_d), Float32(od_ic)/5f0,
                    block_type_onehot(dem.block_type)...]
    hub_feats = zeros(Float32, MAX_HUBS * 4)
    for (j, hub) in enumerate(hub_list[1:min(length(hub_list), MAX_HUBS)])
        d1  = get(env.od_dist_pen, (dem.origin, hub), Inf)
        d2  = get(env.od_dist_pen, (hub, dem.dest),   Inf)
        (isinf(d1) || isinf(d2)) && continue
        ic1 = get(env.od_ic, (dem.origin, hub), 0)
        ic2 = get(env.od_ic, (hub, dem.dest),   0)
        hc  = haskey(env.yards, hub) ? env.yards[hub].handling_cost : 0.0
        hub_feats[(j-1)*4+1] = log1p(d1)
        hub_feats[(j-1)*4+2] = log1p(d2)
        hub_feats[(j-1)*4+3] = Float32(ic1+ic2) / 5f0
        hub_feats[(j-1)*4+4] = Float32(hc) / 500f0
    end
    vcat(base, hub_feats)
end

function ranked_hubs(dem, env)::Vector{Int}
    dem.direct_only && return Int[]
    scored = Tuple{Float64,Int}[]
    for hub in env.all_yard_ids
        (hub == dem.origin || hub == dem.dest) && continue
        d1 = get(env.od_dist_pen, (dem.origin, hub), Inf)
        d2 = get(env.od_dist_pen, (hub, dem.dest),   Inf)
        (isinf(d1) || isinf(d2)) && continue
        push!(scored, (d1+d2, hub))
    end
    sort!(scored)
    [h for (_,h) in scored]
end

function action_mask(dem, hubs::Vector{Int}, env)::Vector{Bool}
    mask = falses(N_ACTIONS)
    !isinf(get(env.od_dist_pen, (dem.origin, dem.dest), Inf)) && (mask[1] = true)
    if !dem.direct_only
        for (j, hub) in enumerate(hubs[1:min(length(hubs), MAX_HUBS)])
            d1 = get(env.od_dist_pen, (dem.origin, hub), Inf)
            d2 = get(env.od_dist_pen, (hub, dem.dest),   Inf)
            (!isinf(d1) && !isinf(d2)) && (mask[j+1] = true)
        end
    end
    mask[N_ACTIONS] = true
    mask
end

function action_to_route(a::Int, dem, hubs::Vector{Int})::Route
    a == 1          && return [dem.origin, dem.dest]
    a == N_ACTIONS  && return nothing
    hi = a - 1
    hi > length(hubs) && return nothing
    [dem.origin, hubs[hi], dem.dest]
end

# ── Precomputed per-demand data ───────────────────────────────────────────────
struct DemandData
    feat ::Vector{Float32}
    mask ::Vector{Bool}
    hubs ::Vector{Int}
    valid::Bool
end

function precompute(env::RASEnv)::Vector{DemandData}
    [begin
        if dem.volume <= 0
            DemandData(zeros(Float32, FEAT_DIM), trues(N_ACTIONS), Int[], false)
        else
            hubs = ranked_hubs(dem, env)
            DemandData(demand_features(dem, hubs, env),
                       action_mask(dem, hubs, env), hubs, true)
        end
    end for dem in env.demands]
end

# ── Policy network ────────────────────────────────────────────────────────────
function make_policy()
    Chain(Dense(FEAT_DIM, 256, relu),
          Dense(256, 256, relu),
          Dense(256, N_ACTIONS))
end

# ── Batched forward: FEAT_DIM×N → N_ACTIONS×N ────────────────────────────────
function batch_logits(policy, dd::Vector{DemandData})
    valid_idxs = [i for (i,d) in enumerate(dd) if d.valid]
    isempty(valid_idxs) && return Matrix{Float32}(undef,0,0), valid_idxs
    feat_mat = hcat([dd[i].feat for i in valid_idxs]...)  # FEAT_DIM × N
    policy(feat_mat), valid_idxs                           # N_ACTIONS × N
end

# ── Masked softmax (single vector, no mutation) ───────────────────────────────
function masked_softmax(logits::AbstractVector{Float32}, mask::Vector{Bool})
    m = ifelse.(mask, logits, fill(-1f8, length(logits)))
    e = exp.(m .- maximum(m))
    e ./ sum(e)
end
