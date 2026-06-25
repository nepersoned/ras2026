# reinforce.jl — REINFORCE (batched, Zygote-safe)

using Flux, Zygote, Statistics, Random, Printf

# ── Batched log_probs (single matrix forward → Zygote sees ONE op) ───────────
function sum_log_probs(policy, dd::Vector{DemandData}, actions::Vector{Int})::Float32
    logits_mat, valid_idxs = batch_logits(policy, dd)
    isempty(valid_idxs) && return 0f0
    total = 0f0
    for (j, i) in enumerate(valid_idxs)
        probs  = masked_softmax(logits_mat[:, j], dd[i].mask)
        total += log(probs[actions[i]] + 1f-8)
    end
    total
end

# ── Sample actions (outside gradient) ────────────────────────────────────────
function sample_actions(policy, dd::Vector{DemandData}, env::RASEnv)::Vector{Int}
    logits_mat, valid_idxs = batch_logits(policy, dd)
    actions  = fill(N_ACTIONS, length(dd))
    routings = Vector{Route}(undef, length(dd))

    vi = 1
    for i in eachindex(dd)
        if !dd[i].valid
            routings[i] = nothing
            continue
        end
        probs = masked_softmax(logits_mat[:, vi], dd[i].mask)
        vi   += 1
        r, a  = rand(Float32), N_ACTIONS
        cp    = 0f0
        for k in 1:N_ACTIONS
            cp += probs[k]
            if r <= cp; a = k; break; end
        end
        actions[i]  = a
        routings[i] = action_to_route(a, env.demands[i], dd[i].hubs)
    end
    env.routings = routings
    actions
end

# ── Greedy inference ──────────────────────────────────────────────────────────
function run_greedy!(policy, dd::Vector{DemandData}, env::RASEnv)::Float64
    logits_mat, valid_idxs = batch_logits(policy, dd)
    routings = Vector{Route}(undef, length(dd))
    vi = 1
    for i in eachindex(dd)
        if !dd[i].valid; routings[i] = nothing; continue; end
        probs       = masked_softmax(logits_mat[:, vi], dd[i].mask)
        vi         += 1
        routings[i] = action_to_route(argmax(probs), env.demands[i], dd[i].hubs)
    end
    env.routings = routings
    evaluate(env)
end

# ── REINFORCE ─────────────────────────────────────────────────────────────────
function train_reinforce!(
    policy, env::RASEnv;
    n_episodes     = 2000,
    lr             = 1f-4,
    baseline_decay = 0.99f0,
    print_every    = 100,
)
    opt_state = Flux.setup(Adam(lr), policy)
    dd        = precompute(env)

    greedy!(env)
    baseline   = Float32(evaluate(env))
    best_score = Float64(baseline)
    best_state = Flux.state(policy)

    println("\n── REINFORCE ──")
    @printf("  n=%d  lr=%.0e  init=%.0f\n", n_episodes, lr, baseline)

    for ep in 1:n_episodes
        actions   = sample_actions(policy, dd, env)
        score     = Float32(evaluate(env))
        advantage = baseline - score

        baseline = baseline_decay * baseline + (1f0 - baseline_decay) * score

        _, grads = Flux.withgradient(policy) do m
            -advantage * sum_log_probs(m, dd, actions)
        end
        Flux.update!(opt_state, policy, grads[1])

        if Float64(score) < best_score
            best_score = Float64(score)
            best_state = Flux.state(policy)
        end

        if ep % print_every == 0
            g = run_greedy!(policy, dd, env)
            @printf("  ep %4d | sample=%12.0f | greedy=%12.0f | best=%12.0f\n",
                    ep, score, g, best_score)
        end
    end

    Flux.loadmodel!(policy, best_state)
    @printf("  Best: %.0f\n", best_score)
    best_score
end

function infer!(policy, env::RASEnv)::Float64
    dd = precompute(env)
    run_greedy!(policy, dd, env)
end
