# bc_pretrain.jl — Behavioral Cloning (batched)

using Flux, Zygote, Statistics, Printf

function greedy_action(route::Route, hubs::Vector{Int})::Int
    route === nothing && return N_ACTIONS
    length(route) == 2 && return 1
    hub     = route[2]
    top     = hubs[1:min(length(hubs), MAX_HUBS)]
    idx     = findfirst(==(hub), top)
    idx === nothing ? N_ACTIONS : idx + 1
end

# Batched cross-entropy: single forward pass → Zygote only sees matrix op
function bc_loss(policy, dd::Vector{DemandData}, labels::Vector{Int})::Float32
    logits_mat, valid_idxs = batch_logits(policy, dd)
    isempty(valid_idxs) && return 0f0
    total = 0f0
    for (j, i) in enumerate(valid_idxs)
        probs  = masked_softmax(logits_mat[:, j], dd[i].mask)
        total -= log(probs[labels[i]] + 1f-8)
    end
    total / length(valid_idxs)
end

function pretrain_bc!(
    policy, env::RASEnv;
    n_epochs    = 300,
    lr          = 1f-3,
    print_every = 50,
)
    println("  Precomputing features...")
    dd = precompute(env)

    greedy!(env)
    greedy_score = evaluate(env)
    all_hubs = [ranked_hubs(d, env) for d in env.demands]
    labels   = [greedy_action(env.routings[i], all_hubs[i]) for i in eachindex(env.demands)]

    opt_state = Flux.setup(Adam(lr), policy)
    println("\n── Behavioral Cloning ──")
    @printf("  Epochs=%d  greedy=%.0f\n", n_epochs, greedy_score)

    for ep in 1:n_epochs
        loss_val, grads = Flux.withgradient(policy) do m
            bc_loss(m, dd, labels)
        end
        Flux.update!(opt_state, policy, grads[1])

        if ep % print_every == 0
            # accuracy check (no gradient needed)
            logits_mat, valid_idxs = batch_logits(policy, dd)
            correct = sum(argmax(logits_mat[:, j]) == labels[valid_idxs[j]]
                         for j in eachindex(valid_idxs))
            acc = correct / length(valid_idxs) * 100
            @printf("  ep %4d | loss=%.4f | acc=%.1f%%\n", ep, loss_val, acc)
        end
    end

    dd
end
