include("../src/env.jl")
include("../src/policy.jl")
include("../src/reinforce.jl")
include("../src/bc_pretrain.jl")
using BSON: @save

const BASE      = joinpath(@__DIR__, "..", "..", "ras_release_v1.1", "ras_release_v1.1")
const MODEL_DIR = joinpath(@__DIR__, "..", "models")
mkpath(MODEL_DIR)

for mult in [0.5, 1.0, 2.0]
    println("\n" * "="^60)
    println("L1 × $mult")
    println("="^60)

    env    = make_env("l1", mult, BASE)
    policy = make_policy()

    # Phase 1: Behavioral Cloning (warm-start)
    dd = pretrain_bc!(policy, env; n_epochs=300, lr=1f-3, print_every=50)

    g0 = run_greedy!(policy, dd, env)
    @printf("  After BC greedy=%.0f\n", g0)

    # Phase 2: REINFORCE fine-tune
    best = train_reinforce!(policy, env;
        n_episodes  = 2000,
        lr          = 1f-4,
        print_every = 100,
    )

    path = joinpath(MODEL_DIR, "policy_l1_$(Int(mult*10)).bson")
    @save path policy
    println("  Saved → $path")

    final = infer!(policy, env)
    @printf("  Final greedy=%.0f\n", final)
end
