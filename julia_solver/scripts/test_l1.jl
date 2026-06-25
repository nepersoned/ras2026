include("../src/env.jl")

BASE = joinpath(@__DIR__, "..", "..", "ras_release_v1.1", "ras_release_v1.1")

env = make_env("l1", 1.0, BASE)

println("Demands: $(length(env.demands))")
println("Yards:   $(length(env.all_yard_ids))")

println("\nRunning greedy...")
t0 = time()
greedy!(env)
println("Greedy done in $(round(time()-t0, digits=2))s")

score = evaluate(env)
println("Stress score: $(round(Int, score))")

served = count(r -> r !== nothing, env.routings)
println("Served: $served / $(length(env.demands))")

println("\nRunning 10 LNS steps (k=20)...")
for i in 1:10
    delta = lns_step!(env, 20)
    score = evaluate(env)
    println("  step $i: score=$(round(Int, score))  delta=$(round(Int, delta))")
end
