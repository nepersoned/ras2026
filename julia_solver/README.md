# RAS 2026 Julia RL Solver

## Structure

```
julia_solver/
├── src/
│   ├── env.jl          # MDP environment (data loading, cost eval, LNS step)
│   ├── dijkstra.jl     # IC-penalized Dijkstra on physical network
│   ├── greedy.jl       # Greedy initial solution
│   ├── cost.jl         # Stress score calculation
│   ├── policy.jl       # Flux neural network policy
│   └── reinforce.jl    # REINFORCE training loop
├── scripts/
│   ├── train_l1.jl     # Train on L1 (×0.5 / ×1.0 / ×2.0)
│   ├── infer_l2.jl     # Inference on L2
│   ├── infer_l3.jl     # Inference on L3
│   └── gurobi_l1.jl    # Gurobi optimal solution for L1 (baseline)
├── models/             # Saved policy weights (.bson)
├── results/            # Output JSON solutions
└── data/               # Symlink or copy of dataset CSVs
```

## Pipeline

1. `gurobi_l1.jl`  → L1 optimal solution (training target)
2. `train_l1.jl`   → train REINFORCE policy on L1
3. `infer_l2.jl`   → apply trained policy to L2
4. `infer_l3.jl`   → apply trained policy to L3
