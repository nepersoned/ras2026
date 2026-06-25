# INFORMS RAS 2026 Problem Solving Competition

**URL**: https://kaggle.com/competitions/informs-ras-2026-problem-solving-competition  
**Deadline**: August 21, 2026 (submission) / June 20, 2026 (registration)  
**Prizes**: $2,000 / $1,000 / $750

## Problem Summary

Railroad Blocking Problem: given a physical rail network + yard-to-yard commodity demands, design a blocking plan:
1. **Block Design** — which blocks to open (from_yard → to_yard, commodity_type, volume)
2. **Blocking Sequence** — which block(s) each demand uses (non-splittable)
3. **Block Route** — physical path on rail network for each block (default = shortest path)

## Network

| Layer | Yards (origin/dest) | OD Pairs | Demands | Volume (×1.0) |
|-------|--------------------:|--------:|--------:|--------------:|
| L1 | 21 / 21 | 411 | 775 | 541,860 cars |
| L2 | 132 / 132 | 11,897 | 18,681 | 2,273,454 cars |
| L3 | 1,041 / 1,413 | 136,677 | 154,907 | 3,204,848 cars |

Physical network shared: 47,193 nodes, 106,570 links, 6 Class I railroads (UP, BNSF, CSX, NS, CN, CPKC).

9 scenarios = 3 layers × 3 demand multipliers (0.5× / 1.0× / 2.0×) — Kaggle IDs 0–8.

## Commodity Types

| Commodity | Block Type | Multi-hop? |
|-----------|-----------|-----------|
| Merchandise | Manifest | Yes (via hump/flat yards) |
| Coal | Bulk | Yes |
| Grain | Bulk | Yes |
| Intermodal | Intermodal | **No** (direct only) |
| Automobile | Multilevel | **No** (direct only) |

## Parameters (setting.csv)

| Parameter | Value |
|-----------|-------|
| block_fixed_cost | $1,500/block |
| transport_cost_coefficient | $1/car-mile |
| interchange_cost | $100/car per railroad change |
| min_block_vol (<100 mi) | 5 cars (350 in problem statement — check setting.csv) |
| min_block_vol (100–500 mi) | 10 cars |
| min_block_vol (>500 mi) | 15 cars |
| max_circuitous_ratio | 1.3 |
| stress_penalty_M | $5/car-mile |

## Objective

**Stress Score** = Operating Cost + M × Unserved Car-Miles

Operating Cost = Fixed + Transport + Handling (intermediate classification) + Interchange

## Constraints

| # | Name | Description |
|---|------|-------------|
| C1 | Flow Conservation | Valid connected path origin→dest, physical route valid |
| C2 | Track Limit | Outbound Manifest/Bulk blocks ≤ num_tracks per yard |
| C3 | Handling Capacity | Classification volume ≤ yard capacity |
| C4 | Min Block Volume | Distance-tier based minimums (see table above) |
| C5 | Link Capacity | Total block flow ≤ link capacity |
| C6 | Max Circuitous Ratio | actual_dist ≤ 1.3 × shortest_path |
| C7 | Single Path | One blocking sequence per commodity, no splitting |
| C8 | Commodity Separation | One commodity type per block |
| C9 | Volume Consistency | No zero/negative rows, no over-service |

## Scoring (Kaggle)

```
Final Score = Tier Score + Normalized Stress Score refinement
```
- Tier score based on how many of 9 cases are solved (lower = better)
- Within tier: averaged Stress Score normalized against benchmark
- Solving more cases → better tier → dramatically lower score

## Evaluation Rubric (Prize)

| Criterion | Weight |
|-----------|--------|
| Solution Quality | 50% |
| Feasibility | 20% |
| Scalability (L2/L3) | 15% |
| Methodology & Report | 15% |

## Key Insights

- **Interchange cost dominates** (~$428M out of $900M for L1 ×1.0): routes crossing railroad boundaries incur $100/car/crossing
- **Interchange-penalized Dijkstra**: adding 100-mile-equivalent penalty per railroad crossing finds paths that minimize total cost
- **C6 constraint**: max circuitous ratio 1.3 — penalized paths must not exceed true shortest path by >30%
- **Sample solution fails C8** (mixes Coal+Grain in same block); our solution beats it on stress score
- **v1.1 rule**: zero/negative volume sequence rows are rejected (relevant for 0.5× scaling)

## Citation

Peiheng Li and Xuesong (Simon) Zhou. INFORMS RAS 2026 Problem Solving Competition.  
https://kaggle.com/competitions/informs-ras-2026-problem-solving-competition, 2026. Kaggle.
