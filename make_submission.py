"""
Generate submission.csv for RAS 2026 PSC using sample solutions as baseline.

Strategy:
  x1.0 scenarios (IDs 1, 4, 7): use sample solutions as-is
  x2.0 scenarios (IDs 2, 5, 8): same solution, demand_multiplier=2.0
                                  (unserved demand penalized but solution is VALID)
  x0.5 scenarios (IDs 0, 3, 6): scale down blocking sequence volumes by 0.5
                                  (volumes rounded down, zero-volume rows removed)

Run from the repo root:
    python make_submission.py
"""
import copy
import json
import math
import csv
import io
from pathlib import Path

BASE = Path(__file__).parent / "ras_release_v1.1/ras_release_v1.1"
SAMPLES = BASE / "scoring/sample_solutions"
OUT_DIR = BASE / "scoring"

SAMPLE_FILES = {
    "l1": SAMPLES / "solution_result_l1_10.json",
    "l2": SAMPLES / "solution_result_l2_10.json",
    "l3": SAMPLES / "solution_result_l3_10.json",
}

# Kaggle case ID mapping: 0-8
# 0=l1_05, 1=l1_10, 2=l1_20, 3=l2_05, 4=l2_10, 5=l2_20, 6=l3_05, 7=l3_10, 8=l3_20
CASE_ORDER = [
    ("l1", 0.5), ("l1", 1.0), ("l1", 2.0),
    ("l2", 0.5), ("l2", 1.0), ("l2", 2.0),
    ("l3", 0.5), ("l3", 1.0), ("l3", 2.0),
]


def load_sample(layer: str) -> dict:
    path = SAMPLE_FILES[layer]
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_x10(data: dict) -> dict:
    """Return a copy with demand_multiplier=1.0 (no change)."""
    out = copy.deepcopy(data)
    out["inputs"]["settings"]["demand_multiplier"] = 1.0
    return out


def make_x20(data: dict) -> dict:
    """Return a copy with demand_multiplier=2.0.
    The same blocking plan is used; half the x2.0 demand is served, rest penalized.
    C9 is satisfied because submitted vol <= demand_vol (original) <= demand_vol*2.0.
    """
    out = copy.deepcopy(data)
    out["inputs"]["settings"]["demand_multiplier"] = 2.0
    return out


def make_x05(data: dict) -> dict:
    """Return a copy with demand_multiplier=0.5.
    Scale blocking sequence volumes down by 0.5 (floor), remove zero-vol rows.
    Blocks that lose all volume are removed from Block Design and Block Route.
    """
    out = copy.deepcopy(data)
    out["inputs"]["settings"]["demand_multiplier"] = 0.5

    # Scale sequence volumes
    new_seqs = []
    for seq in out["outputs"]["2 Blocking Sequence"]:
        new_vol = math.floor(seq["volume"] * 0.5)
        if new_vol <= 0:
            continue  # unserved demand — penalized but valid
        new_seq = dict(seq)
        new_seq["volume"] = new_vol
        new_seqs.append(new_seq)
    out["outputs"]["2 Blocking Sequence"] = new_seqs

    # Find which blocks have actual flow after scaling
    active_blocks: set[int] = set()
    for seq in new_seqs:
        for bid_str in str(seq["blocking_sequence"]).split(" -> "):
            bid_str = bid_str.strip()
            if bid_str:
                active_blocks.add(int(bid_str))

    # Filter Block Design and Block Route to only active blocks
    out["outputs"]["1 Block Design"] = [
        b for b in out["outputs"]["1 Block Design"]
        if int(b["block_id"]) in active_blocks
    ]
    out["outputs"]["3 Block Route"] = [
        r for r in out["outputs"]["3 Block Route"]
        if int(r["block_id"]) in active_blocks
    ]

    # Recompute block_volume in Block Design
    block_vol_map: dict[int, float] = {}
    for seq in new_seqs:
        for bid_str in str(seq["blocking_sequence"]).split(" -> "):
            bid_str = bid_str.strip()
            if bid_str:
                bid = int(bid_str)
                block_vol_map[bid] = block_vol_map.get(bid, 0) + new_seqs[0]["volume"]

    # Simpler recomputation
    block_vol_map2: dict[int, float] = {}
    for seq in new_seqs:
        bids = [int(x.strip()) for x in str(seq["blocking_sequence"]).split(" -> ") if x.strip()]
        for bid in bids:
            block_vol_map2[bid] = block_vol_map2.get(bid, 0) + seq["volume"]

    for b in out["outputs"]["1 Block Design"]:
        bid = int(b["block_id"])
        b["block_volume"] = int(block_vol_map2.get(bid, b.get("block_volume", 0)))

    return out


def generate_solution(layer: str, multiplier: float) -> dict | None:
    """Generate solution dict for a given layer and demand multiplier."""
    data = load_sample(layer)
    if multiplier == 1.0:
        return make_x10(data)
    elif multiplier == 2.0:
        return make_x20(data)
    elif multiplier == 0.5:
        return make_x05(data)
    return None


def main():
    rows = [["ID", "data"]]
    for case_id, (layer, multiplier) in enumerate(CASE_ORDER):
        print(f"[ID {case_id}] {layer} x{multiplier} ...", end=" ")
        sol = generate_solution(layer, multiplier)
        if sol is None:
            data_str = "{}"
            print("EMPTY")
        else:
            data_str = json.dumps(sol, separators=(",", ":"))
            n_blocks = len(sol["outputs"]["1 Block Design"])
            n_seqs = len(sol["outputs"]["2 Blocking Sequence"])
            print(f"blocks={n_blocks}, seqs={n_seqs}, len={len(data_str):,}")
        rows.append([case_id, data_str])

    out_path = OUT_DIR / "submission.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\nWrote {out_path}")
    print(f"Total rows: {len(rows)-1} data + 1 header")


if __name__ == "__main__":
    main()
