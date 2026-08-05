import json
import os
import sys

import numpy as np

METRICS = ["MPJPE_px", "PA-MPJPE_px", "PCK@0.05", "PCK@0.10"]


def mean_std(vals):
    a = np.asarray(vals, dtype=np.float64)
    if len(a) < 2:
        return float(a.mean()) if len(a) else 0.0, 0.0
    return float(a.mean()), float(a.std(ddof=1))


def main():
    default = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "output",
        "WanAnimate_pose_eval.json",
    )
    path = sys.argv[1] if len(sys.argv) > 1 else default
    if not os.path.exists(path):
        print(f"[error] khong tim thay file: {path}")
        print("Usage: python compute_run_stats.py [path_to_json]")
        sys.exit(1)

    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    if isinstance(records, dict):
        records = [records]
    print(f"# Runs: {len(records)}  |  file: {path}\n")

    sections = [("overall", "Overall Mean"),
                ("overall_joint_weighted", "Overall Joint-Weighted")]
    group_names = ["Body & Arms", "Face Landmarks", "Hand Articulations"]

    rows = []  # (label, metric, mean, std, n)
    for key, label in sections:
        vals = {m: [] for m in METRICS}
        for r in records:
            sec = (r.get(key) or {})
            for m in METRICS:
                if m in sec:
                    vals[m].append(sec[m])
        for m in METRICS:
            if vals[m]:
                mu, sd = mean_std(vals[m])
                rows.append((label, m, mu, sd, len(vals[m])))

    for g in group_names:
        vals = {m: [] for m in METRICS}
        for r in records:
            sec = ((r.get("groups") or {}).get(g) or {})
            for m in METRICS:
                if m in sec:
                    vals[m].append(sec[m])
        for m in METRICS:
            if vals[m]:
                mu, sd = mean_std(vals[m])
                rows.append((g, m, mu, sd, len(vals[m])))

    # Result
    print(f"{'Section':<26}{'Metric':<12}{'Mean':>12}{'Std':>12}{'N':>5}")
    print("-" * 70)
    for label, m, mu, sd, n in rows:
        print(f"{label:<26}{m:<12}{mu:>12.4f}{sd:>12.4f}{n:>5}")


if __name__ == "__main__":
    main()