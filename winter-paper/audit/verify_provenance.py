"""Check manuscript-critical counts and checkpoint identity without retraining."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent / "serverless-fewshot"
out = {"cohort_universes": {}, "checkpoints": {}, "source_sha256": {}}
for filename, expected in (("all_counts_universe.npy", 3811), ("counts.npy", 2484)):
    counts = np.load(EXP / "data/processed_2019" / filename, mmap_mode="r")
    n, horizon = counts.shape
    first = np.full(n, -1)
    day1 = np.zeros(n)
    for i, row in enumerate(counts):
        nz = np.flatnonzero(row)
        if len(nz):
            first[i] = nz[0]
            day1[i] = row[nz[0]:nz[0] + 1440].sum()
    eligible = (first >= 4320) & (day1 >= 30)
    n24 = int(np.count_nonzero(eligible & (first < horizon - 1440)))
    n48 = int(np.count_nonzero(eligible & (first < horizon - 2880)))
    assert n24 == expected, (filename, n24)
    record = {"shape": [n, horizon], "candidates_24h": n24, "candidates_48h": n48}
    if filename == "all_counts_universe.npy":
        denominator = int(np.sum(horizon - first[first >= 0]))
        record["post_first_arrival_function_minutes"] = denominator
        record["qualified_first_4h_function_minutes_pct"] = 100 * n24 * 240 / denominator
    if filename == "counts.npy":
        union = set()
        for gap in (1, 3, 7):
            for threshold in (10, 30, 100):
                cand = np.flatnonzero((first >= gap * 1440) & (first < horizon - 1440) & (day1 >= threshold))
                if len(cand) > 100:
                    cand = cand[np.random.default_rng(42).choice(len(cand), 100, replace=False)]
                union.update(cand.tolist())
        holdout = set(np.flatnonzero(eligible & (first < horizon - 2880)).tolist()) - union
        assert n48 == 2419 and len(union) == 772 and len(holdout) == 1755
        record.update(exclusion_union=len(union), holdout_functions=len(holdout))
    out["cohort_universes"][filename] = record
    print(filename, record, flush=True)
splits = np.load(EXP / "data/processed/splits.npz")
train = set(splits["s1_train"].tolist())
out["current_model_overlap"] = {k: len(train & set(splits[k].tolist())) for k in ("s1_test", "s2_test", "s3_test")}
assert out["current_model_overlap"] == {"s1_test": 0, "s2_test": 58, "s3_test": 120}
for split in ("s1", "s2"):
    path = EXP / "results_azure2021/runs" / f"best_anil_ridge_{split}_s0.pt"
    data = torch.load(path, map_location="cpu", weights_only=False)
    out["checkpoints"][split] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "body_elements": sum(t.numel() for t in data["body"].values()),
        "lambda": torch.nn.functional.softplus(data["head"]["log_lambda"]).item(),
    }
    if split == "s1":
        other = torch.load(EXP / "results/runs" / path.name, map_location="cpu", weights_only=False)
        assert all(torch.equal(v, other["body"][k]) for k, v in data["body"].items())
        out["checkpoints"][split]["results_copy_body_equal"] = True
for name in ("revision_a4_crosstrace.py", "revision_h1_huawei_cohort.py",
             "revision_m2a_holdout_cohort.py", "revision_r16_faithful_primary_experiment.py",
             "revision_r17_faithful_gates_2021.py", "revision_r21_cross_provider_faithful.py",
             "revision_r22_ewma_alpha_frontier.py", "revision_v1_hybridfull.py",
             "phase8_ablations.py", "revision_r9_forecast_cost.py", "phase1b_azure2019.py",
             "eval_adapt_biased.py", "phase63_onboarding_drift.py"):
    path = EXP / "scripts" / name
    out["source_sha256"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
(HERE / "verified_provenance.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["checkpoints"], indent=2))
