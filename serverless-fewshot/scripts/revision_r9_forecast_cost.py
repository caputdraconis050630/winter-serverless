#!/usr/bin/env python3
"""R9: end-to-end per-forecast cost of the deployed WINTER path
(pre-registered in revision_r9_PREREG.md).

Torch env: PYTHONPATH=/data/260715/site-packages:. python3.13 scripts/revision_r9_forecast_cost.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.meta.trainer import ANILMetaTrainer  # noqa: E402
from src.models.heads import N_QUANTILES  # noqa: E402

torch.set_num_threads(1)
RUNS = PROJECT_ROOT / "results" / "runs"
L, F = 60, 11
BATCHES = [1000, 10000, 50000]
WARMUP, ITERS = 5, 20


def bench(trainer, device, dtype, batch):
    x = torch.randn(batch, L, F, device=device, dtype=dtype)
    W = torch.randn(batch, 64, N_QUANTILES, device=device, dtype=dtype)
    body = trainer.body.to(device).to(dtype).eval()
    ts = []
    with torch.no_grad():
        for i in range(WARMUP + ITERS):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            phi = body(x)
            _ = torch.bmm(phi.unsqueeze(1), W)      # head read
            if device == "cuda":
                torch.cuda.synchronize()
            if i >= WARMUP:
                ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / batch * 1e6      # us per function


def main():
    ckpt = RUNS / "best_anil_ridge_s1_s0.pt"
    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=F, embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1,
                              device="cuda")
    trainer.load(ckpt)
    print(f"checkpoint: {ckpt}", flush=True)

    out = {"prereg": "revision_r9_PREREG.md", "checkpoint": str(ckpt),
           "batches": BATCHES, "iters": ITERS, "results": {}}
    for device, dtype, tag in [("cuda", torch.bfloat16, "gpu_bf16"),
                               ("cuda", torch.float32, "gpu_fp32"),
                               ("cpu", torch.float32, "cpu_1thread")]:
        out["results"][tag] = {}
        for b in BATCHES:
            if tag == "cpu_1thread" and b > 10000:
                continue
            us = bench(trainer, device, dtype, b)
            out["results"][tag][str(b)] = us
            print(f"  {tag} batch={b}: {us:.2f} us/function", flush=True)

    chronos = json.load(open(RUNS / "revision_e5_chronos.json"))["inference"]
    chronos_us = chronos["mean_batch100_sec"] / 100 * 1e6
    winter_us = out["results"]["gpu_bf16"]["10000"]
    out["comparison"] = {
        "chronos_small_us_per_forecast": chronos_us,
        "winter_us_per_forecast_gpu_bf16_b10k": winter_us,
        "honest_compute_multiplier": chronos_us / winter_us,
        "paper_multiplier_basis": "69us forecast vs 0.55us adaptation solve (asymmetric)",
        "adaptation_solve_us": 0.55,
        "state_multiplier_unaffected": "8KB vs ~95MB bf16",
    }
    path = RUNS / "revision_r9_forecast_cost.json"
    json.dump(out, open(path, "w"), indent=1)
    print(f"honest multiplier: {out['comparison']['honest_compute_multiplier']:.1f}x "
          f"(chronos {chronos_us:.1f}us / winter {winter_us:.2f}us)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
