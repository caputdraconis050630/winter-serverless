# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP-H8: does meta-training *on Huawei* close the cell-2 gap?

Everything in the H-series so far is transfer-only -- the Azure-2021 body is
frozen and never sees Huawei -- while the Azure steady-state parity the paper
claims was measured with a body meta-trained on that trace.  So the Huawei
cell-2 failure has two readings that the WP-H5 diagnostics could narrow but not
separate: cross-provider transfer degradation, or a property of the converged
regime itself.  This script builds the missing arm.

Protocol -- the registered S1 one, not a new one:

  * meta-training set: a random 70/10/20 *function* hold-out (seed 42, the
    registered split) over Huawei's 848 active functions, exactly
    TaskBuilder.split_s1_random.  Function-disjoint, same time window -- this
    is what the Azure campaign does, so no extra leakage is introduced.
  * evaluation set: the h_mixed functions that fall in the S1 *test* split.
    The body has never seen them.
  * comparison: on that same subset, the transfer arm (Azure-2021 checkpoint)
    and EWMA are re-read from the WP-H5 per-function results.  The simulator
    treats functions independently, so per-function CSR from the full-pool run
    is exact for any subset -- no re-simulation of the old arms is needed.

Stages:
  --prepare  data/processed_huawei_meta/  (848 active functions x 14 days)
  --train    results/runs/best_anil_ridge_huawei_s0.pt
  --export   DES jobs for the Huawei-trained arm on the eval subset
  --report   transfer gap vs Huawei-trained gap on the identical function set
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SRC_DIR = PROJECT_ROOT / "data" / "processed_huawei"
META_DIR = PROJECT_ROOT / "data" / "processed_huawei_meta"
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
STEADY_T = 20160
SPLIT_SEED = 42          # registered
TRAIN_SEED = 0           # registered (s0)
TAG = "anil_ridge_huawei"
AZURE_CKPT = "results_azure2021/runs/best_anil_ridge_s2_s0.pt"
RHOS = [1.0, 10.0, 100.0]
SEEDS = [0, 1, 2]
SPLIT_NAME = "h_mixed_metatest"
CHUNK = 64               # rows per write, keeps peak RSS bounded


def prepare():
    """Compact the active pool into a trainable dataset (848 x 20160 x 11)."""
    META_DIR.mkdir(parents=True, exist_ok=True)
    splits = np.load(SRC_DIR / "splits.npz")
    rows = np.sort(np.concatenate([np.asarray(splits[k])
                                   for k in ("h_mixed", "h_sparse", "h_saturated")]))
    assert len(np.unique(rows)) == len(rows), "pools overlap"
    print(f"Active pool: {len(rows)} functions")

    feats_mm = np.load(SRC_DIR / "features.npy", mmap_mode="r")
    counts_mm = np.load(SRC_DIR / "counts.npy", mmap_mode="r")
    nF = feats_mm.shape[2]

    from numpy.lib.format import open_memmap
    fo = open_memmap(META_DIR / "features.npy", mode="w+", dtype=np.float32,
                     shape=(len(rows), STEADY_T, nF))
    co = open_memmap(META_DIR / "counts.npy", mode="w+", dtype=np.int32,
                     shape=(len(rows), STEADY_T))
    for s in range(0, len(rows), CHUNK):
        e = min(s + CHUNK, len(rows))
        idx = rows[s:e]
        fo[s:e] = np.asarray(feats_mm[idx][:, :STEADY_T, :], dtype=np.float32)
        co[s:e] = np.asarray(counts_mm[idx][:, :STEADY_T], dtype=np.int32)
        print(f"  rows {e}/{len(rows)}", flush=True)
    fo.flush(); co.flush()
    del fo, co

    # Descriptor table, re-indexed to the compact rows.  active_functions.csv is
    # keyed by func_id, so align through func_ids.npy rather than by position.
    func_ids = np.load(SRC_DIR / "func_ids.npy", allow_pickle=True)
    af = pd.read_csv(SRC_DIR / "active_functions.csv")
    # Huawei func_ids are the raw column indices, stored as strings in
    # func_ids.npy but as int64 in active_functions.csv -- align on str.
    want = pd.DataFrame({"func_id": [str(x) for x in func_ids[rows]]})
    af = af.copy()
    af["func_id"] = af["func_id"].astype(str)
    merged = want.merge(af, on="func_id", how="left")
    assert len(merged) == len(rows)
    missing = int(merged["mean_rate_per_min"].isna().sum())
    print(f"  descriptor rows matched: {len(merged) - missing}/{len(rows)}")
    merged.to_csv(META_DIR / "active_functions.csv", index=False)
    np.save(META_DIR / "func_ids.npy", func_ids[rows])
    np.save(META_DIR / "source_rows.npy", rows)

    # h_mixed membership in the compact index space
    pos = {int(r): i for i, r in enumerate(rows)}
    h_mixed_compact = np.array([pos[int(r)] for r in np.asarray(splits["h_mixed"])])
    np.save(META_DIR / "h_mixed_compact.npy", h_mixed_compact)

    from src.tasks.task_builder import TaskBuilder
    tb = TaskBuilder(np.load(META_DIR / "features.npy", mmap_mode="r"),
                     np.load(META_DIR / "counts.npy", mmap_mode="r"),
                     merged, context_len=60, horizons=(1,))
    s1 = tb.split_s1_random(seed=SPLIT_SEED)
    np.savez(META_DIR / "splits.npz",
             s1_train=s1["train"], s1_val=s1["val"], s1_test=s1["test"],
             h_mixed=h_mixed_compact, source_rows=rows)
    ev = np.intersect1d(s1["test"], h_mixed_compact)
    json.dump({"n_active": int(len(rows)), "n_train": int(len(s1["train"])),
               "n_val": int(len(s1["val"])), "n_test": int(len(s1["test"])),
               "n_h_mixed": int(len(h_mixed_compact)),
               "n_eval_h_mixed_test": int(len(ev)),
               "split_seed": SPLIT_SEED, "steady_t": STEADY_T},
              open(META_DIR / "meta.json", "w"), indent=2)
    print(f"S1 split: train {len(s1['train'])}, val {len(s1['val'])}, "
          f"test {len(s1['test'])}; h_mixed in test = {len(ev)} (the eval set)")


def train(seed=TRAIN_SEED, eval_every=500, tag=TAG):
    """Meta-train on Huawei.

    Defaults are the registered settings.  Note that the registered rule stops
    after a single non-improving validation check (max_patience // eval_every
    = 1 at eval_every=500); on Huawei that halts at step 2500 of 5000 with val
    CRPS still falling.  The same rule produced the Azure checkpoints, so the
    primary comparison is like-for-like -- `--eval-every 250` is run as a
    robustness variant that survives one uptick, and extra seeds check seed
    stability.
    """
    os.environ["SF_DATA_DIR"] = "processed_huawei_meta"
    import torch
    torch.set_num_threads(1)
    from scripts.phase4_train import train_meta_learner

    features = np.load(META_DIR / "features.npy", mmap_mode="r")
    counts = np.load(META_DIR / "counts.npy", mmap_mode="r")
    sp = np.load(META_DIR / "splits.npz")
    splits = {"train": sp["s1_train"], "val": sp["s1_val"], "test": sp["s1_test"]}
    print(f"Meta-training on Huawei: {len(splits['train'])} train / "
          f"{len(splits['val'])} val functions, {features.shape[1]} ticks, "
          f"seed {seed}, eval_every {eval_every}, tag {tag}")
    trainer, best = train_meta_learner(
        features, counts, splits, seed=seed,
        body_type="tcn", head_type="ridge",
        n_steps=5000, batch_size=32, k_support=10, k_query=32,
        horizons=(1,), eval_every=eval_every, tag=tag)
    print(f"Best val CRPS {best:.4f} -> results/runs/best_{tag}_s{seed}.pt")


def export(metas):
    """DES jobs for the Huawei-trained arms on the held-out h_mixed functions."""
    os.environ["SF_DATA_DIR"] = "processed_huawei_meta"
    from scripts.phase6_des import (COLD_INIT, decisions_from_rates,
                                    rates_ewma, rates_oracle, rates_a5)
    from src.decision.newsvendor import newsvendor_quantile
    import torch
    torch.set_num_threads(1)
    from src.models.heads import N_QUANTILES
    from src.meta.trainer import ANILMetaTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = np.load(META_DIR / "features.npy", mmap_mode="r")
    counts_mm = np.load(META_DIR / "counts.npy", mmap_mode="r")
    sp = np.load(META_DIR / "splits.npz")
    rows = np.intersect1d(sp["s1_test"], sp["h_mixed"])
    src_rows = np.asarray(sp["source_rows"])[rows]
    print(f"Eval set: {len(rows)} held-out h_mixed functions")

    dur_df = pd.read_csv(SRC_DIR / "duration_stats.csv")
    dm = np.nan_to_num(dur_df["dur_mean"].to_numpy()[src_rows], nan=1.0)
    ds = np.nan_to_num(dur_df["dur_std"].to_numpy()[src_rows], nan=0.5)

    counts = np.asarray(counts_mm[rows], dtype=np.float32)
    feats = np.asarray(features[rows], dtype=np.float32)

    trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge",
                              in_features=feats.shape[2], embedding_dim=64,
                              n_quantiles=N_QUANTILES, n_horizons=1, device=device)

    rate_mats = {"B4a_ewma": rates_ewma(counts), "Oracle": rates_oracle(counts)}
    arms = [("A5_transfer", PROJECT_ROOT / AZURE_CKPT)]
    for spec in metas:
        ck = RUNS_DIR / f"best_{spec}.pt"
        if ck.exists():
            arms.append((f"A5_meta_{spec.replace(TAG, 'hw')}", ck))
        else:
            print(f"  (missing checkpoint {ck.name} -- skipped)")
    for name, ck in arms:
        trainer.load(ck)
        print(f"  {name}: {Path(ck).name}", flush=True)
        rate_mats[name] = rates_a5(counts, feats, trainer, device)

    jdir = RUNS_DIR / "split_jobs_huawei_meta" / SPLIT_NAME
    jdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(jdir / "shared.npz", counts=counts.astype(np.int64),
                        dur_means=dm, dur_stds=ds)
    job_meta = []
    for method, rates in rate_mats.items():
        for rho in RHOS:
            pw, ka = decisions_from_rates(rates, newsvendor_quantile(rho))
            fname = f"{method}__rho{rho}.npz"
            np.savez_compressed(jdir / fname, prewarm=pw, keepalive=ka)
            job_meta.append({"method": method, "rho": rho, "file": fname})
    json.dump({"split": SPLIT_NAME, "seeds": SEEDS, "cold_init": COLD_INIT,
               "jobs": job_meta, "eval_rows_compact": rows.tolist(),
               "eval_rows_source": src_rows.tolist()},
              open(jdir / "jobs.json", "w"))
    print(f"  exported {len(job_meta)} jobs -> {jdir}")


def report():
    from scipy import stats as sstats
    res = json.load(open(RUNS_DIR / "revision_h8_huawei_meta.json"))
    jdir = RUNS_DIR / "split_jobs_huawei_meta" / SPLIT_NAME
    meta = json.load(open(jdir / "jobs.json"))
    src_rows = np.asarray(meta["eval_rows_source"])

    # transfer-arm and EWMA per-function CSR from the full-pool WP-H5 run,
    # restricted to the same functions (the simulator is per-function exact)
    old = json.load(open(RUNS_DIR / "revision_h5_recal_huawei.json"))
    h_mixed_rows = np.asarray(np.load(SRC_DIR / "splits.npz")["h_mixed"])
    sel = np.searchsorted(h_mixed_rows, src_rows)
    assert np.array_equal(h_mixed_rows[sel], src_rows), "row alignment failed"

    def pooled(rows_json, method, rho, idx=None):
        s = [r for r in rows_json if r["method"] == method and r["cost_ratio"] == rho]
        if not s:
            return None
        cold = np.mean([r["func_cold"] for r in s], axis=0)
        total = np.mean([r["func_total"] for r in s], axis=0)
        if idx is not None:
            cold, total = cold[idx], total[idx]
        return float(cold.sum() / max(total.sum(), 1e-9)) * 100.0, cold, total

    out = {"n_eval_functions": int(len(src_rows)), "rhos": RHOS, "by_rho": {}}
    lines = ["rho,method,csr_pct,source"]
    for rho in RHOS:
        meta_names = sorted({r["method"] for r in res
                             if r["method"].startswith("A5_meta_")})
        new_tr = pooled(res, "A5_transfer", rho)
        new_ew = pooled(res, "B4a_ewma", rho)
        # headline arm = the registered settings at the registered seed;
        # the other arms are the seed / patience robustness spread
        primary = "A5_meta_hw_s0" if "A5_meta_hw_s0" in meta_names else (
            meta_names[0] if meta_names else None)
        new_meta = pooled(res, primary, rho) if primary else None
        e_primary = primary
        old_tr = pooled(old, "A5_full_system", rho, sel)
        old_ew = pooled(old, "B4a_ewma", rho, sel)
        if not (new_meta and new_ew):
            continue
        e = {"ewma_csr_pct": new_ew[0],
             "primary_meta_arm": e_primary,
             "huawei_meta_csr_pct": new_meta[0],
             "all_meta_arms": {m: pooled(res, m, rho)[0] for m in meta_names},
             "all_meta_gaps_pp": {m: pooled(res, m, rho)[0] - new_ew[0]
                                  for m in meta_names},
             "transfer_csr_pct": new_tr[0] if new_tr else None,
             "gap_huawei_meta_pp": new_meta[0] - new_ew[0],
             "gap_transfer_pp": (new_tr[0] - new_ew[0]) if new_tr else None,
             "gap_transfer_pp_wp_h5": (old_tr[0] - old_ew[0]) if old_tr else None}
        if new_tr:
            fa = new_meta[1] / np.maximum(new_meta[2], 1e-9)
            fb = new_tr[1] / np.maximum(new_tr[2], 1e-9)
            m = ~np.isclose(fa, fb)
            if m.sum() > 10:
                e["wilcoxon_p_meta_vs_transfer"] = float(
                    sstats.wilcoxon(fa[m], fb[m]).pvalue)
                e["win_rate_meta_vs_transfer"] = float((fa < fb).mean())
        out["by_rho"][str(rho)] = e
        for k, v in (("B4a_ewma", new_ew[0]), ("A5_huawei_meta", new_meta[0]),
                     ("A5_transfer", new_tr[0] if new_tr else float("nan"))):
            lines.append(f"{rho},{k},{v:.4f},h8")

    json.dump(out, open(RUNS_DIR / "revision_h8_huawei_meta_stats.json", "w"),
              indent=2, default=float)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    (TABLES_DIR / "T_huawei_meta.csv").write_text("\n".join(lines) + "\n")

    print(f"\nHeld-out h_mixed functions: {len(src_rows)}")
    for rho, e in out["by_rho"].items():
        print(f"  rho={rho:<6}: EWMA {e['ewma_csr_pct']:.4f}%  "
              f"transfer {e['transfer_csr_pct']:.4f}%  "
              f"Huawei-meta {e['huawei_meta_csr_pct']:.4f}%")
        print(f"      gap  transfer {e['gap_transfer_pp']:+.4f} pp  ->  "
              f"Huawei-meta {e['gap_huawei_meta_pp']:+.4f} pp"
              + (f"   (Wilcoxon p={e['wilcoxon_p_meta_vs_transfer']:.2e}, "
                 f"win {e['win_rate_meta_vs_transfer']:.3f})"
                 if "wilcoxon_p_meta_vs_transfer" in e else ""))
    print("\nSaved revision_h8_huawei_meta_stats.json + T_huawei_meta.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("prepare", "train", "export", "report"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--seed", type=int, default=TRAIN_SEED)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--metas", default=f"{TAG}_s{TRAIN_SEED}",
                    help="comma-separated checkpoint stems under results/runs, "
                         "without the best_ prefix and .pt suffix")
    a = ap.parse_args()
    if not any([a.prepare, a.train, a.export, a.report]):
        ap.error("pass one of --prepare/--train/--export/--report")
    if a.prepare:
        prepare()
    if a.train:
        train(seed=a.seed, eval_every=a.eval_every, tag=a.tag)
    if a.export:
        export([m for m in a.metas.split(",") if m])
    if a.report:
        report()
