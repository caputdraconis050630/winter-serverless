# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP-A: Azure Functions 2019 (D1) preprocessing at scale.

Builds the primary-dataset pipeline the proposal specified:
- 14-day per-minute counts for the full function universe (union of days,
  zero-filled where a function is absent)
- vectorized descriptors -> active pool (>=100 invocations),
  stratified-sampled to a cap while preserving frequency-bucket proportions
- per-function duration stats from the durations CSVs (ms -> s)
- 11-channel feature tensor written chunk-wise into a float32 memmap
  (RAM is 23 GB; the tensor is ~26 GB at 30k functions)
- app co-rate (ch.10) computed from real HashApp sibling functions

Outputs to data/processed_2019/ (2021 artifacts are preserved).
"""

import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import features_from_counts

RAW = PROJECT_ROOT / "data" / "raw" / "azure2019"
OUT = PROJECT_ROOT / "data" / "processed_2019"
TABLES = PROJECT_ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

N_DAYS = 14
MIN_PER_DAY = 1440
T = N_DAYS * MIN_PER_DAY  # 20160
MIN_INVOCATIONS = 100
ACTIVE_CAP = 30000
SPARSE_SAMPLE = 5000
FEAT_CHUNK = 500
# sdb aborts its ext4 journal under sustained write bursts -> write big
# memmaps in bounded bursts, flush, and pause between bursts.
WRITE_CHUNK_ROWS = 8192
IO_PAUSE = 0.5

BUCKETS = [("<1/day", 0, 1 / 1440), ("1/day-1/h", 1 / 1440, 1 / 60),
           ("1/h-1/min", 1 / 60, 1.0), (">1/min", 1.0, np.inf)]


def bucket_of(rate):
    for name, lo, hi in BUCKETS:
        if lo <= rate < hi:
            return name
    return BUCKETS[-1][0]


def inv_file(d):
    return RAW / f"invocations_per_function_md.anon.d{d:02d}.csv"


def dur_file(d):
    return RAW / f"function_durations_percentiles.anon.d{d:02d}.csv"


# ------------------------------------------------------------------
def pass1_universe():
    """Union of (owner, app, function) keys across all days."""
    print("Pass 1: building function universe...")
    key2idx = {}
    key_app = []
    key_trigger = []
    for d in range(1, N_DAYS + 1):
        df = pd.read_csv(inv_file(d),
                         usecols=["HashOwner", "HashApp", "HashFunction",
                                  "Trigger"], dtype=str)
        keys = (df["HashOwner"] + "|" + df["HashApp"] + "|"
                + df["HashFunction"]).values
        for k, app, trig in zip(keys, df["HashApp"].values,
                                df["Trigger"].values):
            if k not in key2idx:
                key2idx[k] = len(key2idx)
                key_app.append(app)
                key_trigger.append(trig)
        print(f"  day {d:2d}: cumulative universe {len(key2idx):,}")
    return key2idx, np.array(key_app), np.array(key_trigger)


def pass2_counts(key2idx):
    """Fill the [N_all, 20160] int32 counts memmap day by day."""
    print("Pass 2: assembling counts memmap...")
    N_all = len(key2idx)
    counts = np.lib.format.open_memmap(
        OUT / "all_counts_universe.npy", mode="w+",
        dtype=np.int32, shape=(N_all, T))
    total_check = 0
    for d in range(1, N_DAYS + 1):
        t0 = time.time()
        df = pd.read_csv(inv_file(d), dtype={"HashOwner": str, "HashApp": str,
                                             "HashFunction": str,
                                             "Trigger": str})
        keys = (df["HashOwner"] + "|" + df["HashApp"] + "|"
                + df["HashFunction"]).values
        rows = np.array([key2idx[k] for k in keys])
        mat = df[[str(i) for i in range(1, MIN_PER_DAY + 1)]].to_numpy(
            dtype=np.int64)
        total_check += int(mat.sum())
        order = np.argsort(rows)          # sorted rows -> mostly sequential I/O
        rows_s = rows[order]
        mat_s = mat[order].astype(np.int32)
        col = slice((d - 1) * MIN_PER_DAY, d * MIN_PER_DAY)
        for ws in range(0, len(rows_s), WRITE_CHUNK_ROWS):
            we = min(ws + WRITE_CHUNK_ROWS, len(rows_s))
            counts[rows_s[ws:we], col] = mat_s[ws:we]
            counts.flush()
            time.sleep(IO_PAUSE)
        print(f"  day {d:2d}: {len(df):,} functions, {mat.sum():,} invocations "
              f"[{time.time()-t0:.0f}s]")
    counts.flush()
    print(f"  raw total invocations: {total_check:,}")
    return counts, total_check


def descriptors(counts):
    """Vectorized per-function descriptors (chunked)."""
    print("Computing descriptors...")
    N_all = counts.shape[0]
    total = np.zeros(N_all, dtype=np.int64)
    nz_frac = np.zeros(N_all)
    fano = np.zeros(N_all)
    iat_cv = np.zeros(N_all)
    e_daily = np.zeros(N_all)
    e_weekly = np.zeros(N_all)

    CH = 4096
    for s in range(0, N_all, CH):
        e = min(s + CH, N_all)
        c = counts[s:e].astype(np.float64)
        total[s:e] = c.sum(axis=1).astype(np.int64)
        nz = c > 0
        nz_frac[s:e] = nz.mean(axis=1)
        mean = c.mean(axis=1)
        var = c.var(axis=1)
        fano[s:e] = np.where(mean > 0, var / np.maximum(mean, 1e-12), 0)
        # spectral energy at daily (14 cycles) and weekly (2 cycles)
        F = np.abs(np.fft.rfft(c, axis=1)) ** 2
        tot_e = F[:, 1:].sum(axis=1) + 1e-12
        e_daily[s:e] = F[:, 14] / tot_e
        e_weekly[s:e] = F[:, 2] / tot_e
        # IAT CV from nonzero-minute gaps (only worth computing when active)
        for i in range(s, e):
            if total[i] >= MIN_INVOCATIONS:
                nzi = np.flatnonzero(counts[i])
                if len(nzi) > 2:
                    iats = np.diff(nzi)
                    m = iats.mean()
                    iat_cv[i] = iats.std() / m if m > 0 else 0
        if (s // CH) % 5 == 0:
            print(f"  chunk {s:,}/{N_all:,}")
    return total, nz_frac, fano, iat_cv, e_daily, e_weekly


def load_durations():
    """Per-function duration mean/std across days (ms -> s)."""
    print("Loading durations...")
    acc = {}
    for d in range(1, N_DAYS + 1):
        df = pd.read_csv(dur_file(d), dtype={"HashOwner": str, "HashApp": str,
                                             "HashFunction": str})
        keys = (df["HashOwner"] + "|" + df["HashApp"] + "|"
                + df["HashFunction"]).values
        avg = df["Average"].to_numpy(np.float64)
        cnt = df["Count"].to_numpy(np.float64)
        p25 = df["percentile_Average_25"].to_numpy(np.float64)
        p75 = df["percentile_Average_75"].to_numpy(np.float64)
        for k, a, n, lo, hi in zip(keys, avg, cnt, p25, p75):
            if k in acc:
                pa, pn, plo, phi_ = acc[k]
                acc[k] = (pa + a * n, pn + n, plo + lo * n, phi_ + hi * n)
            else:
                acc[k] = (a * n, n, lo * n, hi * n)
    return acc


# ------------------------------------------------------------------
def main():
    t_start = time.time()
    key2idx, key_app, key_trigger = pass1_universe()
    counts_all, total_check = pass2_counts(key2idx)
    N_all = counts_all.shape[0]

    total, nz_frac, fano, iat_cv, e_daily, e_weekly = descriptors(counts_all)

    # verification: conservation
    assert int(total.sum()) == total_check, "counts conservation FAILED"
    print(f"Conservation check PASS ({total.sum():,} invocations)")

    rate = total / T
    active_mask = total >= MIN_INVOCATIONS
    n_active_all = int(active_mask.sum())
    print(f"Universe {N_all:,} functions; active(>= {MIN_INVOCATIONS}): "
          f"{n_active_all:,}; sparse: {N_all - n_active_all:,}")

    # ---- stratified sampling of the active pool ----
    rng = np.random.default_rng(42)
    active_idx = np.flatnonzero(active_mask)
    if n_active_all > ACTIVE_CAP:
        buckets = np.array([bucket_of(r) for r in rate[active_idx]])
        keep = []
        for name, _, _ in BUCKETS:
            b_idx = active_idx[buckets == name]
            quota = int(round(ACTIVE_CAP * len(b_idx) / n_active_all))
            if len(b_idx) > quota:
                b_idx = rng.choice(b_idx, size=quota, replace=False)
            keep.append(b_idx)
        sel = np.sort(np.concatenate(keep))
        print(f"Stratified-sampled active pool: {len(sel):,} (cap {ACTIVE_CAP:,})")
    else:
        sel = active_idx
        print(f"Active pool below cap; using all {len(sel):,}")

    sparse_idx = np.flatnonzero(~active_mask & (total > 0))
    sparse_sel = (rng.choice(sparse_idx, SPARSE_SAMPLE, replace=False)
                  if len(sparse_idx) > SPARSE_SAMPLE else sparse_idx)

    # ---- durations ----
    dur_acc = load_durations()
    idx2key = {v: k for k, v in key2idx.items()}
    dur_mean = np.full(len(sel), 1.0)
    dur_std = np.full(len(sel), 0.5)
    for j, gi in enumerate(sel):
        k = idx2key[gi]
        if k in dur_acc:
            s_a, s_n, s_lo, s_hi = dur_acc[k]
            if s_n > 0:
                dur_mean[j] = (s_a / s_n) / 1000.0            # ms -> s
                dur_std[j] = max(((s_hi - s_lo) / s_n) / 1.349 / 1000.0, 0.01)
    dur_mean = np.clip(np.nan_to_num(dur_mean, nan=1.0), 0.01, 600.0)
    dur_std = np.clip(np.nan_to_num(dur_std, nan=0.5), 0.01, 300.0)

    # ---- selected counts (materialized, ~2.4 GB at 30k) ----
    print("Materializing selected counts...")
    counts_sel = np.lib.format.open_memmap(
        OUT / "counts.npy", mode="w+", dtype=np.int32,
        shape=(len(sel), T))
    for s in range(0, len(sel), 2048):
        counts_sel[s:s + 2048] = counts_all[sel[s:s + 2048]]
        counts_sel.flush()
        time.sleep(IO_PAUSE)
    counts_sel.flush()

    # ---- app co-rate over selected active functions ----
    print("Computing app co-rates...")
    sel_apps = key_app[sel]
    app_order = {}
    for j, a in enumerate(sel_apps):
        app_order.setdefault(a, []).append(j)
    corate = np.lib.format.open_memmap(
        OUT / "app_corate.npy", mode="w+", dtype=np.float32,
        shape=(len(sel), T))
    rows_since_flush = 0
    for a, members in app_order.items():
        if len(members) == 1:
            corate[members[0]] = 0.0
        else:
            m = np.array(members)
            app_total = counts_sel[m].astype(np.float64).sum(axis=0)
            for j in m:
                corate[j] = np.log1p(app_total - counts_sel[j])
        rows_since_flush += len(members)
        if rows_since_flush >= WRITE_CHUNK_ROWS:
            corate.flush()
            time.sleep(IO_PAUSE)
            rows_since_flush = 0
    corate.flush()

    # ---- features memmap (chunked) ----
    print("Writing features memmap...")
    feats = np.lib.format.open_memmap(
        OUT / "features.npy", mode="w+", dtype=np.float32,
        shape=(len(sel), T, 11))
    for s in range(0, len(sel), FEAT_CHUNK):
        e = min(s + FEAT_CHUNK, len(sel))
        f = features_from_counts(np.asarray(counts_sel[s:e]),
                                 app_corate=np.asarray(corate[s:e]))
        feats[s:e] = f
        feats.flush()
        time.sleep(IO_PAUSE)
        print(f"  features {e:,}/{len(sel):,}")
    feats.flush()

    # ---- descriptor CSVs (task_builder-compatible schema) ----
    def make_df(indices):
        rows = []
        for j, gi in enumerate(np.asarray(indices)):
            k = idx2key[gi]
            rows.append({
                "func_id": k.split("|")[2],
                "app_id": key_app[gi],
                "total_invocations": float(total[gi]),
                "mean_rate_per_min": float(rate[gi]),
                "iat_cv": float(iat_cv[gi]),
                "fano_factor": float(fano[gi]),
                "sparsity": float(1.0 - nz_frac[gi]),
                "daily_spectral_energy": float(e_daily[gi]),
                "weekly_spectral_energy": float(e_weekly[gi]),
                "mean_duration": float(dur_mean[j]) if j < len(dur_mean) else 1.0,
                "n_invocations_in_trace": int(total[gi]),
                "freq_bucket": bucket_of(rate[gi]),
                "trigger": key_trigger[gi],
            })
        return pd.DataFrame(rows)

    active_df = make_df(sel)
    active_df.to_csv(OUT / "active_functions.csv", index=False)
    make_df(sparse_sel).to_csv(OUT / "sparse_functions.csv", index=False)

    np.save(OUT / "func_ids.npy",
            np.array([idx2key[g].split("|")[2] for g in sel]))
    pd.DataFrame({"dur_mean": dur_mean, "dur_std": dur_std}).to_csv(
        OUT / "duration_stats.csv", index=False)

    # ---- T1 ----
    stats = []
    for name, lo, hi in BUCKETS:
        m_all = (rate >= lo) & (rate < hi) & (total > 0)
        m_act = np.zeros(N_all, bool); m_act[sel] = True
        m_act &= m_all
        stats.append({
            "frequency_bucket": name,
            "total_functions": int(m_all.sum()),
            "active_functions": int(m_act.sum()),
            "sparse_functions": int((m_all & ~active_mask).sum()),
            "total_invocations": int(total[m_all].sum()),
            "mean_rate_per_min": float(rate[m_all].mean()) if m_all.any() else 0,
            "mean_sparsity": float(1 - nz_frac[m_all].mean()) if m_all.any() else 0,
            "mean_fano": float(fano[m_all].mean()) if m_all.any() else 0,
        })
    stats.append({
        "frequency_bucket": "TOTAL",
        "total_functions": int((total > 0).sum()),
        "active_functions": len(sel),
        "sparse_functions": int((~active_mask & (total > 0)).sum()),
        "total_invocations": int(total.sum()),
        "mean_rate_per_min": float(rate[total > 0].mean()),
        "mean_sparsity": float(1 - nz_frac[total > 0].mean()),
        "mean_fano": float(fano[total > 0].mean()),
    })
    t1 = pd.DataFrame(stats)
    t1.to_csv(TABLES / "T1_dataset_stats_2019.csv", index=False)
    print(t1.to_string(index=False))

    meta = {"n_universe": int(N_all), "n_active_all": n_active_all,
            "n_selected": int(len(sel)), "n_sparse_sample": int(len(sparse_sel)),
            "total_invocations": int(total.sum()), "cap": ACTIVE_CAP}
    with open(OUT / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nPhase 1b COMPLETE in {(time.time()-t_start)/60:.1f} min")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
