# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""WP-H1: Huawei Cloud 2023 (SoCC'23 SIR-lab release) preprocessing.

Third trace for the regime-map replication (non-Azure provider). Mirrors
phase1b_azure2019.py so every downstream script runs unchanged under
SF_DATA_DIR=processed_huawei.

Source (CC BY 4.0, https://github.com/sir-lab/data-release):
  public_dataset/csv_files/requests_minute/day_{00..25}.csv
      wide: columns [day, time, 0..5092]; 1440 rows/day; NaN = function not
      present in that minute.  26 days -> T=37,440 minutes, 5,093 functions.
  private_dataset/function_delay_minute/day_{000..140}.csv
      wide: same layout, 200 functions; values are per-minute mean function
      execution time in MILLISECONDS -> the duration model for the public set.
  private_dataset/requests_minute/day_{000..140}.csv
      used only to test whether duration correlates with volume.

Registered constants inherited unchanged from the Azure pipeline:
  MIN_INVOCATIONS=100 (active pool), frequency buckets, sampling seed 42.
New constants, fixed a priori before any evaluation (Table 1 registry):
  STEADY_DAYS=14      steady-state window = first 14 days (Azure tick parity)
  DUR_RANK_RHO=0.30   |Spearman(volume, mean delay)| threshold on the private
                      set: at or above it durations are rank-matched to volume,
                      below it they are drawn i.i.d. from the same empirical
                      pool.  Decided by measurement, not by outcome.
  DUR_SEED=42         duration assignment RNG.

Two documented deviations from the Azure pipeline (both controlled in WP-H4):
  1. No execution durations in the public release -> imputed from the private
     release as above.
  2. No app/owner grouping -> feature channel 10 (app_corate) is identically
     zero.  features_from_counts(app_corate=None) already emits zeros; on
     Azure 2021 that channel is already zero for 90.6% of function-minutes.

Outputs to data/processed_huawei/ (Azure artifacts untouched).
"""

import sys, time, json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import features_from_counts

RAW = PROJECT_ROOT / "data" / "raw" / "huawei2023"
PUB = RAW / "public_dataset" / "csv_files" / "requests_minute"
PRIV_DELAY = RAW / "private_dataset" / "function_delay_minute"
PRIV_REQ = RAW / "private_dataset" / "requests_minute"
OUT = PROJECT_ROOT / "data" / "processed_huawei"
TABLES = PROJECT_ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

N_DAYS = 26
MIN_PER_DAY = 1440
T = N_DAYS * MIN_PER_DAY            # 37,440
STEADY_DAYS = 14
STEADY_T = STEADY_DAYS * MIN_PER_DAY  # 20,160 (Azure parity)
MIN_INVOCATIONS = 100
DUR_RANK_RHO = 0.30
DUR_SEED = 42
FEAT_CHUNK = 250
# sdb aborts its ext4 journal under sustained write bursts -> bounded bursts,
# flush, pause (same guard as phase1b_azure2019).
WRITE_CHUNK_ROWS = 8192
IO_PAUSE = 0.5

BUCKETS = [("<1/day", 0, 1 / 1440), ("1/day-1/h", 1 / 1440, 1 / 60),
           ("1/h-1/min", 1 / 60, 1.0), (">1/min", 1.0, np.inf)]

# Density pools for the DES campaign, defined by the registered frequency
# buckets on the steady window (no new thresholds):
#   sparse    : active but < 1 invocation/hour   (regime-map cell 3, sparse)
#   mixed     : 1/h .. 1/min                     (cell 2, the S1-like middle)
#   saturated : >= 1 invocation/minute           (cell 3, saturated)
POOL_CAP = 1000


def bucket_of(rate):
    for name, lo, hi in BUCKETS:
        if lo <= rate < hi:
            return name
    return BUCKETS[-1][0]


# ------------------------------------------------------------------
def load_public_counts():
    """[N, T] int32 counts + first/last present minute per function."""
    print("Pass 1: reading public requests_minute...")
    counts = None
    func_ids = None
    total_raw = 0.0
    first_present = None
    for d in range(N_DAYS):
        f = PUB / f"day_{d:02d}.csv"
        df = pd.read_csv(f)
        assert list(df.columns[:2]) == ["day", "time"], df.columns[:2]
        assert len(df) == MIN_PER_DAY, (f, len(df))
        arr = df.iloc[:, 2:].to_numpy(dtype=np.float64)     # [1440, N]
        if counts is None:
            n_funcs = arr.shape[1]
            func_ids = np.array(df.columns[2:])
            counts = np.lib.format.open_memmap(
                OUT / "counts.npy", mode="w+", dtype=np.int32,
                shape=(n_funcs, T))
            first_present = np.full(n_funcs, -1, dtype=np.int64)
            last_present = np.full(n_funcs, -1, dtype=np.int64)
        assert arr.shape[1] == counts.shape[0]
        total_raw += np.nansum(arr)
        present = ~np.isnan(arr)                            # [1440, N]
        block = np.nan_to_num(arr, nan=0.0)
        assert np.abs(block - np.round(block)).max() < 1e-6, "non-integral"
        block = np.round(block).astype(np.int32).T          # [N, 1440]
        col = slice(d * MIN_PER_DAY, (d + 1) * MIN_PER_DAY)
        for s in range(0, counts.shape[0], WRITE_CHUNK_ROWS):
            e = min(s + WRITE_CHUNK_ROWS, counts.shape[0])
            counts[s:e, col] = block[s:e]
            counts.flush()
            time.sleep(IO_PAUSE)
        any_p = present.any(axis=0)
        off = d * MIN_PER_DAY
        fp = off + present.argmax(axis=0)
        lp = off + MIN_PER_DAY - 1 - present[::-1].argmax(axis=0)
        new = any_p & (first_present < 0)
        first_present[new] = fp[new]
        last_present[any_p] = lp[any_p]
        print(f"  day {d:2d}: {np.nansum(arr):,.0f} requests, "
              f"{int(any_p.sum()):,} functions present")
    counts.flush()

    np.save(OUT / "first_present.npy", first_present.astype(np.int32))
    np.save(OUT / "last_present.npy", last_present.astype(np.int32))
    np.save(OUT / "func_ids.npy", func_ids)
    tot = int(np.asarray(counts).sum(dtype=np.int64))
    print(f"  universe {counts.shape[0]:,} functions, {tot:,} requests "
          f"(raw sum {total_raw:,.0f})")
    assert abs(tot - total_raw) < 1.0, "conservation FAILED"
    return counts, func_ids


def descriptors(counts, t_end):
    """Per-function descriptors over counts[:, :t_end] (phase1b semantics)."""
    print(f"Computing descriptors over first {t_end:,} minutes...")
    N_all = counts.shape[0]
    total = np.zeros(N_all, dtype=np.int64)
    nz_frac = np.zeros(N_all)
    fano = np.zeros(N_all)
    iat_cv = np.zeros(N_all)
    e_daily = np.zeros(N_all)
    e_weekly = np.zeros(N_all)
    med_active = np.zeros(N_all)   # median invocations per ACTIVE minute

    CH = 1024
    n_days = t_end // MIN_PER_DAY
    for s in range(0, N_all, CH):
        e = min(s + CH, N_all)
        c = np.asarray(counts[s:e, :t_end], dtype=np.float64)
        total[s:e] = c.sum(axis=1).astype(np.int64)
        nz = c > 0
        nz_frac[s:e] = nz.mean(axis=1)
        mean = c.mean(axis=1)
        var = c.var(axis=1)
        fano[s:e] = np.where(mean > 0, var / np.maximum(mean, 1e-12), 0)
        F = np.abs(np.fft.rfft(c, axis=1)) ** 2
        tot_e = F[:, 1:].sum(axis=1) + 1e-12
        e_daily[s:e] = F[:, n_days] / tot_e            # daily cycles
        e_weekly[s:e] = F[:, max(n_days // 7, 1)] / tot_e
        for i in range(s, e):
            row = c[i - s]
            nzi = np.flatnonzero(row)
            if len(nzi):
                med_active[i] = float(np.median(row[nzi]))
            if total[i] >= MIN_INVOCATIONS and len(nzi) > 2:
                iats = np.diff(nzi)
                m = iats.mean()
                iat_cv[i] = iats.std() / m if m > 0 else 0
        if (s // CH) % 2 == 0:
            print(f"  chunk {s:,}/{N_all:,}")
    return total, nz_frac, fano, iat_cv, e_daily, e_weekly, med_active


# ------------------------------------------------------------------
def private_duration_pool():
    """(mean, std) execution seconds per private function + its volume.

    function_delay_minute holds the per-minute mean execution time in ms.
    Per function we pool the observed minute means: dur_mean = mean over
    observed minutes, dur_std = std over observed minutes (the same
    dispersion proxy the Azure path derives from the p25/p75 spread).
    """
    print("Building duration pool from Huawei Private...")
    n_days = len(sorted(PRIV_DELAY.glob("day_*.csv")))
    s1 = None
    for i, f in enumerate(sorted(PRIV_DELAY.glob("day_*.csv"))):
        a = pd.read_csv(f).iloc[:, 2:].to_numpy(dtype=np.float64)
        if s1 is None:
            n = a.shape[1]
            s1 = np.zeros(n); s2 = np.zeros(n); cnt = np.zeros(n)
        m = ~np.isnan(a)
        s1 += np.where(m, a, 0.0).sum(axis=0)
        s2 += (np.where(m, a, 0.0) ** 2).sum(axis=0)
        cnt += m.sum(axis=0)
        if i % 40 == 0:
            print(f"  delay day {i}/{n_days}")
    ok = cnt > 0
    mean_ms = np.where(ok, s1 / np.maximum(cnt, 1), np.nan)
    var_ms = np.where(ok, s2 / np.maximum(cnt, 1) - mean_ms ** 2, np.nan)
    std_ms = np.sqrt(np.maximum(var_ms, 0.0))

    vol = np.zeros_like(mean_ms)
    for i, f in enumerate(sorted(PRIV_REQ.glob("day_*.csv"))):
        a = pd.read_csv(f).iloc[:, 2:].to_numpy(dtype=np.float64)
        vol += np.nansum(a, axis=0)
        if i % 40 == 0:
            print(f"  requests day {i}")

    keep = ok & np.isfinite(mean_ms) & (mean_ms > 0)
    dur_mean = np.clip(mean_ms[keep] / 1000.0, 0.01, 600.0)
    dur_std = np.clip(np.nan_to_num(std_ms[keep], nan=0.0) / 1000.0, 0.01, 300.0)
    vol_keep = vol[keep]

    from scipy import stats as sp
    rho, p = sp.spearmanr(vol_keep, dur_mean)
    print(f"  {keep.sum()} private functions; duration median "
          f"{np.median(dur_mean):.3f}s, p90 {np.quantile(dur_mean, 0.9):.3f}s; "
          f"Spearman(volume, mean duration) = {rho:.3f} (p={p:.3g})")
    return dur_mean, dur_std, vol_keep, float(rho), float(p)


def assign_durations(total_pub, dur_mean, dur_std, vol_priv, rho):
    """Map the private duration pool onto public functions (registered rule)."""
    rng = np.random.default_rng(DUR_SEED)
    n = len(total_pub)
    if abs(rho) >= DUR_RANK_RHO:
        mode = "rank-matched"
        # public volume percentile -> private volume percentile
        order_priv = np.argsort(vol_priv)
        pct_pub = (np.argsort(np.argsort(total_pub)) + 0.5) / n
        idx = order_priv[np.clip((pct_pub * len(vol_priv)).astype(int),
                                 0, len(vol_priv) - 1)]
    else:
        mode = "iid"
        idx = rng.integers(0, len(dur_mean), size=n)
    print(f"  duration assignment: {mode} (|rho|={abs(rho):.3f} vs "
          f"threshold {DUR_RANK_RHO})")
    return dur_mean[idx], dur_std[idx], mode


# ------------------------------------------------------------------
def main():
    t_start = time.time()
    counts, func_ids = load_public_counts()
    N_all = counts.shape[0]

    (total, nz_frac, fano, iat_cv, e_daily, e_weekly,
     med_active) = descriptors(counts, STEADY_T)
    total_full = np.asarray(counts).sum(axis=1, dtype=np.int64)

    rate = total / STEADY_T
    active_mask = total >= MIN_INVOCATIONS
    n_active = int(active_mask.sum())
    print(f"Universe {N_all:,}; active(>= {MIN_INVOCATIONS} in 14d): "
          f"{n_active:,}; sparse remainder: {N_all - n_active:,}")

    # ---- density pools (registered frequency buckets) ----
    rng = np.random.default_rng(42)
    pools_full = {
        "sparse": np.flatnonzero(active_mask & (rate < 1 / 60)),
        "mixed": np.flatnonzero(active_mask & (rate >= 1 / 60) & (rate < 1.0)),
        "saturated": np.flatnonzero(active_mask & (rate >= 1.0)),
    }
    pools = {}
    for k, idx in pools_full.items():
        if len(idx) > POOL_CAP:
            idx = np.sort(rng.choice(idx, POOL_CAP, replace=False))
        pools[k] = idx
        print(f"  pool {k:9s}: {len(pools_full[k]):5,} functions "
              f"-> {len(idx):4,} sampled; median inv/active-min "
              f"{np.median(med_active[pools_full[k]]) if len(pools_full[k]) else 0:.1f}")

    # ---- durations ----
    dm_pool, ds_pool, vol_priv, rho, p_rho = private_duration_pool()
    dur_mean, dur_std, dur_mode = assign_durations(
        total, dm_pool, ds_pool, vol_priv, rho)
    pd.DataFrame({"dur_mean": dur_mean, "dur_std": dur_std}).to_csv(
        OUT / "duration_stats.csv", index=False)
    # alternates for the WP-H4 sensitivity (same row order)
    np.save(OUT / "dur_alt_iid.npy", np.stack(assign_durations(
        total, dm_pool, ds_pool, vol_priv, 0.0)[:2]))

    # ---- features (app_corate = 0: no app grouping in this release) ----
    print("Writing features memmap...")
    feats = np.lib.format.open_memmap(
        OUT / "features.npy", mode="w+", dtype=np.float32, shape=(N_all, T, 11))
    for s in range(0, N_all, FEAT_CHUNK):
        e = min(s + FEAT_CHUNK, N_all)
        feats[s:e] = features_from_counts(np.asarray(counts[s:e]))
        feats.flush()
        time.sleep(IO_PAUSE)
        if (s // FEAT_CHUNK) % 4 == 0:
            print(f"  features {e:,}/{N_all:,}")
    feats.flush()

    # ---- splits.npz: pool manifest (no meta-training on this trace) ----
    np.savez(OUT / "splits.npz",
             h_mixed=pools["mixed"], h_sparse=pools["sparse"],
             h_saturated=pools["saturated"],
             h_mixed_all=pools_full["mixed"],
             h_sparse_all=pools_full["sparse"],
             h_saturated_all=pools_full["saturated"],
             steady_t=np.array([STEADY_T]))

    # ---- descriptor CSVs (task_builder-compatible schema) ----
    def make_df(indices):
        rows = []
        for gi in np.asarray(indices):
            rows.append({
                "func_id": str(func_ids[gi]),
                "app_id": "",                       # no app grouping
                "total_invocations": float(total[gi]),
                "mean_rate_per_min": float(rate[gi]),
                "iat_cv": float(iat_cv[gi]),
                "fano_factor": float(fano[gi]),
                "sparsity": float(1.0 - nz_frac[gi]),
                "daily_spectral_energy": float(e_daily[gi]),
                "weekly_spectral_energy": float(e_weekly[gi]),
                "median_inv_per_active_min": float(med_active[gi]),
                "mean_duration": float(dur_mean[gi]),
                "n_invocations_in_trace": int(total_full[gi]),
                "freq_bucket": bucket_of(rate[gi]),
                "trigger": "",                      # not published
            })
        return pd.DataFrame(rows)

    make_df(np.flatnonzero(active_mask)).to_csv(
        OUT / "active_functions.csv", index=False)
    sparse_idx = np.flatnonzero(~active_mask & (total_full > 0))
    make_df(sparse_idx).to_csv(OUT / "sparse_functions.csv", index=False)

    # ---- T1 analogue ----
    stats = []
    for name, lo, hi in BUCKETS:
        m_all = (rate >= lo) & (rate < hi) & (total_full > 0)
        stats.append({
            "frequency_bucket": name,
            "total_functions": int(m_all.sum()),
            "active_functions": int((m_all & active_mask).sum()),
            "sparse_functions": int((m_all & ~active_mask).sum()),
            "total_invocations": int(total[m_all].sum()),
            "mean_rate_per_min": float(rate[m_all].mean()) if m_all.any() else 0,
            "mean_sparsity": float(1 - nz_frac[m_all].mean()) if m_all.any() else 0,
            "mean_fano": float(fano[m_all].mean()) if m_all.any() else 0,
        })
    stats.append({
        "frequency_bucket": "TOTAL",
        "total_functions": int((total_full > 0).sum()),
        "active_functions": n_active,
        "sparse_functions": int((~active_mask & (total_full > 0)).sum()),
        "total_invocations": int(total.sum()),
        "mean_rate_per_min": float(rate[total_full > 0].mean()),
        "mean_sparsity": float(1 - nz_frac[total_full > 0].mean()),
        "mean_fano": float(fano[total_full > 0].mean()),
    })
    t1 = pd.DataFrame(stats)
    t1.to_csv(TABLES / "T1_dataset_stats_huawei.csv", index=False)
    print(t1.to_string(index=False))

    meta = {
        "trace": "huawei_public_2023",
        "n_universe": int(N_all),
        "T_full": int(T), "T_steady": int(STEADY_T),
        "n_active_steady": n_active,
        "total_invocations_steady": int(total.sum()),
        "total_invocations_full": int(total_full.sum()),
        "pools": {k: {"n_pool": int(len(pools_full[k])),
                      "n_sampled": int(len(pools[k])),
                      "median_inv_per_active_min":
                          float(np.median(med_active[pools_full[k]]))
                          if len(pools_full[k]) else 0.0}
                  for k in pools},
        "duration_model": {"source": "huawei_private_2023/function_delay_minute",
                           "mode": dur_mode, "spearman_vol_dur": rho,
                           "spearman_p": p_rho, "threshold": DUR_RANK_RHO,
                           "seed": DUR_SEED,
                           "median_dur_s": float(np.median(dm_pool))},
        "deviations": ["app_corate channel identically zero (no app grouping)",
                       "durations imputed from the private release"],
    }
    with open(OUT / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nPhase 1c COMPLETE in {(time.time()-t_start)/60:.1f} min")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
