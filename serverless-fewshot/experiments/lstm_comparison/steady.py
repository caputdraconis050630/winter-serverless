"""All nine retained active pools, with their exact rows and sampling weights."""
import argparse
import sys
import numpy as np
import pandas as pd
import torch

from cases import (RUNS, ROOT, OUT, begin, finish, action, cached_action,
                   basic_actions, lstm_actions, rate_actions, read_json, save_npz, digest)

RHOS = (.1, 1., 10., 100.)


def prepare(provider, split):
    pnames = dict(azure2021="processed", azure2019="processed_2019", huawei="processed_huawei")
    proc = ROOT / "data" / pnames[provider]
    splits = np.load(proc / "splits.npz")
    if provider == "azure2021":
        roots = [RUNS / "des_jobs_r26_azure2021" / split]
        ids = splits[split.lower() + "_test"]
        weights = np.ones(len(ids))
        offset = int(splits["s3_test_t_start"].reshape(-1)[0]) if split == "S3" else 0
    elif provider == "azure2019":
        manifest = read_json(RUNS / "revision_r21_azure2019_rho10_manifest.json")["splits"][split]
        ids, weights = np.array(manifest["rows"]), np.array(manifest["weights"])
        roots = [RUNS / "des_jobs_r21_azure2019_rho10" / split,
                 RUNS / "des_jobs_r21_azure2019_remaining" / split]
        offset = int(splits["s3_test_t_start"].reshape(-1)[0]) if split == "S3" else 0
    else:
        filename = "revision_r21_huawei_hsaturated_rho10_manifest.json" if split == "h_saturated" else "revision_r21_huawei_manifest.json"
        manifest = read_json(RUNS / filename)["splits"][split]
        ids, weights = np.array(manifest["rows"]), np.array(manifest["weights"])
        roots = [RUNS / "des_jobs_r21_huawei" / split] if split != "h_saturated" else [
            RUNS / "des_jobs_r21_huawei_hsaturated_rho10" / split,
            RUNS / "des_jobs_r21_huawei_hsaturated_remaining" / split]
        offset = 0
    with np.load(roots[0] / "shared.npz") as z:
        data = dict(counts=z["counts"], duration_mean=z["dur_means"], duration_std=z["dur_stds"])
    full = np.load(proc / "counts.npy", mmap_mode="r")
    if provider == "huawei":
        # Huawei's active pools use the first steady_t minutes.
        expected = full[ids, :data["counts"].shape[1]]
    else:
        expected = full[ids, offset:]
    np.testing.assert_array_equal(data["counts"], expected)
    keys = np.load(proc / "func_ids.npy", allow_pickle=True)[ids].astype(str)
    apps = (pd.read_csv(proc / "active_functions.csv").app_id.to_numpy()[ids].astype(str)
            if provider != "huawei" and (proc / "active_functions.csv").exists() else keys)
    data.update(ids=ids, keys=keys, apps=apps, weights=weights)
    directory, meta = begin(f"steady_{provider}_{split}", data, dict(surface="steady", provider=provider,
        split=split, source=[str(r) for r in roots], seeds=[0,1,2], rhos=list(RHOS),
        event_key_prefix=f"steady:{provider}:{split}:", model_source="Azure2021 S1",
        source_scope="same retained source checkpoint for all pools; split labels identify target pools",
        sampling="retained rows and inverse inclusion weights", offset_minutes=offset))
    lstm_actions(directory, meta, data, split="s1", rhos=RHOS)
    # These predictors are recomputed from the exact retained counts.
    basic_actions(directory, meta, data, rhos=RHOS)
    if provider == "azure2021":
        path = directory / "winter_rates.npz"
        if path.exists():
            rates = np.load(path)["rate"]
        else:
            from scripts.revision_r11_faithful import rates_a5_faithful
            from scripts.phase63_onboarding_drift import build_prototypes
            from src.meta.trainer import ANILMetaTrainer
            from src.models.heads import N_QUANTILES
            features = np.load(proc / "features.npy", mmap_mode="r")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge", in_features=features.shape[2],
                embedding_dim=64, n_quantiles=N_QUANTILES, n_horizons=1, device=device)
            trainer.load(ROOT / "results_azure2021/runs/best_anil_ridge_s1_s0.pt")
            pm = build_prototypes(trainer, features, full, splits["s1_train"])
            rates = rates_a5_faithful(data["counts"].astype(np.float32), features[ids,offset:], trainer, pm)
            save_npz(path, rate=rates)
        rate_actions(directory, meta, "WINTER", rates, RHOS, dict(file=str(path), sha256=digest(path)))
        for rho in RHOS:
            cached_action(directory, meta, "WINTER_G", rho, roots, "G_WE_A720_current")
        if split == "S3":
            from cases import decisions_from_rates, newsvendor_quantile
            cache = RUNS / "revision_e5d_chronos_native_forecasts.npz"
            with np.load(cache) as z:
                levels, grid = z["q_levels"], z["steady_q"]
            assert grid.shape[1:] == data["counts"].shape
            for rho in RHOS:
                tau = min(newsvendor_quantile(rho), float(levels[-1]))
                hi = min(max(int(np.searchsorted(levels, tau)), 1), len(levels)-1)
                blend = np.clip((tau-levels[hi-1])/(levels[hi]-levels[hi-1]), 0., 1.)
                rate = grid[hi-1]*(1-blend)+grid[hi]*blend
                _, ttl = decisions_from_rates(rate, newsvendor_quantile(rho))
                action(directory, meta, "Chronos", rho, np.ceil(rate-1e-9), ttl, dict(file=str(cache), sha256=digest(cache)))
    else:
        for rho in RHOS:
            for method, archived in (("WINTER", "A5_faithful"), ("WINTER_G", "G_WE_A720")):
                cached_action(directory, meta, method, rho, roots, archived)
    finish(directory, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("provider", choices=["azure2021", "azure2019", "huawei", "all"])
    ap.add_argument("--split")
    args = ap.parse_args()
    for provider in (["azure2021", "azure2019", "huawei"] if args.provider == "all" else [args.provider]):
        for split in ([args.split] if args.split else (["h_mixed", "h_sparse", "h_saturated"] if provider == "huawei" else ["S1", "S2", "S3"])):
            prepare(provider, split)
