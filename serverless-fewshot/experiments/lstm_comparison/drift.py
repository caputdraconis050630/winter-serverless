"""Frozen R23 event cohorts with causal features and common-event replay inputs.

The original R23 encoder included the current minute in phi[t]. This export
uses phi[t] built strictly from completed minutes, and zero-start EWMA. The
scheduled adapter has no access to the drift onset. Frozen-at-onset is a
mechanistic diagnostic only. No event is reselected from new policy results.
"""
import argparse
import numpy as np
import pandas as pd
import torch

from cases import (RUNS, ROOT, OUT, begin, finish, basic_actions, lstm_actions,
                   rate_actions, rates_ewma, read_json, save_npz, digest)
from scripts.revision_r23_drift_v2 import (integer_base, apply_synthetic_drift,
    make_segment_features, embed_features, load_trainer, winter_rates)


def causal_embeddings(trainer, feats):
    past = np.zeros_like(feats)
    past[1:] = feats[:-1]
    return embed_features(trainer, past)


def prepare(provider, source):
    suffix = "" if provider == "azure2021" else "_" + provider
    archive_path = RUNS / f"revision_r23_drift_v2{suffix}.json"
    archive = read_json(archive_path)
    proc = ROOT / "data" / dict(azure2021="processed", azure2019="processed_2019", huawei="processed_huawei")[provider]
    counts = np.load(proc / "counts.npy", mmap_mode="r")
    features = np.load(proc / "features.npy", mmap_mode="r")
    rawkeys = np.load(proc / "func_ids.npy", allow_pickle=True).astype(str)
    rawapps = (pd.read_csv(proc / "active_functions.csv").app_id.to_numpy().astype(str)
               if provider != "huawei" and (proc / "active_functions.csv").exists() else rawkeys)
    stats = pd.read_csv(proc / "duration_stats.csv")
    # Reproduce the duration row resolution without allocating F x T constants.
    means = np.full(len(counts), float(stats.dur_mean.median()))
    stds = np.full(len(counts), max(float(stats.dur_std.median()), .001))
    for index, row in stats.iterrows():
        try:
            f = int(row.func_id) if "func_id" in stats.columns else index
        except (ValueError, TypeError):
            f = index
        if 0 <= f < len(counts):
            means[f], stds[f] = max(float(row.dur_mean), .001), max(float(row.dur_std), .001)
    configs = []
    for i, event in enumerate(archive[source + "_events"]):
        kinds = archive["synthetic_kinds"] if source == "synthetic" else [event["kind"]]
        for k, kind in enumerate(kinds):
            configs.append((event, kind, 23023 + i*100 + k))
    series_all, feats_all, keys, ids = [], [], [], []
    records = []
    for event, kind, seed in configs:
        f, start = event["func_id"], event["onset"]-1440
        series = integer_base(counts[f,start:start+1680])
        if source == "synthetic":
            series = apply_synthetic_drift(series, 1440, kind, np.random.default_rng(seed))
            feats = make_segment_features(series, features[f,start:start+1680,10].astype(np.float32), start)
        else:
            feats = features[f,start:start+1680].astype(np.float32)
        series_all.append(series)
        feats_all.append(feats)
        ids.append(f)
        keys.append(event["event_id"] + ":" + kind)
        records.append(dict(event, replay_kind=kind, transform_seed=seed))
    ids = np.array(ids)
    data = dict(counts=np.stack(series_all), ids=ids, keys=np.array(keys), apps=rawapps[ids],
                weights=np.ones(len(ids)), duration_mean=means[ids], duration_std=stds[ids])
    split = "s1" if provider == "azure2021" else "s2"
    directory, meta = begin(f"drift_{provider}_{source}", data, dict(surface="drift", provider=provider,
        drift_source=source, source=str(archive_path), source_sha256=digest(archive_path),
        seeds=[0,1,2], rhos=[1.,10.], onset_minute=1440, post_minutes=240, events=records,
        event_key_prefix=f"drift:{provider}:", model_source="Azure2021 " + split.upper(),
        encoder_input="60 completed minutes ending at t-1; no current count",
        primary_adapter="scheduled every 10 min, buffer 120, scalar ridge lambda .01; no onset input",
        diagnostic_adapter="same adapter frozen at known onset, diagnostic only"))
    lstm_actions(directory, meta, data, split=split, rhos=(1.,10.))
    basic_actions(directory, meta, data, rhos=(1.,10.))
    rate_actions(directory, meta, "EWMA_0.5", rates_ewma(data["counts"], .5), (1.,10.), "causal log-count EWMA sensitivity")
    checkpoint = ROOT / "results_azure2021/runs" / f"best_anil_ridge_{split}_s0.pt"
    path = directory / "winter_rates.npz"
    if path.exists():
        with np.load(path) as z:
            scheduled, frozen = z["scheduled"], z["frozen"]
    else:
        trainer = load_trainer("cuda" if torch.cuda.is_available() else "cpu", 11, checkpoint)
        scheduled, frozen = np.zeros(data["counts"].shape, np.float32), np.zeros(data["counts"].shape, np.float32)
        for f, feats in enumerate(feats_all):
            phi = causal_embeddings(trainer, feats)
            for mode, dest in (("scheduled", scheduled), ("frozen", frozen)):
                dest[f], _ = winter_rates(phi, data["counts"][f], 1440, mode, .01, 10, 120, .1, 30, 120)
            if (f+1) % 100 == 0:
                print(f"drift {provider} {source}: encoded {f+1}/{len(ids)}", flush=True)
        save_npz(path, scheduled=scheduled, frozen=frozen)
    provenance = dict(file=str(path), sha256=digest(path), checkpoint_sha256=digest(checkpoint))
    rate_actions(directory, meta, "WINTER", scheduled, (1.,10.), provenance)
    rate_actions(directory, meta, "WINTER_frozen", frozen, (1.,10.), provenance)
    finish(directory, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("provider", choices=["azure2021", "azure2019", "huawei", "all"])
    args = ap.parse_args()
    for provider in (["azure2021", "azure2019", "huawei"] if args.provider == "all" else [args.provider]):
        for source in ("natural", "synthetic"):
            prepare(provider, source)
