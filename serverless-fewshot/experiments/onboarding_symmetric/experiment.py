"""Versioned controller-symmetric follow-up; historical artifacts are read-only."""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD_CODE = HERE.parent / "onboarding_attribution"
sys.path.insert(0, str(OLD_CODE))
from common import digest, load_cohort, read_json, require_tests, save_npz, write_json
from policies import decisions, actions, native_baselines
from simulator import certificate, reuse_certificate, simulate
from study import action_hash, get_tape, priors

OLD = ROOT / "results/onboarding_attribution_v1"
OUT = ROOT / "results/onboarding_symmetric_v2"
COHORTS = ("azure_calibration", "azure_primary", "azure_evaluation", "huawei")
RHOS = (10., 1., 100.)
FAMILIES = ("component", "gate", "no_prior", "pooled_prior", "trained_lambda",
            "random0", "random1", "random2", "random3", "random4")
MULTIPLIERS = (.5, .75, 1., 1.25, 1.5, 2.)
GRID = [dict(scale=s, floor=k, ttl_factor=h, horizon=H,
             id=f"s{s:g}_k{k}_h{h:g}_H{H}")
        for s, k, h, H in itertools.product((.5, 1., 2., 4.), (0, 1, 2, 4, 8),
                                           (1., 2., 4.), (15, 60, 240))]
SEED_POOL = ThreadPoolExecutor(max_workers=2)


def frozen_json(path, value):
    if path.exists() and read_json(path) != value:
        raise RuntimeError(f"Frozen artifact differs: {path}")
    if not path.exists():
        write_json(path, value)


def prepare():
    require_tests()
    OUT.mkdir(parents=True, exist_ok=True)
    files = [OLD / name for name in ("protocol.json", "manifest.json", "selection.json",
             "representation_selection.json", "simple_priors.json", "simulation_contract.json")]
    files += list((OLD / "cohorts").glob("*.npz"))
    files += list((OLD / "models").glob("*/*_rates.npz"))
    files += list((OLD / "models/trained0").glob("*.npz"))
    files += [OLD / f"calibration_scores_rho{rho:g}.npz" for rho in RHOS]
    files += [OLD / f"screen_proofs_rho{rho:g}.json" for rho in RHOS]
    files += [OLD / "candidate_configs.json"]
    files += [OLD_CODE / name for name in ("common.py", "policies.py", "simulator.py",
              "study.py", "models.py", "analyze.py")]
    manifest = {str(p.relative_to(ROOT)): digest(p) for p in sorted(set(files))}
    frozen_json(OUT / "inputs.json", manifest)
    protocol = dict(version=2, question="Additional value after common controller calibration",
        cohorts=list(COHORTS), source="Azure2021 S2 seed0, C16; existing trace cohorts already inspected",
        families=list(FAMILIES), grid=GRID, rhos=list(RHOS), primary_rho=10,
        calibration_seeds=list(range(5)), evaluation_seeds=list(range(1000, 1020)),
        budgets="Six absolute in-window idle-GBs limits from v1 native component on calibration",
        multipliers=list(MULTIPLIERS), selection="minimum exact sum of five-seed cold counts; then WM; then ID",
        baseline="retain v1 exhaustive 3701-candidate winner at each identical absolute budget",
        representations="freeze v1 budget-1 selected lambda per representation and rho; no joint retuning",
        priors="component/no_prior/pooled_prior share native lambda and body; matched source-support union",
        conditions=["main", "strict"], strict="q[0]=0, TTL[0]=10 for every arm; no reselection",
        primary_tests="six two-sided paired cold contrasts: component vs strong simple at budget1; three cohorts x two starts; Holm6",
        secondary_tests="nine main-start rho10 budget1 contrasts: component/no_prior, component/pooled, trained_lambda/mean5random; Holm9",
        intervals="4h primary; first15/60 descriptive; drain idle and modeled cost both mandatory",
        resources="actual evaluation WM not hard quota; no equivalence without CI within [.99,1.01]",
        scope="controller calibration only, no meta retraining, no mature/48h/drift revalidation",
        simulator=digest(OLD_CODE / "simulator.py"), timing="prospective TTL; common semantic event tapes",
        non_blind="Follow-up designed after seeing v1 results; no claims of external preregistration")
    frozen_json(OUT / "protocol.json", protocol)
    print("Frozen inputs", len(manifest), "controller settings", len(GRID), flush=True)


def transform(q, ttl, config):
    q, ttl = q.copy(), ttl.copy()
    h = config["horizon"]
    q[..., :h] = np.minimum(200, np.ceil(q[..., :h] * config["scale"]) + config["floor"])
    ttl[..., :h] = np.minimum(240, ttl[..., :h] * config["ttl_factor"])
    return q, ttl


def rates(cohort, rho):
    with np.load(OLD / "models/trained0" / f"{cohort}_rates.npz") as z:
        result = {key: z[key] for key in ("component", "gate", "no_prior")}
    result["pooled_prior"] = np.load(OUT / "models" / f"{cohort}_pooled.npz")["rates"]
    selected = read_json(OLD / "representation_selection.json")[str(float(rho))]
    for name in ("trained_lambda", "random0", "random1", "random2", "random3", "random4"):
        model = "trained0" if name == "trained_lambda" else name
        index = next(x["lambda_index"] for x in selected[model] if x["multiplier"] == 1.)
        result[name] = np.load(OLD / "models" / model / f"{cohort}_rates.npz")["lambda_rates"][index]
    return result


def budgets(rho):
    choices = read_json(OLD / "selection.json")[str(float(rho))]["component"]["choices"]
    return np.array([x["wm_limit"] for x in choices])


def can_improve(cold, wm, limits, best):
    return bool(np.any((wm <= limits + 1e-7) & (cold <= best + 1e-7)))


def values_for(tapes, q, ttl, memo, certs, use_cert):
    key = action_hash(q, ttl)
    if key not in memo:
        initial = int(q[0])
        if use_cert and initial not in certs:
            certs[initial] = [certificate(t, initial) for t in tapes]
        def one(item):
            i, tape = item
            value = reuse_certificate(certs[initial][i], q, ttl) if initial in certs else None
            if value is None:
                value = simulate(*tape, q.astype(np.int64), ttl.astype(np.float64))
            return value
        values = list(SEED_POOL.map(one, enumerate(tapes))) if use_cert else [one(item) for item in enumerate(tapes)]
        memo[key] = np.mean(values, axis=0)
    return memo[key]


def screen(rho, workers=3, benchmark=False):
    require_tests()
    z = load_cohort("azure_calibration")
    base = {name: decisions(value, rho) for name, value in rates("azure_calibration", rho).items()}
    directory = OUT / "screen" / f"rho{rho:g}"
    directory.mkdir(parents=True, exist_ok=True)
    cheap = np.flatnonzero(z["counts"].sum(1) <= 10000).tolist()
    if benchmark:
        cheap = sorted(cheap, key=lambda f: (z["counts"][f].sum(), f))
        cheap = [cheap[i] for i in np.linspace(0, len(cheap)-1, 8, dtype=int)]
    def worker(f):
        start = time.monotonic()
        path = directory / f"f{f:04d}.npz"
        if path.exists():
            return f, 0., 0
        tapes = [get_tape("azure_calibration", z, f, s) for s in range(5)]
        memo, certs = {}, {}
        out = np.empty((len(FAMILIES), len(GRID), 5, 8))
        for fi, family in enumerate(FAMILIES):
            q, ttl = base[family]
            for ci, config in enumerate(GRID):
                qa, ta = transform(q[f], ttl[f], config)
                out[fi, ci] = values_for(tapes, qa, ta, memo, certs, False)
        save_npz(path, metrics=out, unique=np.array(len(memo)))
        return f, time.monotonic()-start, len(memo)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(worker, f) for f in cheap]
        for k, job in enumerate(as_completed(jobs)):
            f, sec, unique = job.result()
            if benchmark or (k+1) % 25 == 0:
                print("screen", rho, k+1, "/", len(cheap), "f", f, "sec", round(sec, 2), "unique", unique, flush=True)
    print("screen complete", rho, "seconds", time.monotonic()-start, flush=True)


def select(rho):
    with open(OUT / f"selection_rho{rho:g}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _select(rho)


def _select(rho):
    require_tests()
    dest = OUT / f"selection_rho{rho:g}.json"
    if dest.exists():
        print("selection already frozen", rho, flush=True)
        return
    z = load_cohort("azure_calibration")
    cheap = np.flatnonzero(z["counts"].sum(1) <= 10000).tolist()
    lower = np.zeros((len(FAMILIES), len(GRID), 5, 8))
    for f in cheap:
        lower += np.load(OUT / "screen" / f"rho{rho:g}" / f"f{f:04d}.npz")["metrics"]
    reference = np.load(OLD / "evaluation/azure_calibration" / f"native_rho{rho:g}/component.npz")["metrics"]
    cold_order = reference[:, :, :4, 1].sum(2).mean(1)
    remaining = sorted(set(range(len(z["counts"]))) - set(cheap), key=lambda f: (-cold_order[f], f))
    limits = budgets(rho)
    base = {name: decisions(value, rho) for name, value in rates("azure_calibration", rho).items()}
    tapes, memo, certs, unavoidable = {}, {}, {}, {}
    cache_dir = OUT / "large_cache" / f"rho{rho:g}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for f in remaining:
        tapes[f] = [get_tape("azure_calibration", z, f, s) for s in range(5)]
        memo[f], certs[f] = {}, {}
        # Reuse exact action values from the previous identical calibration DES.
        for cache in (OLD / "screen_cache" / f"f{f:04d}.npz",
                      OLD / "screen_cache" / f"rho{rho:g}" / f"f{f:04d}.npz",
                      cache_dir / f"f{f:04d}.npz"):
            if cache.exists():
                with np.load(cache) as data:
                    memo[f].update(zip(data["keys"].tolist(), data["values"]))
        unavoidable[f] = np.mean([np.count_nonzero(t[1] < min(float(np.min(t[4] + np.arange(240)[:, None]*60.)),
                                                float(np.min(t[1]+t[3])))) for t in tapes[f]])
    selection, proofs = {}, {}
    for fi, family in enumerate(FAMILIES):
        best = np.full(len(limits), np.inf)
        full, rejected = {}, []
        directory = OUT / "complete" / f"rho{rho:g}" / family
        directory.mkdir(parents=True, exist_ok=True)
        totals = lower[fi, :, :4].sum(1)
        def update(ci, m):
            cold = round(5*m[:4, 1].sum())/5
            wm = m[:4, 2].sum()
            best[wm <= limits] = np.minimum(best[wm <= limits], cold)
            full[ci] = m
        def evaluate(ci, prune):
            target = directory / (GRID[ci]["id"] + ".npz")
            if target.exists():
                update(ci, np.load(target)["metrics"])
                return
            m = lower[fi, ci].copy()
            q, ttl = transform(*base[family], GRID[ci])
            tail = sum(unavoidable.values())
            for f in remaining:
                cb, wb = float(m[:4, 1].sum()+tail), float(m[:4, 2].sum())
                if prune and not can_improve(cb, wb, limits, best):
                    rejected.append(dict(index=ci, cold_lower_bound=cb, wm_lower_bound=wb))
                    return
                value = values_for(tapes[f], q[f], ttl[f], memo[f], certs[f], True)
                if value[:4, 1].sum()+1e-7 < unavoidable[f]:
                    raise AssertionError("Invalid unavoidable-cold bound")
                m += value
                tail -= unavoidable[f]
            update(ci, m)
            save_npz(target, metrics=m)
        anchors = {next(i for i, c in enumerate(GRID) if c == dict(scale=1., floor=0, ttl_factor=1., horizon=240, id="s1_k0_h1_H240"))}
        for limit in limits:
            feasible = np.flatnonzero(totals[:, 2] <= limit)
            anchors.update(sorted(feasible, key=lambda i: (totals[i, 1], totals[i, 2], GRID[i]["id"]))[:2])
        for ci in sorted(anchors):
            evaluate(ci, False)
            print("anchor", rho, family, GRID[ci]["id"], flush=True)
        for counter, ci in enumerate(sorted(range(len(GRID)), key=lambda i: (totals[i, 1], totals[i, 2], GRID[i]["id"]))):
            if ci not in full:
                evaluate(ci, True)
            if (counter+1) % 20 == 0:
                print("bounds", rho, family, counter+1, "full", len(full), "pruned", len(rejected), flush=True)
        choices = []
        for multiplier, limit in zip(MULTIPLIERS, limits):
            eligible = [i for i, m in full.items() if m[:4, 2].sum() <= limit]
            chosen = min(eligible, key=lambda i: (round(5*full[i][:4, 1].sum()), full[i][:4, 2].sum(), GRID[i]["id"])) if eligible else None
            choices.append(dict(multiplier=multiplier, wm_limit=float(limit),
                                config=GRID[chosen] if chosen is not None else None,
                                metrics=full[chosen].tolist() if chosen is not None else None))
        for item in rejected:
            if can_improve(item["cold_lower_bound"], item["wm_lower_bound"], limits, best):
                raise AssertionError("Pruning certificate failed")
        if len(full) + len(rejected) != len(GRID):
            raise AssertionError("Incomplete candidate accounting")
        selection[family] = choices
        proofs[family] = dict(fully_evaluated=[int(i) for i in sorted(full)], rejected=rejected,
                              best_cold=[float(x) if np.isfinite(x) else None for x in best])
        for f in remaining:
            save_npz(cache_dir / f"f{f:04d}.npz", keys=np.array(list(memo[f])), values=np.stack(list(memo[f].values())))
        print("selected", rho, family, "full", len(full), "pruned", len(rejected), flush=True)
    selection["strong_simple"] = read_json(OLD / "selection.json")[str(float(rho))]["component"]["choices"]
    write_json(OUT / f"proofs_rho{rho:g}.json", dict(budgets=limits.tolist(), families=proofs))
    frozen_json(dest, selection)


def selected_actions(cohort, rho):
    base = {name: decisions(value, rho) for name, value in rates(cohort, rho).items()}
    selected = read_json(OUT / f"selection_rho{rho:g}.json")
    result = {}
    for family in FAMILIES:
        result[family + "__native"] = base[family]
        for i, choice in enumerate(selected[family]):
            if choice["config"] is not None:
                result[family + f"__b{i}"] = transform(*base[family], choice["config"])
    # Fixed-wrapper substitutions distinguish forecast changes from retuning.
    for reference, families in (("component", ("no_prior", "pooled_prior")),
                                ("trained_lambda", tuple(f"random{s}" for s in range(5)))):
        config = selected[reference][2]["config"]
        if config is not None:
            for family in families:
                result[f"{family}__{reference}_wrapper"] = transform(*base[family], config)
    z = load_cohort(cohort)
    fleet, profile = priors()
    native = native_baselines(z["counts"], rho, fleet, profile)
    result.update({name: native[name] for name in ("ewma", "fleetprior", "fixed_keepalive", "ageka60", "b2f")})
    cache = {}
    for i, choice in enumerate(selected["strong_simple"]):
        if choice["config"] is not None:
            result[f"strong_simple__b{i}"] = actions(choice["config"], z["counts"], rho, fleet, native, cache)
    return result


def evaluate(rho, workers=3):
    require_tests()
    lock = read_json(OUT / "evaluation_lock.json")
    for name, expected in lock.items():
        if digest(OUT / name) != expected:
            raise RuntimeError("Selected policy changed after evaluation lock")
    if len(lock) != len(RHOS):
        raise RuntimeError("Every rho selection must be locked before evaluation")
    for name, expected in read_json(OUT / "evaluation_code_lock.json").items():
        if digest(HERE / name) != expected:
            raise RuntimeError("Action-generating code changed after evaluation lock")
    for cohort in COHORTS[1:]:
        z = load_cohort(cohort)
        policies = selected_actions(cohort, rho)
        for condition in ("main", "strict"):
            directory = OUT / "evaluation" / cohort / f"{condition}_rho{rho:g}"
            directory.mkdir(parents=True, exist_ok=True)
            aa = {}
            for name, (q, ttl) in policies.items():
                q, ttl = q.copy(), ttl.copy()
                if condition == "strict":
                    q[:, 0], ttl[:, 0] = 0, 10.
                aa[name] = (q, ttl)
            def worker(f):
                path = directory / f"f{f:04d}.npz"
                q = np.stack([a[0][f] for a in aa.values()])
                ttl = np.stack([a[1][f] for a in aa.values()])
                if path.exists():
                    with np.load(path) as cached:
                        np.testing.assert_array_equal(cached["q"], q)
                        np.testing.assert_array_equal(cached["ttl"], ttl)
                        np.testing.assert_array_equal(cached["names"], list(aa))
                    return
                tapes = [get_tape(cohort, z, f, seed) for seed in range(1000, 1020)]
                memo, certs = {}, {}
                out = []
                for name, (qa, ta) in aa.items():
                    key = action_hash(qa[f], ta[f])
                    if key not in memo:
                        initial = int(qa[f, 0])
                        if z["counts"][f].sum() > 10000 and initial not in certs:
                            certs[initial] = [certificate(t, initial) for t in tapes]
                        values = []
                        for si, tape in enumerate(tapes):
                            value = reuse_certificate(certs[initial][si], qa[f], ta[f]) if initial in certs else None
                            if value is None:
                                value = simulate(*tape, qa[f].astype(np.int64), ta[f].astype(float))
                            values.append(value)
                        memo[key] = np.stack(values)
                    out.append(memo[key])
                save_npz(path, metrics=np.stack(out), names=np.array(list(aa)), q=q, ttl=ttl,
                         seeds=np.arange(1000, 1020))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = [pool.submit(worker, f) for f in range(len(z["counts"]))]
                for k, job in enumerate(as_completed(jobs)):
                    job.result()
                    if (k+1) % 50 == 0:
                        print("evaluate", rho, cohort, condition, k+1, "/", len(z["counts"]), flush=True)
            frozen_json(directory / "complete.json", dict(n=len(z["counts"]), names=list(aa), seeds=list(range(1000, 1020))))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("prepare", "benchmark", "screen", "select", "evaluate"))
    ap.add_argument("--rho", type=float, choices=RHOS, default=10.)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage in ("screen", "benchmark"):
        screen(args.rho, args.workers, args.stage == "benchmark")
    elif args.stage == "select":
        select(args.rho)
    else:
        evaluate(args.rho, args.workers)
