"""Independent artifact audit: inputs, selection proofs, actions and conservation."""
import argparse
import numpy as np

from experiment import (COHORTS, FAMILIES, GRID, HERE, OLD, OUT, RHOS, ROOT, action_hash,
                        budgets, can_improve, decisions, digest, get_tape, load_cohort,
                        rates, read_json, save_npz, selected_actions, transform, values_for, write_json)


def main(calibration_only=False):
    for manifest in ("inputs.json", "source_inputs.json"):
        for name, expected in read_json(OUT / manifest).items():
            if digest(ROOT / name) != expected:
                raise AssertionError(f"Original input changed: {name}")
    lock = read_json(OUT / "evaluation_lock.json")
    for name, expected in lock.items():
        if digest(OUT / name) != expected:
            raise AssertionError("Selection changed after lock")
    for name, expected in read_json(OUT / "evaluation_code_lock.json").items():
        if digest(HERE / name) != expected:
            raise AssertionError("Action code changed after lock")
    if not read_json(OUT / "tests.json")["passed"]:
        raise AssertionError("Tests not passed")
    if not read_json(OUT / "reporting_tests.json")["passed"]:
        raise AssertionError("Reporting tests not passed")
    z = load_cohort("azure_calibration")
    cheap = np.flatnonzero(z["counts"].sum(1) <= 10000)
    large = np.flatnonzero(z["counts"].sum(1) > 10000)
    native_i = next(i for i, c in enumerate(GRID) if c["id"] == "s1_k0_h1_H240")
    checks = 0
    for rho in RHOS:
        chosen = read_json(OUT / f"selection_rho{rho:g}.json")
        proof = read_json(OUT / f"proofs_rho{rho:g}.json")
        limits = budgets(rho)
        np.testing.assert_array_equal(proof["budgets"], limits)
        accum = np.zeros((len(FAMILIES), len(GRID), 5, 8))
        original_native = {}
        for family in ("component", "gate", "no_prior"):
            original_native[family] = np.load(OLD / "evaluation/azure_calibration" / f"native_rho{rho:g}" / f"{family}.npz")["metrics"]
        for f in cheap:
            data = np.load(OUT / "screen" / f"rho{rho:g}" / f"f{f:04d}.npz")["metrics"]
            assert data.shape == accum.shape
            assert np.isfinite(data).all() and data.min() >= -1e-8
            np.testing.assert_allclose(data[:, :, :4, 0].sum(2), z["counts"][f].sum(), atol=1e-8)
            for i, family in enumerate(FAMILIES[:3]):
                np.testing.assert_allclose(data[i, native_i], original_native[family][f].mean(0), rtol=1e-10, atol=1e-6)
            accum += data
        base = {name: decisions(value, rho) for name, value in rates("azure_calibration", rho).items()}
        large_memo = {}
        audit_memo, audit_tapes, audit_certs = {}, {}, {}
        audit_dir = OUT / "verification_cache" / f"rho{rho:g}"
        for f in large:
            data = np.load(OUT / "large_cache" / f"rho{rho:g}" / f"f{f:04d}.npz")
            large_memo[f] = dict(zip(data["keys"].tolist(), data["values"]))
            audit_memo[f] = {}
            path = audit_dir / f"f{f:04d}.npz"
            if path.exists():
                with np.load(path) as data:
                    audit_memo[f].update(zip(data["keys"].tolist(), data["values"]))
        for fi, family in enumerate(FAMILIES):
            pp = proof["families"][family]
            full, scores = set(pp["fully_evaluated"]), {}
            rejected = {x["index"] for x in pp["rejected"]}
            assert not full & rejected and full | rejected == set(range(180))
            for ci in full:
                m = np.load(OUT / "complete" / f"rho{rho:g}" / family / (GRID[ci]["id"]+".npz"))["metrics"]
                np.testing.assert_allclose(m[:4, 0].sum(), z["counts"].sum(), atol=1e-6)
                scores[ci] = m
            best = []
            for bi, limit in enumerate(limits):
                eligible = [ci for ci, m in scores.items() if m[:4, 2].sum() <= limit]
                expected = min(eligible, key=lambda ci: (round(5*scores[ci][:4, 1].sum()), scores[ci][:4, 2].sum(), GRID[ci]["id"])) if eligible else None
                tolerant = [ci for ci, m in scores.items() if m[:4, 2].sum() <= limit+1e-7]
                robust = min(tolerant, key=lambda ci: (round(5*scores[ci][:4, 1].sum()), scores[ci][:4, 2].sum(), GRID[ci]["id"])) if tolerant else None
                assert robust == expected, "Winner depends on sub-micro-GBs feasibility rounding"
                choice = chosen[family][bi]
                assert choice["wm_limit"] == limit
                assert choice["config"] == (GRID[expected] if expected is not None else None)
                best.append(round(5*scores[expected][:4, 1].sum())/5 if expected is not None else np.inf)
                if expected is not None:
                    qa, ta = transform(*base[family], GRID[expected])
                    independent = accum[fi, expected].copy()
                    for f in large:
                        key = action_hash(qa[f], ta[f])
                        if key in large_memo[f]:
                            value = large_memo[f][key]
                        else:
                            # A restart may retain a complete total before the
                            # corresponding per-function cache was flushed.
                            if key not in audit_memo[f]:
                                if f not in audit_tapes:
                                    audit_tapes[f] = [get_tape("azure_calibration", z, f, seed) for seed in range(5)]
                                    audit_certs[f] = {}
                                values_for(audit_tapes[f], qa[f], ta[f], audit_memo[f], audit_certs[f], True)
                                save_npz(audit_dir / f"f{f:04d}.npz", keys=np.array(list(audit_memo[f])),
                                         values=np.stack(list(audit_memo[f].values())))
                            value = audit_memo[f][key]
                        independent += value
                    np.testing.assert_allclose(independent, scores[expected], rtol=1e-10, atol=1e-6)
                    np.testing.assert_allclose(choice["metrics"], scores[expected], rtol=0, atol=0)
                    checks += 1
            for p in pp["rejected"]:
                assert not can_improve(p["cold_lower_bound"], p["wm_lower_bound"], limits, np.array(best))
        assert chosen["strong_simple"] == read_json(OLD / "selection.json")[str(rho)]["component"]["choices"]
        print("verified calibration and exact bounds", rho, flush=True)
    if calibration_only:
        print("CALIBRATION VERIFIED", checks, "independently reaggregated selections", flush=True)
        return
    files = 0
    policy_replays = 0
    for cohort in COHORTS[1:]:
        z = load_cohort(cohort)
        # Identical exogenous durations mean exact all-tail execution GBs are policy invariant.
        execution = np.empty((len(z["counts"]), 20))
        for f in range(len(z["counts"])):
            for si, seed in enumerate(range(1000, 1020)):
                execution[f, si] = .25*get_tape(cohort, z, f, seed)[2].sum()
        for rho in RHOS:
            actions = selected_actions(cohort, rho)
            for condition in ("main", "strict"):
                directory = OUT / "evaluation" / cohort / f"{condition}_rho{rho:g}"
                meta = read_json(directory / "complete.json")
                assert meta["n"] == len(z["counts"])
                originals = {}
                for name in ("component", "gate", "no_prior", "ewma", "fleetprior", "fixed_keepalive", "ageka60", "b2f"):
                    originals[name+"__native" if name in ("component", "gate", "no_prior") else name] = np.load(OLD / "evaluation" / cohort / f"{condition}_rho{rho:g}" / (name+".npz"))["metrics"]
                selected = read_json(OUT / f"selection_rho{rho:g}.json")
                for bi, choice in enumerate(selected["strong_simple"]):
                    if choice["config"]:
                        name = choice["config"]["id"]
                        originals[f"strong_simple__b{bi}"] = np.load(OLD / "evaluation" / cohort / f"{condition}_rho{rho:g}" / (name+".npz"))["metrics"]
                for f in range(meta["n"]):
                    with np.load(directory / f"f{f:04d}.npz") as data:
                        np.testing.assert_array_equal(data["names"], meta["names"])
                        np.testing.assert_array_equal(data["seeds"], np.arange(1000, 1020))
                        m = data["metrics"]
                        saved_q, saved_ttl = data["q"], data["ttl"]
                        assert m.shape == (len(actions), 20, 5, 8)
                        assert np.isfinite(m).all() and m.min() >= -1e-7
                        np.testing.assert_allclose(m[:, :, :4, 0].sum(2), z["counts"][f].sum(), atol=1e-8)
                        np.testing.assert_allclose(m[:, :, :, 4].sum(2), np.broadcast_to(execution[f], (len(actions), 20)), rtol=1e-9, atol=1e-6)
                        assert (m[:, :, :, 1] <= m[:, :, :, 0]+1e-8).all()
                        np.testing.assert_array_equal(m[:, :, 4, 0], 0)
                        for i, name in enumerate(meta["names"]):
                            qa, ta = [v[f].copy() for v in actions[name]]
                            if condition == "strict":
                                qa[0], ta[0] = 0, 10.
                            np.testing.assert_array_equal(saved_q[i], qa)
                            np.testing.assert_array_equal(saved_ttl[i], ta)
                            if name in originals:
                                np.testing.assert_allclose(m[i], originals[name][f], rtol=1e-10, atol=1e-6)
                        policy_replays += len(actions)
                        files += 1
                print("verified replay", cohort, rho, condition, flush=True)
    rows = read_json(OUT / "contrasts.json")
    assert sum(x["primary"] for x in rows) == 6
    assert sum(x["mechanistic_family"] for x in rows) == 9
    artifacts = list(HERE.glob("*.py")) + list(OUT.glob("*.json")) + list(OUT.glob("*.csv"))
    artifacts += list((OUT / "models").glob("*")) + list((OUT / "figures").glob("*"))
    write_json(OUT / "artifact_hashes.json", {str(p.relative_to(ROOT)): digest(p) for p in artifacts
               if p.name not in ("artifact_hashes.json", "verification.json")})
    write_json(OUT / "verification.json", dict(passed=True, evaluation_files=files,
               function_policy_replays=policy_replays, selected_candidates_independently_reaggregated=checks,
               primary_tests=6, secondary_tests=9, tests=read_json(OUT / "tests.json"),
               reporting_tests=read_json(OUT / "reporting_tests.json"),
               historical_replays_reproduced=True, original_input_hashes_unchanged=True,
               selection_robust_to_1e_7_gbs_feasibility_rounding=True,
               common_actions_verified=True, execution_memory_conserved=True,
               all_pruning_certificates_verified=True))
    print("VERIFIED", files, "files", policy_replays, "function-policy replays", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-only", action="store_true")
    main(parser.parse_args().calibration_only)
