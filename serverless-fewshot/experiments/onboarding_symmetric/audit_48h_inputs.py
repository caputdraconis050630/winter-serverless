"""Read-only readiness audit for a separate, future 48-hour accounting replay."""
import numpy as np
import pandas as pd

from experiment import OLD, OUT, ROOT, decisions, digest, load_cohort, write_json
from prepare import scan, capped, candidates


def main():
    directory = ROOT / "results/runs/des_jobs_r20_2019_holdout/azure2019_holdout"
    counts = np.load(ROOT / "data/processed_2019/counts.npy", mmap_mode="r")
    first, day1 = scan(counts)
    union = set()
    for gap in (1, 3, 7):
        for threshold in (10, 30, 100):
            ids = np.flatnonzero((first >= gap*1440) & (first < counts.shape[1]-1440) & (day1 >= threshold))
            union.update(map(int, capped(ids)))
    ids = np.array([int(f) for f in candidates(first, day1, 2880, counts.shape[1]) if f not in union])
    assert len(union) == 772 and len(ids) == 1755
    with np.load(directory / "shared.npz") as z:
        shared = {key: z[key] for key in z.files}
    for i, f in enumerate(ids):
        np.testing.assert_array_equal(shared["counts"][i], counts[f, first[f]:first[f]+2880])
    duration = pd.read_csv(ROOT / "data/processed_2019/duration_stats.csv")
    np.testing.assert_allclose(shared["dur_means"], np.nan_to_num(duration.dur_mean.to_numpy()[ids], nan=1.0), rtol=0, atol=0)
    np.testing.assert_allclose(shared["dur_stds"], np.nan_to_num(duration.dur_std.to_numpy()[ids], nan=.5), rtol=0, atol=0)
    names = {"gate": "G_WE_A720", "no_age_handoff": "G_WE_Ainf", "ewma": "B4a_ewma", "component": "A5_proto"}
    actions = {}
    for name, stem in names.items():
        with np.load(directory / f"{stem}__rho10.npz") as z:
            actions[name] = (z["prewarm"], z["keepalive"])
        q, ttl = actions[name]
        assert q.shape == ttl.shape == (1755, 2880)
        assert np.isfinite(q).all() and np.isfinite(ttl).all()
        assert q.min() >= 0 and ttl.min() >= 0
    for column in (0, 1):
        np.testing.assert_array_equal(actions["gate"][column][:, :720], actions["no_age_handoff"][column][:, :720])
        np.testing.assert_array_equal(actions["gate"][column][:, 720:], actions["ewma"][column][:, 720:])
    lookup = {int(f): i for i, f in enumerate(ids)}
    differences = []
    for cohort in ("azure_calibration", "azure_evaluation"):
        data = load_cohort(cohort)
        rows = [lookup[int(f)] for f in data["ids"]]
        with np.load(OLD / "models/trained0" / f"{cohort}_rates.npz") as z:
            for family in ("component", "gate"):
                q, ttl = decisions(z[family], 10.)
                saved_q, saved_ttl = [a[rows, :240] for a in actions[family]]
                differences.append(dict(cohort=cohort, family=family,
                    raw_q_different_function_minutes=int(np.count_nonzero(q != saved_q)),
                    capped_q_different_function_minutes=int(np.count_nonzero(q != np.minimum(200, saved_q))),
                    saved_q_max=int(saved_q.max()),
                    function_minutes=int(q.size),
                    ttl_different_over_1e_4_function_minutes=int(np.count_nonzero(np.abs(ttl-saved_ttl) > 1e-4)),
                    ttl_max_absolute_difference_minutes=float(np.abs(ttl-saved_ttl).max()),
                    ttl_mean_absolute_difference_minutes=float(np.abs(ttl-saved_ttl).mean())))
    files = [directory / "shared.npz", directory / "jobs.json"] + [directory / f"{n}__rho10.npz" for n in names.values()]
    result = dict(scope="Input/schedule audit only; no new 48h DES outcomes or performance validation",
                  functions=len(ids), ticks=2880, count_order_matches_original_trace=True,
                  durations_match_original=True, gate_equals_no_handoff_before720=True,
                  gate_equals_ewma_from720=True, four_hour_action_comparison=differences,
                  input_hashes={str(p.relative_to(ROOT)): digest(p) for p in files},
                  warning="Existing v1/v2 DES accounting bins are fixed to 240 minutes; a separate generalized and tested replay is required. Equal post-handoff actions do not imply equal inherited sandbox state or outcomes.")
    write_json(OUT / "dependency_48h_inputs.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
