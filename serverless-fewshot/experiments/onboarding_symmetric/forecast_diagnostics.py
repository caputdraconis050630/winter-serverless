"""Descriptive prequential prediction and fixed-wrapper action diagnostics."""
import numpy as np

from experiment import COHORTS, OLD, OUT, decisions, load_cohort, priors, rates, read_json, selected_actions, write_json
from policies import ewma
from analyze import PairedBootstrap, csv_write


def main():
    read_json(OUT / "evaluation_lock.json")
    rows, paired, routing = [], [], []
    proto = np.load(OLD / "models/trained0/prototypes.npz")
    centroid = proto["centroids"]
    centroid = centroid/np.maximum(np.linalg.norm(centroid, axis=1, keepdims=True), 1e-8)
    selected = read_json(OUT / "selection_rho10.json")
    fleet, _ = priors()
    for cohort in COHORTS[1:]:
        z = load_cohort(cohort)
        forecasts = rates(cohort, 10.)
        forecasts["ewma_default"] = ewma(z["counts"])
        config = selected["strong_simple"][2]["config"]
        if config and config.get("family") == "ewma":
            forecasts["strong_simple_underlying_ewma"] = ewma(z["counts"], config["alpha"], np.log1p(fleet) if config["prior"] else 0.)
        target = np.log1p(z["counts"])
        losses = {name: np.abs(np.log1p(pred)-target) for name, pred in forecasts.items()}
        losses["random_mean"] = np.mean([losses[f"random{s}"] for s in range(5)], axis=0)
        bs = PairedBootstrap(z["apps"], z["counts"].sum(1))
        aa = selected_actions(cohort, 10.)
        hist = np.concatenate([np.zeros((len(target), 1)), np.cumsum(z["counts"][:, :-1], axis=1)], axis=1)
        phi = np.load(OLD / "models/trained0" / f"{cohort}_phi.npz")["phi"]
        assignment = (phi[:, 0]@centroid.T).argmax(1)
        routing.append(dict(cohort=cohort, first_tick_prototype_counts=np.bincount(assignment, minlength=16).tolist(),
                            first60_ewma_function_minute_share=float(((hist[:, :60]>0)&(hist[:, :60]<100)).mean())))
        for horizon in (1, 5, 15, 60, 240):
            for name, error in losses.items():
                rows.append(dict(cohort=cohort, first_minutes=horizon, forecast=name,
                                 mean_log_count_mae=float(error[:, :horizon].mean())))
            for a, b, action_a, action_b in (
                ("component", "no_prior", "component__b2", "no_prior__component_wrapper"),
                ("component", "pooled_prior", "component__b2", "pooled_prior__component_wrapper"),
                ("trained_lambda", "random_mean", None, None)):
                delta = (losses[a][:, :horizon]-losses[b][:, :horizon]).mean(1)
                samples = (bs.weights@bs.group_sum(delta))/bs.bn
                lo, hi = np.quantile(samples, [.025, .975])
                row = dict(cohort=cohort, first_minutes=horizon, arm=a, comparator=b,
                           log_mae_delta=float(delta.mean()), ci_low=float(lo), ci_high=float(hi))
                if action_a in aa and action_b in aa:
                    row["q_agreement_same_wrapper"] = float((aa[action_a][0][:, :horizon] == aa[action_b][0][:, :horizon]).mean())
                    row["ttl_mean_absolute_difference_minutes"] = float(np.abs(aa[action_a][1][:, :horizon]-aa[action_b][1][:, :horizon]).mean())
                paired.append(row)
    csv_write(OUT / "forecast_errors.csv", rows)
    csv_write(OUT / "forecast_paired.csv", paired)
    write_json(OUT / "routing_diagnostics.json", routing)
    print("Descriptive forecast diagnostics saved; not used in selection or primary tests", flush=True)


if __name__ == "__main__":
    main()
