"""One pooled prior on the exact source-support union of the C16 builder."""
import os
import sys
import numpy as np
import torch
from sklearn.cluster import KMeans

from experiment import COHORTS, OLD, OUT, ROOT, digest, frozen_json, load_cohort, save_npz
sys.path.insert(0, str(ROOT))
os.environ["SF_RUNS_DIR"] = str(ROOT / "results_azure2021/runs")
os.environ.setdefault("SF_DATA_DIR", "processed_2019")
from scripts.eval_adapt_biased import load_trainer, embed_functions, PROTO_FUNC_CAP
from models import canonical


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(.20)
    source = ROOT / "data/processed"
    ckpt = ROOT / "results_azure2021/runs/best_anil_ridge_s2_s0.pt"
    frozen_json(OUT / "source_inputs.json", {str(p.relative_to(ROOT)): digest(p)
                for p in (source / "features.npy", source / "counts.npy", source / "splits.npz", ckpt)})
    directory = OUT / "models"
    directory.mkdir(exist_ok=True)
    trainer, _ = load_trainer(11, 0)
    lam = trainer.head.ridge_lambda.item()
    path = directory / "pooled_prior.npz"
    if not path.exists():
        features = np.load(source / "features.npy", mmap_mode="r")
        counts = np.load(source / "counts.npy", mmap_mode="r")
        ids = np.load(source / "splits.npz")["s2_train"]
        emb = embed_functions(trainer.body, features, ids)
        km = KMeans(n_clusters=16, random_state=42, n_init=10).fit(emb.numpy())
        old = np.load(OLD / "models/trained0/prototypes.npz")
        np.testing.assert_allclose(km.cluster_centers_, old["centroids"], rtol=1e-4, atol=1e-4)
        device = trainer.device
        phis, targets, members, reconstructed = [], [], [], []
        starts = np.linspace(60, features.shape[1]-2, 20, dtype=int)
        for cluster in range(16):
            cp, cy = [], []
            selected = ids[km.labels_ == cluster][:PROTO_FUNC_CAP]
            members.extend(selected.tolist())
            for f in selected:
                for t in starts:
                    x = torch.from_numpy(np.asarray(features[f:f+1, t-60:t], dtype=np.float32)).to(device)
                    cp.append(trainer.body(x))
                    cy.append(float(np.log1p(counts[f, t])))
            p = torch.cat(cp)
            y = torch.tensor(cy, device=device).float().unsqueeze(1)
            w = trainer.head.adapt(p, y.expand(-1, 19))
            reconstructed.append(w.cpu().numpy())
            phis.append(p)
            targets.append(y)
        error = float(np.max(np.abs(np.stack(reconstructed)-old["prior"])))
        np.testing.assert_allclose(np.stack(reconstructed), old["prior"], rtol=1e-3, atol=2e-4)
        phi, y = torch.cat(phis), torch.cat(targets)
        pooled = trainer.head.adapt(phi, y.expand(-1, 19)).cpu().numpy()[None]
        save_npz(path, prior=pooled, centroid=emb.mean(0).numpy()[None],
                 source_ids=np.array(members), source_phi=phi.cpu().numpy(), source_y=y.cpu().numpy(),
                 reconstructed_cluster_prior=np.stack(reconstructed))
        frozen_json(directory / "provenance.json", dict(n_source_functions=len(members),
                    n_support_rows=len(phi), source_cluster_cap=PROTO_FUNC_CAP, lambda_value=lam,
                    prior_reconstruction_max_abs_error=error,
                    note="Pooled and C16 priors use the identical union of capped source support rows; no target data"))
    proto = np.load(path)
    for cohort in COHORTS:
        dest = directory / f"{cohort}_pooled.npz"
        if dest.exists():
            continue
        phi = np.load(OLD / "models/trained0" / f"{cohort}_phi.npz")["phi"]
        counts = load_cohort(cohort)["counts"]
        results = canonical(phi, counts, lam, proto["centroid"], proto["prior"])
        original = np.load(OLD / "models/trained0" / f"{cohort}_rates.npz")["no_prior"]
        np.testing.assert_allclose(results["no_prior"], original, rtol=1e-5, atol=1e-5)
        save_npz(dest, rates=results["component"])
        print("pooled forecasts", cohort, flush=True)


if __name__ == "__main__":
    main()
