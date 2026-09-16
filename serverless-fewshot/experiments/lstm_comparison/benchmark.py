"""Equal-batch body/read latency measurements for the deployed predictors."""
import sys
import time
import os
import numpy as np
import torch
from common import ROOT, OUT, digest, write_json
from model import load_model

sys.path.insert(0,str(ROOT))
from src.meta.trainer import ANILMetaTrainer
from src.models.heads import N_QUANTILES


def main():
    torch.set_num_threads(1)
    torch.manual_seed(260915)
    rows = []
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        trainer = ANILMetaTrainer(body_type="tcn", head_type="ridge", in_features=11,
            embedding_dim=64, n_quantiles=N_QUANTILES, n_horizons=1, device=device)
        trainer.load(ROOT / "results_azure2021/runs/best_anil_ridge_s1_s0.pt")
        trainer.body.eval()
        lstm = load_model("s1", device=device)
        for batch in (1, 100, 1000):
            x_tcn = torch.rand(batch,60,11,device=device)
            x_lstm = torch.rand(batch,20,1,device=device)
            head = torch.rand(batch,64,device=device,dtype=torch.float64)
            def winter():
                return (trainer.body(x_tcn).double()*head).sum(1)
            for name, op in (("WINTER_body_scalar_read",winter),("LSTM_peak",lambda:lstm(x_lstm))):
                with torch.inference_mode():
                    for _ in range(5):op()
                    times = []
                    for _ in range(30):
                        if device == "cuda":torch.cuda.synchronize()
                        start=time.perf_counter_ns()
                        op()
                        if device == "cuda":torch.cuda.synchronize()
                        times.append((time.perf_counter_ns()-start)/1e6)
                rows.append(dict(model=name,device=device,batch=batch,repetitions=30,warmup=5,
                                 median_batch_ms=float(np.median(times)),p95_batch_ms=float(np.percentile(times,95)),
                                 median_us_per_function=float(np.median(times)*1000/batch),samples_ms=times))
        del trainer,lstm
    write_json(OUT / "benchmark.json",dict(rows=rows,torch=torch.__version__,cpu_threads=1,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        scope="model body and scalar read; resident tensors; excludes feature creation, device transfers, controller and ridge refit",
        cpu_affinity=sorted(os.sched_getaffinity(0)),
        concurrency="model export finished; DES workers pinned to CPUs 0-5 while benchmark uses CPU 6; background control plane remains",
        source_sha256=digest(__file__)))


if __name__ == "__main__":main()
