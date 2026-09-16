"""Fifer-inspired LSTM peak-load forecast at the public trace's minute resolution.

Fifer specifies peak forecasting and pretraining, but not neural-layer widths
or optimizer settings. Those choices are explicitly registered here, rather
than attributed to the original paper. No target-set tuning is performed.
"""
import argparse
import copy
import time

import numpy as np
import torch

from common import OUT, ROOT, digest, freeze, read_json, write_json

torch.set_num_threads(1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIG = {
    "source_paper": "10.1145/3423211.3425683",
    "context_bins": 20, "bin_seconds": 60, "forecast_peak_bins": 10,
    "hidden_size": 64, "layers": 1, "features": "log1p past invocation counts",
    "loss": "MSE of log1p maximum count in next ten minutes",
    "optimizer": "Adam", "learning_rate": .001,
    "batch_size": 256, "batches_per_epoch": 64, "maximum_epochs": 50,
    "validation_windows": 16384, "early_stopping_patience": 8,
    "training_time_fraction": .6, "gradient_clip": 5.,
    "forecast_update_minutes": 1, "online_weight_updates": False,
    "native_target": "ceil(predicted peak / batch_size); batch_size=1",
    "native_idle_timeout_minutes": 10,
    "adaptations": [
        "Twenty one-minute count bins replace twenty five-second rate bins.",
        "One-minute decisions replace ten-second monitoring; ten-minute lookahead retained.",
        "Source Azure 2021 functions replace WITS pretraining, with official split separation.",
        "No function-chain deadlines or batching metadata: independent functions, batch size one.",
        "Architecture, optimizer, log transform, padding and stopping are study choices.",
    ],
}


class PeakLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(1, CONFIG["hidden_size"], batch_first=True)
        self.head = torch.nn.Linear(CONFIG["hidden_size"], 1)

    def forward(self, x):
        hidden, _ = self.lstm(x)
        return self.head(hidden[:, -1]).squeeze(-1)


def samples(counts, ids, start, end, n, rng):
    rows = rng.choice(ids, n)
    ticks = rng.integers(max(CONFIG["context_bins"], start), end - 9, n)
    past = counts[rows[:, None], ticks[:, None] + np.arange(-20, 0)]
    future = counts[rows[:, None], ticks[:, None] + np.arange(10)]
    return (np.log1p(past).astype(np.float32)[..., None],
            np.log1p(future.max(1)).astype(np.float32))


def train(split="s2", seed=0):
    directory = OUT / "models" / f"{split}_seed{seed}"
    checkpoint = directory / "model.pt"
    if checkpoint.exists() and (directory / "training.json").exists():
        record = read_json(directory / "training.json")
        if record["checkpoint_sha256"] != digest(checkpoint):
            raise RuntimeError("Checkpoint does not match completed training record")
        return
    source = ROOT / "data/processed"
    counts = np.load(source / "counts.npy", mmap_mode="r")
    splits = np.load(source / "splits.npz")
    train_ids, val_ids = splits[split + "_train"], splits[split + "_val"]
    available_end = int(splits["s3_train_t_end"][0]) if split == "s3" else counts.shape[1]
    train_end = int(available_end * .6)
    validation_end = int(available_end * .8)
    protocol = dict(CONFIG, split=split, seed=seed, train_ids=train_ids.tolist(),
                    validation_ids=val_ids.tolist(), train_end=train_end,
                    validation_start=train_end, validation_end=validation_end,
                    source_counts_sha256=digest(source / "counts.npy"),
                    source_splits_sha256=digest(source / "splits.npz"))
    freeze(directory / "protocol.json", protocol)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.default_rng(260915 + seed)
    vx, vy = samples(counts, val_ids, train_end, validation_end,
                     CONFIG["validation_windows"], np.random.default_rng(190915))
    vx, vy = torch.from_numpy(vx).to(DEVICE), torch.from_numpy(vy).to(DEVICE)
    model = PeakLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    best_loss, best_epoch, best_state = float("inf"), -1, None
    history = []
    start_time = time.monotonic()
    for epoch in range(CONFIG["maximum_epochs"]):
        model.train()
        losses = []
        for _ in range(CONFIG["batches_per_epoch"]):
            x, y = samples(counts, train_ids, 20, train_end, CONFIG["batch_size"], rng)
            x, y = torch.from_numpy(x).to(DEVICE), torch.from_numpy(y).to(DEVICE)
            loss = torch.nn.functional.mse_loss(model(x), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"])
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            prediction = torch.cat([model(vx[j:j + 2048]) for j in range(0, len(vx), 2048)])
            val_loss = float(torch.nn.functional.mse_loss(prediction, vy))
        history.append({"epoch": epoch + 1, "train_mse": float(np.mean(losses)), "validation_mse": val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"LSTM {split} seed={seed} epoch={epoch + 1} val={val_loss:.6f} best={best_loss:.6f}", flush=True)
        if epoch - best_epoch >= CONFIG["early_stopping_patience"]:
            break
    temporary = checkpoint.with_suffix(".tmp.pt")
    torch.save({"state_dict": best_state, "config": CONFIG}, temporary)
    temporary.replace(checkpoint)
    write_json(directory / "training.json", {
        "history": history, "selected_epoch": best_epoch + 1,
        "selection_metric": "source-validation log-peak MSE",
        "elapsed_seconds": time.monotonic() - start_time,
        "checkpoint_sha256": digest(checkpoint), "protocol_sha256": digest(directory / "protocol.json"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": DEVICE, "torch_version": torch.__version__,
    })


def load_model(split="s2", seed=0, device=None):
    model = PeakLSTM()
    record = torch.load(OUT / "models" / f"{split}_seed{seed}" / "model.pt",
                        map_location="cpu", weights_only=False)
    if record["config"] != CONFIG:
        raise RuntimeError("Forecast protocol differs from trained model")
    model.load_state_dict(record["state_dict"])
    return model.to(device or DEVICE).eval()


@torch.no_grad()
def predict(model, counts, prefix=None, batch_size=4096):
    """Forecast t from counts strictly before t; optional earlier observed prefix."""
    counts = np.asarray(counts)
    if counts.ndim == 1:
        counts = counts[None, :]
    if prefix is None:
        prefix = np.zeros((len(counts), 20), dtype=np.float32)
    prefix = np.asarray(prefix)
    if prefix.shape != (len(counts), 20):
        raise ValueError("Exactly twenty past count bins are required")
    full = np.log1p(np.concatenate([prefix, counts], axis=1)).astype(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(full, 20, axis=1)[:, :-1]
    result = np.empty(counts.shape, np.float32)
    device = next(model.parameters()).device
    for f in range(len(counts)):
        for start in range(0, counts.shape[1], batch_size):
            x = np.array(windows[f, start:start + batch_size, :, None], copy=True)
            y = model(torch.from_numpy(x).to(device)).clamp(0., 20.)
            result[f, start:start + batch_size] = torch.expm1(y).cpu().numpy()
    return result


def native_actions(rates):
    q = np.ceil(np.clip(rates, 0., 200.)).astype(np.int64)
    return q, np.full(rates.shape, 10., dtype=np.float64)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["s1", "s2", "s3"], default="s2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    train(args.split, args.seed)
