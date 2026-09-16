"""ANIL meta-training with batched ridge solves.

Meta-trains the shared body through the closed-form head solution.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from src.models.bodies import build_body
from src.models.heads import build_head, pinball_loss, N_QUANTILES, QUANTILES, crps_from_quantiles


class MetaDataset(torch.utils.data.Dataset):
    """Wraps numpy arrays for episodic sampling."""

    def __init__(self, features, counts, func_indices, context_len=60,
                 k_support=10, k_query=32, horizons=(1,)):
        # Keep numpy references (memmap-safe); convert per-window in
        # __getitem__ so multi-GB feature tensors never materialize in RAM.
        self.features = features
        self.counts = counts
        self.func_indices = func_indices
        self.L = context_len
        self.k_support = k_support
        self.k_query = k_query
        self.horizons = horizons
        self.max_h = max(horizons)
        self.T = features.shape[1]

    def __len__(self):
        return len(self.func_indices) * 100  # virtual length for epoch

    def __getitem__(self, idx):
        # Sample a random function
        func_idx = self.func_indices[idx % len(self.func_indices)]

        # Sample support + query windows
        total = self.k_support + self.k_query
        valid_start = self.L
        valid_end = self.T - self.max_h
        valid_range = valid_end - valid_start

        if valid_range < total:
            times = np.random.randint(valid_start, valid_end, size=total)
        else:
            times = np.sort(np.random.choice(valid_range, size=total, replace=False) + valid_start)

        support_t = times[:self.k_support]
        query_t = times[self.k_support:]

        # Extract windows (np.asarray forces the memmap read per window)
        support_x = torch.stack([
            torch.from_numpy(np.asarray(self.features[func_idx, t - self.L:t],
                                        dtype=np.float32))
            for t in support_t])
        query_x = torch.stack([
            torch.from_numpy(np.asarray(self.features[func_idx, t - self.L:t],
                                        dtype=np.float32))
            for t in query_t])

        # Targets: log1p(count) at each horizon
        support_y = torch.stack([
            torch.tensor([np.log1p(float(self.counts[func_idx, min(t + h - 1, self.T - 1)]))
                          for h in self.horizons])
            for t in support_t
        ])
        query_y = torch.stack([
            torch.tensor([np.log1p(float(self.counts[func_idx, min(t + h - 1, self.T - 1)]))
                          for h in self.horizons])
            for t in query_t
        ])

        return support_x, support_y, query_x, query_y


def collate_episodes(batch):
    """Collate a batch of episodes into batched tensors."""
    sx = torch.stack([b[0] for b in batch])  # [B, K, L, F]
    sy = torch.stack([b[1] for b in batch])  # [B, K, H]
    qx = torch.stack([b[2] for b in batch])  # [B, Q, L, F]
    qy = torch.stack([b[3] for b in batch])  # [B, Q, H]
    return sx, sy, qx, qy


class ANILMetaTrainer:
    """ANIL meta-trainer with batched ridge head."""

    def __init__(self, body_type="tcn", head_type="ridge", in_features=11,
                 embedding_dim=64, n_quantiles=N_QUANTILES, n_horizons=1,
                 lr=3e-4, weight_decay=0.01, grad_clip=1.0, use_amp=True,
                 device="cuda", body_cfg=None, head_cfg=None):

        self.device = device
        self.n_quantiles = n_quantiles
        self.n_horizons = n_horizons
        self.n_outputs = n_quantiles * n_horizons
        self.use_amp = use_amp
        self.grad_clip = grad_clip

        # Build model
        self.body = build_body(body_type, in_features, body_cfg).to(device)
        self.head = build_head(head_type, embedding_dim, n_quantiles, n_horizons, head_cfg).to(device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.body.parameters()) + list(self.head.parameters()),
            lr=lr, weight_decay=weight_decay,
        )

        self.scaler = GradScaler("cuda", enabled=use_amp)
        self.quantiles = QUANTILES.to(device)

    def meta_train_step(self, support_x, support_y, query_x, query_y):
        """Single meta-training step.

        Args:
            support_x: [B, K, L, F]
            support_y: [B, K, H] (H horizons)
            query_x: [B, Q, L, F]
            query_y: [B, Q, H]

        Returns:
            loss: scalar
            metrics: dict
        """
        B, K, L, F = support_x.shape
        _, Q, _, _ = query_x.shape
        H = self.n_horizons

        support_x = support_x.float().to(self.device)
        support_y = support_y.float().to(self.device)
        query_x = query_x.float().to(self.device)
        query_y = query_y.float().to(self.device)

        with autocast("cuda", enabled=self.use_amp, dtype=torch.bfloat16):
            # Encode support and query
            # Reshape to [B*K, L, F] and [B*Q, L, F] for batched body forward
            phi_s = self.body(support_x.reshape(B * K, L, F))  # [B*K, d]
            phi_q = self.body(query_x.reshape(B * Q, L, F))    # [B*Q, d]

            d = phi_s.shape[-1]
            phi_s = phi_s.reshape(B, K, d)   # [B, K, d]
            phi_q = phi_q.reshape(B, Q, d)   # [B, Q, d]

            # Expand support targets to quantile outputs
            # support_y: [B, K, H] -> [B, K, n_outputs]
            # For ridge, targets should be the actual values repeated for each quantile
            # (the pinball loss on query does the quantile differentiation)
            support_y_expanded = support_y.unsqueeze(-1).expand(B, K, H, self.n_quantiles)
            support_y_expanded = support_y_expanded.reshape(B, K, self.n_outputs)

            # Batched ridge solve
            W = self.head.adapt(phi_s, support_y_expanded)  # [B, d, n_outputs]

            # Predict on query
            pred = self.head.predict(phi_q, W)  # [B, Q, n_outputs]

            # Compute pinball loss
            # pred: [B, Q, H*Q_quantiles], query_y: [B, Q, H]
            pred_flat = pred.reshape(B * Q, self.n_outputs)
            target_flat = query_y.reshape(B * Q, H)
            loss = pinball_loss(pred_flat, target_flat, self.quantiles, n_horizons=H)

        # Backward
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(self.body.parameters()) + list(self.head.parameters()),
            self.grad_clip,
        )
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Metrics
        with torch.no_grad():
            # CRPS approximation on median (horizon 0)
            median_idx = self.n_quantiles // 2  # ~0.5 quantile
            median_pred = pred[:, :, median_idx]  # [B, Q]
            mae = (median_pred - query_y[:, :, 0]).abs().mean()
            crps = crps_from_quantiles(
                pred[:, :, :self.n_quantiles].reshape(-1, self.n_quantiles),
                query_y[:, :, 0].reshape(-1),
                self.quantiles,
            )

        return loss.item(), {"mae": mae.item(), "crps": crps.item(),
                              "lambda": self.head.ridge_lambda.item()}

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate on a dataset."""
        self.body.eval()
        total_loss = 0
        total_crps = 0
        n_batches = 0

        for sx, sy, qx, qy in dataloader:
            B, K, L, F = sx.shape
            _, Q, _, _ = qx.shape
            H = self.n_horizons

            sx = sx.float().to(self.device)
            sy = sy.float().to(self.device)
            qx = qx.float().to(self.device)
            qy = qy.float().to(self.device)

            phi_s = self.body(sx.reshape(B * K, L, F)).reshape(B, K, -1)
            phi_q = self.body(qx.reshape(B * Q, L, F)).reshape(B, Q, -1)

            sy_exp = sy.unsqueeze(-1).expand(B, K, H, self.n_quantiles).reshape(B, K, self.n_outputs)
            W = self.head.adapt(phi_s, sy_exp)
            pred = self.head.predict(phi_q, W)

            pred_flat = pred.reshape(B * Q, self.n_outputs)
            target_flat = qy.reshape(B * Q, H)
            loss = pinball_loss(pred_flat, target_flat, self.quantiles, n_horizons=H)
            crps = crps_from_quantiles(
                pred[:, :, :self.n_quantiles].reshape(-1, self.n_quantiles),
                qy[:, :, 0].reshape(-1),
                self.quantiles,
            )

            total_loss += loss.item()
            total_crps += crps.item()
            n_batches += 1

        self.body.train()
        return {"loss": total_loss / max(1, n_batches),
                "crps": total_crps / max(1, n_batches)}

    def save(self, path):
        torch.save({
            "body": self.body.state_dict(),
            "head": self.head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.body.load_state_dict(ckpt["body"])
        self.head.load_state_dict(ckpt["head"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
