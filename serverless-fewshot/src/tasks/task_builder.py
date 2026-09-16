"""Task construction for meta-learning: episodes, splits, drift detection."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from typing import Optional
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class Episode:
    """A single meta-learning episode (task) for one function."""
    func_idx: int
    support_x: np.ndarray   # [K, L, F] support feature windows
    support_y: np.ndarray   # [K, H] support targets (counts at horizons)
    query_x: np.ndarray     # [Q, L, F] query feature windows
    query_y: np.ndarray     # [Q, H] query targets
    meta: dict              # metadata: func_id, split, drift_info, etc.


class TaskBuilder:
    """Builds meta-learning tasks from processed data."""

    def __init__(self, features, counts, func_df, context_len=60, horizons=(1,)):
        """
        Args:
            features: [N, T, F] feature tensor
            counts: [N, T] raw count matrix
            func_df: DataFrame with per-function descriptors
            context_len: L, number of past minutes as input
            horizons: tuple of prediction horizons in minutes
        """
        self.features = features
        self.counts = counts
        self.func_df = func_df
        self.L = context_len
        self.horizons = horizons
        self.n_funcs, self.n_time, self.n_feat = features.shape
        self.max_horizon = max(horizons)

    def _extract_window(self, func_idx, t):
        """Extract (x, y) pair at time t for a function.

        x = features[func_idx, t-L:t, :]
        y = counts at t+h for each horizon h
        """
        x = self.features[func_idx, t - self.L:t, :]
        y = np.array([self.counts[func_idx, min(t + h - 1, self.n_time - 1)]
                       for h in self.horizons], dtype=np.float32)
        return x, y

    def build_episode(self, func_idx, k_support, k_query,
                      t_start=None, t_end=None, rng=None):
        """Build a single episode for a function.

        Samples support and query windows from the given time range.
        Support windows come before query windows (temporal ordering).
        """
        if rng is None:
            rng = np.random.default_rng()

        if t_start is None:
            t_start = self.L
        if t_end is None:
            t_end = self.n_time - self.max_horizon

        valid_range = t_end - t_start
        if valid_range < k_support + k_query:
            # Not enough data — use what we have
            k_total = valid_range
            k_support = min(k_support, k_total // 2)
            k_query = k_total - k_support

        # Sample time indices, then split into support (earlier) and query (later)
        total_needed = k_support + k_query
        if total_needed <= valid_range:
            indices = rng.choice(valid_range, size=total_needed, replace=False)
        else:
            indices = rng.choice(valid_range, size=total_needed, replace=True)

        indices.sort()
        indices = indices + t_start

        support_times = indices[:k_support]
        query_times = indices[k_support:k_support + k_query]

        support_x, support_y = [], []
        for t in support_times:
            x, y = self._extract_window(func_idx, t)
            support_x.append(x)
            support_y.append(y)

        query_x, query_y = [], []
        for t in query_times:
            x, y = self._extract_window(func_idx, t)
            query_x.append(x)
            query_y.append(y)

        return Episode(
            func_idx=func_idx,
            support_x=np.stack(support_x) if support_x else np.zeros((0, self.L, self.n_feat)),
            support_y=np.stack(support_y) if support_y else np.zeros((0, len(self.horizons))),
            query_x=np.stack(query_x) if query_x else np.zeros((0, self.L, self.n_feat)),
            query_y=np.stack(query_y) if query_y else np.zeros((0, len(self.horizons))),
            meta={"func_idx": func_idx, "func_id": str(self.func_df.iloc[func_idx].get("func_id", func_idx))},
        )

    # -------- Splits --------

    def split_s1_random(self, seed=42):
        """S1: Random hold-out split by function."""
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.n_funcs)
        n_train = int(0.7 * self.n_funcs)
        n_val = int(0.1 * self.n_funcs)
        return {
            "train": indices[:n_train],
            "val": indices[n_train:n_train + n_val],
            "test": indices[n_train + n_val:],
        }

    def split_s2_cluster_holdout(self, n_clusters=12, holdout_clusters=3, seed=42):
        """S2: Cluster hold-out split. Hold out entire clusters for meta-test."""
        # Build descriptor matrix for clustering
        desc_cols = ["mean_rate_per_min", "iat_cv", "fano_factor", "sparsity",
                     "daily_spectral_energy", "weekly_spectral_energy"]
        available_cols = [c for c in desc_cols if c in self.func_df.columns]

        X = self.func_df[available_cols].values.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        labels = km.fit_predict(X_scaled)

        rng = np.random.default_rng(seed)
        held_out = rng.choice(n_clusters, size=holdout_clusters, replace=False)

        test_mask = np.isin(labels, held_out)
        train_val_indices = np.where(~test_mask)[0]
        test_indices = np.where(test_mask)[0]

        # Split train_val into train and val
        rng.shuffle(train_val_indices)
        n_val = max(1, int(0.125 * len(train_val_indices)))  # ~10% of total
        val_indices = train_val_indices[:n_val]
        train_indices = train_val_indices[n_val:]

        return {
            "train": train_indices,
            "val": val_indices,
            "test": test_indices,
            "cluster_labels": labels,
            "held_out_clusters": held_out.tolist(),
        }

    def split_s3_temporal(self, train_days=10, seed=42):
        """S3: Temporal split. Train on days 1-10, test on days 11-14."""
        train_end = train_days * 1440  # minutes
        rng = np.random.default_rng(seed)

        # All functions participate, but time ranges differ
        all_indices = np.arange(self.n_funcs)
        rng.shuffle(all_indices)
        n_val = max(1, int(0.1 * self.n_funcs))

        return {
            "train": all_indices[n_val:],
            "val": all_indices[:n_val],
            "test": all_indices,  # all functions, but evaluated on days 11-14
            "train_t_end": train_end,
            "test_t_start": train_end,
        }

    # -------- Episode iterators --------

    def episode_iterator(self, func_indices, k_support, k_query,
                         t_start=None, t_end=None, n_episodes=None, seed=0):
        """Yield episodes for a set of functions."""
        rng = np.random.default_rng(seed)
        count = 0
        while n_episodes is None or count < n_episodes:
            func_idx = rng.choice(func_indices)
            episode = self.build_episode(func_idx, k_support, k_query,
                                         t_start=t_start, t_end=t_end, rng=rng)
            yield episode
            count += 1

    def onboarding_scenario(self, func_idx, k_checkpoints=(0, 1, 5, 10, 20),
                            window_size=60):
        """New-function onboarding: replay stream, evaluate at K checkpoints."""
        results = []
        t_start = self.L
        for k in k_checkpoints:
            t_support_end = t_start + k * window_size
            t_support_end = min(t_support_end, self.n_time - self.max_horizon - window_size)
            t_eval_start = t_support_end
            t_eval_end = min(t_eval_start + 32 * window_size, self.n_time - self.max_horizon)

            if k > 0:
                support_times = np.linspace(t_start, t_support_end - 1, k, dtype=int)
                sx, sy = [], []
                for t in support_times:
                    x, y = self._extract_window(func_idx, max(t, self.L))
                    sx.append(x)
                    sy.append(y)
                support_x = np.stack(sx)
                support_y = np.stack(sy)
            else:
                support_x = np.zeros((0, self.L, self.n_feat), dtype=np.float32)
                support_y = np.zeros((0, len(self.horizons)), dtype=np.float32)

            # Query windows
            n_query = min(32, (t_eval_end - t_eval_start) // window_size)
            if n_query > 0:
                query_times = np.linspace(t_eval_start, t_eval_end - 1, n_query, dtype=int)
                qx, qy = [], []
                for t in query_times:
                    x, y = self._extract_window(func_idx, max(t, self.L))
                    qx.append(x)
                    qy.append(y)
                query_x = np.stack(qx)
                query_y = np.stack(qy)
            else:
                query_x = np.zeros((0, self.L, self.n_feat), dtype=np.float32)
                query_y = np.zeros((0, len(self.horizons)), dtype=np.float32)

            results.append(Episode(
                func_idx=func_idx,
                support_x=support_x, support_y=support_y,
                query_x=query_x, query_y=query_y,
                meta={"k": k, "scenario": "onboarding"},
            ))
        return results
