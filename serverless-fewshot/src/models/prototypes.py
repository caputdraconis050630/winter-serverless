"""Prototype-initialized biased ridge system.

1. Embed every meta-train function (mean-pooled body outputs)
2. K-means cluster the embeddings
3. Precompute prototype head W_c per cluster
4. New function: assign nearest prototype, use biased ridge
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans


class PrototypeManager:
    """Manages function prototypes for biased ridge initialization."""

    def __init__(self, n_clusters=16, device="cuda"):
        self.n_clusters = n_clusters
        self.device = device
        self.centroids = None       # [K, d] cluster centroids
        self.proto_weights = None   # [K, d, O] prototype head weights per cluster
        self.kmeans = None

    @torch.no_grad()
    def compute_embeddings(self, body, features_tensor, func_indices, batch_size=256):
        """Compute mean-pooled embeddings for a set of functions.

        Args:
            body: shared body network
            features_tensor: [N, T, F] full feature tensor (torch)
            func_indices: list of function indices to embed
            batch_size: GPU batch size

        Returns:
            embeddings: [len(func_indices), d] mean-pooled embeddings
        """
        body.eval()
        all_embeds = []
        context_len = 60  # L

        for start in range(0, len(func_indices), batch_size):
            batch_indices = func_indices[start:start + batch_size]
            batch_embeds = []

            # Sample windows across time for each function
            n_windows = 50  # mean-pool over 50 windows
            T = features_tensor.shape[1]
            window_starts = np.linspace(context_len, T - 1, n_windows, dtype=int)

            for t in window_starts:
                # [B, L, F]
                x = features_tensor[batch_indices, t - context_len:t, :].to(self.device)
                phi = body(x)  # [B, d]
                batch_embeds.append(phi)

            # Mean pool: [B, n_windows, d] -> [B, d]
            stacked = torch.stack(batch_embeds, dim=1)  # [B, n_windows, d]
            mean_embed = stacked.mean(dim=1)  # [B, d]
            all_embeds.append(mean_embed.cpu())

        return torch.cat(all_embeds, dim=0)  # [N_train, d]

    def fit_clusters(self, embeddings):
        """Fit k-means on function embeddings.

        Args:
            embeddings: [N, d] numpy or torch
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.numpy()

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = self.kmeans.fit_predict(embeddings)
        self.centroids = torch.from_numpy(self.kmeans.cluster_centers_).float()
        return labels

    def compute_prototype_heads(self, body, head, features_tensor, counts_tensor,
                                func_indices, labels, n_quantiles=19, n_horizons=1):
        """Precompute prototype head W_c per cluster using pooled cluster data.

        Args:
            body: shared body
            head: RidgeHead instance
            features_tensor, counts_tensor: data tensors
            func_indices: training function indices
            labels: cluster assignments for func_indices
            n_quantiles, n_horizons: output dimensions
        """
        body.eval()
        d = body.embedding_dim
        n_outputs = n_quantiles * n_horizons
        self.proto_weights = torch.zeros(self.n_clusters, d, n_outputs)

        context_len = 60
        T = features_tensor.shape[1]

        for c in range(self.n_clusters):
            cluster_mask = labels == c
            cluster_funcs = func_indices[cluster_mask]
            if len(cluster_funcs) == 0:
                continue

            # Pool all support data from cluster
            all_phi = []
            all_y = []
            n_windows = min(20, T // context_len)
            window_starts = np.linspace(context_len, T - 2, n_windows, dtype=int)

            with torch.no_grad():
                for fi in cluster_funcs[:50]:  # cap per cluster
                    for t in window_starts:
                        x = features_tensor[fi:fi + 1, t - context_len:t, :].to(self.device)
                        phi = body(x)  # [1, d]
                        # log1p target for consistency with meta-training
                        y = torch.log1p(counts_tensor[fi, min(t, T - 1)]).unsqueeze(0)
                        all_phi.append(phi.cpu())
                        all_y.append(y)

            if len(all_phi) == 0:
                continue

            phi_pool = torch.cat(all_phi, dim=0).to(self.device)  # [N_pool, d]
            y_pool = torch.stack(all_y).float().to(self.device)  # [N_pool, 1]
            # Expand y to quantile targets (for ridge fit, use repeated y)
            y_expanded = y_pool.expand(-1, n_outputs)  # [N_pool, O]

            # Ridge solve for prototype
            W = head.adapt(phi_pool, y_expanded)
            self.proto_weights[c] = W.detach().cpu()

    def assign_prototype(self, embedding):
        """Assign a function to nearest prototype.

        Args:
            embedding: [d] or [B, d]

        Returns:
            cluster_idx: int or [B]
            W_proto: [d, O] or [B, d, O]
        """
        if self.centroids is None:
            raise RuntimeError("Must fit clusters first")

        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float()

        centroids = self.centroids.to(embedding.device)
        proto_w = self.proto_weights.to(embedding.device)

        if embedding.dim() == 1:
            # Single function
            sims = F.cosine_similarity(embedding.unsqueeze(0), centroids, dim=1)
            idx = sims.argmax().item()
            return idx, proto_w[idx]
        else:
            # Batch
            # [B, d] @ [d, K] -> [B, K]
            sims = F.cosine_similarity(
                embedding.unsqueeze(1),  # [B, 1, d]
                centroids.unsqueeze(0),  # [1, K, d]
                dim=2
            )
            indices = sims.argmax(dim=1)  # [B]
            W_protos = proto_w[indices]  # [B, d, O]
            return indices, W_protos
