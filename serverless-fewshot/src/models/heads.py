"""Per-function heads: ANIL-GD, Ridge, Bayesian, Full adaptation.

All heads map embedding [B, d] -> predictions [B, Q] (multi-quantile).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


N_QUANTILES = 19  # 0.05, 0.10, ..., 0.95
QUANTILES = torch.linspace(0.05, 0.95, N_QUANTILES)


def pinball_loss(pred, target, quantiles=None, n_horizons=1):
    """Pinball (quantile) loss.

    Args:
        pred: [B, Q*H] predicted quantiles (Q quantiles × H horizons)
        target: [B], [B, 1], or [B, H] actual values
        quantiles: [Q] quantile levels
        n_horizons: number of horizons (pred has Q*H outputs)
    """
    if quantiles is None:
        quantiles = QUANTILES
    quantiles = quantiles.to(pred.device)
    Q = len(quantiles)

    if target.dim() == 1:
        target = target.unsqueeze(-1)  # [B, 1]

    B = pred.shape[0]
    H = n_horizons

    if pred.shape[-1] == Q * H and H > 1:
        # Reshape pred to [B, H, Q] and target to [B, H, 1]
        pred_r = pred.reshape(B, H, Q)
        if target.shape[-1] == 1:
            target_r = target.unsqueeze(-1).expand(B, H, 1)
        else:
            target_r = target.unsqueeze(-1)  # [B, H, 1]
        errors = target_r - pred_r  # [B, H, Q]
        q = quantiles.unsqueeze(0).unsqueeze(0)  # [1, 1, Q]
        loss = torch.maximum(q * errors, (q - 1) * errors)
    else:
        errors = target - pred  # [B, Q]
        loss = torch.maximum(quantiles * errors, (quantiles - 1) * errors)

    return loss.mean()


def crps_from_quantiles(pred_quantiles, target, quantiles=None):
    """Approximate CRPS from predicted quantiles (for evaluation)."""
    if quantiles is None:
        quantiles = QUANTILES
    quantiles = quantiles.to(pred_quantiles.device)

    if target.dim() == 1:
        target = target.unsqueeze(-1)

    errors = target - pred_quantiles
    pinball = torch.maximum(quantiles * errors, (quantiles - 1) * errors)
    # CRPS ≈ 2 * mean pinball loss
    return 2 * pinball.mean()


class RidgeHead(nn.Module):
    """Closed-form ridge regression head (H2).

    Multi-quantile linear head: W* = (Φᵀ Φ + λI)⁻¹ Φᵀ Y
    Meta-trains through torch.linalg.solve for end-to-end learning of body + λ.
    """
    def __init__(self, embedding_dim, n_outputs=N_QUANTILES, n_horizons=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_outputs = n_outputs * n_horizons
        # Learnable log-lambda (softplus-parameterized)
        self.log_lambda = nn.Parameter(torch.tensor(0.0))

    @property
    def ridge_lambda(self):
        return F.softplus(self.log_lambda)

    def adapt(self, phi_support, y_support):
        """Solve for head weights given support embeddings and targets.

        Args:
            phi_support: [K, d] or [B, K, d] support embeddings
            y_support: [K, O] or [B, K, O] support targets (O = n_outputs)

        Returns:
            W: [d, O] or [B, d, O] head weights
        """
        lam = self.ridge_lambda
        batched = phi_support.dim() == 3

        # linalg.solve requires consistent dtype; disable autocast for the solve
        with torch.amp.autocast("cuda", enabled=False):
            phi_f = phi_support.float()
            y_f = y_support.float()

            if batched:
                B, K, d = phi_f.shape
                PhiTPhi = torch.bmm(phi_f.transpose(1, 2), phi_f)  # [B, d, d]
                reg = lam * torch.eye(d, device=phi_f.device).unsqueeze(0)  # [1, d, d]
                A = PhiTPhi + reg  # [B, d, d]
                PhiTY = torch.bmm(phi_f.transpose(1, 2), y_f)  # [B, d, O]
                W = torch.linalg.solve(A, PhiTY)  # [B, d, O]
            else:
                K, d = phi_f.shape
                PhiTPhi = phi_f.T @ phi_f  # [d, d]
                reg = lam * torch.eye(d, device=phi_f.device)
                A = PhiTPhi + reg
                PhiTY = phi_f.T @ y_f  # [d, O]
                W = torch.linalg.solve(A, PhiTY)  # [d, O]

        return W

    def predict(self, phi_query, W):
        """Predict using adapted weights.

        Args:
            phi_query: [Q, d] or [B, Q, d]
            W: [d, O] or [B, d, O]

        Returns:
            predictions: [Q, O] or [B, Q, O]
        """
        if phi_query.dim() == 3:
            return torch.bmm(phi_query, W)
        return phi_query @ W

    def forward(self, phi_support, y_support, phi_query):
        W = self.adapt(phi_support, y_support)
        return self.predict(phi_query, W)


class BiasedRidgeHead(RidgeHead):
    """Prototype-biased ridge head (H2 + prototype initialization).

    W* = argmin ||Φ_s W - Y_s||² + λ||W - W_proto||²
    Solution: W* = (Φᵀ Φ + λI)⁻¹ (Φᵀ Y + λ W_proto)
    """
    def adapt_biased(self, phi_support, y_support, W_proto):
        """Solve biased ridge with prototype prior.

        Args:
            phi_support: [K, d] or [B, K, d]
            y_support: [K, O] or [B, K, O]
            W_proto: [d, O] or [B, d, O] prototype weights

        Returns:
            W: [d, O] or [B, d, O]
        """
        lam = self.ridge_lambda
        batched = phi_support.dim() == 3

        with torch.amp.autocast("cuda", enabled=False):
            phi_f = phi_support.float()
            y_f = y_support.float()
            wp_f = W_proto.float()

            if batched:
                B, K, d = phi_f.shape
                PhiTPhi = torch.bmm(phi_f.transpose(1, 2), phi_f)
                reg = lam * torch.eye(d, device=phi_f.device).unsqueeze(0)
                A = PhiTPhi + reg
                PhiTY = torch.bmm(phi_f.transpose(1, 2), y_f)
                rhs = PhiTY + lam * wp_f
                W = torch.linalg.solve(A, rhs)
            else:
                K, d = phi_f.shape
                PhiTPhi = phi_f.T @ phi_f
                reg = lam * torch.eye(d, device=phi_f.device)
                A = PhiTPhi + reg
                PhiTY = phi_f.T @ y_f
                rhs = PhiTY + lam * wp_f
                W = torch.linalg.solve(A, rhs)

        return W


class ANILGDHead(nn.Module):
    """ANIL-GD head (H1): linear head adapted by SGD steps on support set."""
    def __init__(self, embedding_dim, n_outputs=N_QUANTILES, n_horizons=1,
                 inner_steps=5, inner_lr=0.01):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_outputs = n_outputs * n_horizons
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        # Initial weights (meta-learned)
        self.W_init = nn.Parameter(torch.randn(embedding_dim, self.n_outputs) * 0.01)
        self.b_init = nn.Parameter(torch.zeros(self.n_outputs))

    def adapt(self, phi_support, y_support):
        """Inner-loop SGD adaptation.

        Args:
            phi_support: [K, d]
            y_support: [K, O]

        Returns:
            W: [d, O], b: [O]
        """
        W = self.W_init.clone()
        b = self.b_init.clone()

        for _ in range(self.inner_steps):
            pred = phi_support @ W + b  # [K, O]
            loss = F.mse_loss(pred, y_support[:, :self.n_outputs])
            # Manual gradient (to allow second-order through body)
            grad_W = torch.autograd.grad(loss, W, create_graph=True)[0]
            grad_b = torch.autograd.grad(loss, b, create_graph=True)[0]
            W = W - self.inner_lr * grad_W
            b = b - self.inner_lr * grad_b

        return W, b

    def predict(self, phi_query, W, b):
        return phi_query @ W + b

    def forward(self, phi_support, y_support, phi_query):
        W, b = self.adapt(phi_support, y_support)
        return self.predict(phi_query, W, b)


class BayesianLinearHead(nn.Module):
    """Bayesian linear head (H3): neural-linear with predictive uncertainty.

    Posterior: Σ = (Φᵀ Φ/σ² + I/σₚ²)⁻¹
    Mean: μ = Σ Φᵀ Y/σ²
    Predictive variance: σ²_pred = σ² + φ_q Σ φ_qᵀ
    """
    def __init__(self, embedding_dim, n_outputs=N_QUANTILES, n_horizons=1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_outputs = n_outputs * n_horizons
        self.log_sigma_noise = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_prior = nn.Parameter(torch.tensor(0.0))

    @property
    def sigma_noise(self):
        return F.softplus(self.log_sigma_noise) + 1e-4

    @property
    def sigma_prior(self):
        return F.softplus(self.log_sigma_prior) + 1e-4

    def adapt(self, phi_support, y_support):
        """Compute posterior parameters.

        Returns:
            mu: [d, O] posterior mean weights
            Sigma: [d, d] posterior covariance (shared across outputs)
        """
        K, d = phi_support.shape
        sigma_n2 = self.sigma_noise ** 2
        sigma_p2 = self.sigma_prior ** 2

        PhiTPhi = phi_support.T @ phi_support  # [d, d]
        A = PhiTPhi / sigma_n2 + torch.eye(d, device=phi_support.device) / sigma_p2
        Sigma = torch.linalg.inv(A)  # [d, d]
        mu = Sigma @ (phi_support.T @ y_support / sigma_n2)  # [d, O]
        return mu, Sigma

    def predict(self, phi_query, mu, Sigma):
        """Predict mean and variance.

        Returns:
            mean: [Q, O]
            var: [Q] predictive variance (scalar per query point)
        """
        mean = phi_query @ mu  # [Q, O]
        # Predictive variance: σ² + φ Σ φᵀ (per query)
        var = self.sigma_noise ** 2 + (phi_query @ Sigma * phi_query).sum(dim=-1)  # [Q]
        return mean, var

    def forward(self, phi_support, y_support, phi_query):
        mu, Sigma = self.adapt(phi_support, y_support)
        return self.predict(phi_query, mu, Sigma)


def build_head(head_type, embedding_dim, n_quantiles=N_QUANTILES, n_horizons=1, cfg=None):
    """Factory for head types."""
    if head_type == "ridge":
        return BiasedRidgeHead(embedding_dim, n_quantiles, n_horizons)
    elif head_type == "anil_gd":
        inner_steps = 5 if cfg is None else cfg.get("gd_inner_steps", 5)
        inner_lr = 0.01 if cfg is None else cfg.get("gd_inner_lr", 0.01)
        return ANILGDHead(embedding_dim, n_quantiles, n_horizons,
                          inner_steps=inner_steps, inner_lr=inner_lr)
    elif head_type == "bayesian":
        return BayesianLinearHead(embedding_dim, n_quantiles, n_horizons)
    elif head_type == "full":
        # For full adaptation (H4/MAML), the "head" is just a ridge head
        # but adaptation updates ALL parameters (body + head)
        return RidgeHead(embedding_dim, n_quantiles, n_horizons)
    else:
        raise ValueError(f"Unknown head type: {head_type}")
