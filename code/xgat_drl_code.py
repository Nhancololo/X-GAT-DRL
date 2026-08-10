"""
Reusable components for X-GAT-DRL experiments and controlled benchmarks.

Includes:
* causal preprocessing and higher-moment statistical diagnostics
* Gaussian HMM filtering with multiple restarts
* regime-weighted covariance and EBIC Graphical Lasso topologies
* hybrid TCN/LSTM temporal encoding, lag-specific graph experts, and residual PPO
* benchmark components for 1/N, GMV, HMM-GMV, GLASSO-GAT, TC-MAC, and JM-MPC
* financial metrics, Deflated Sharpe Ratio (DSR), and Model Confidence Set (MCS)
* graph, regime, allocation, ablation, and performance figures
"""

from __future__ import annotations

import logging
import math
import os

_native_thread_count = os.getenv("XGAT_NATIVE_THREADS", "1")
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = _native_thread_count

import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import chi2, jarque_bera, kurtosis, multivariate_normal, norm, skew
from sklearn.covariance import LedoitWolf, graphical_lasso
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning
from torch.distributions import Beta, Dirichlet, Independent, Normal


ArrayLike = np.ndarray | Sequence[float]
PathLike = str | Path


# -----------------------------------------------------------------------------
# 1. Configuration and Result Containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PortfolioConstraints:
    max_cash: float = 0.30
    minimum_weight: float = 0.0
    maximum_risk_weight: float = 1.0
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not 0.0 <= self.max_cash <= 1.0:
            raise ValueError("max_cash must lie in [0, 1].")
        if not 0.0 <= self.minimum_weight < 1.0:
            raise ValueError("minimum_weight must lie in [0, 1).")
        if not 0.0 < self.maximum_risk_weight <= 1.0:
            raise ValueError("maximum_risk_weight must lie in (0, 1].")
        if self.minimum_weight > self.maximum_risk_weight:
            raise ValueError("minimum_weight cannot exceed maximum_risk_weight.")
        if self.minimum_weight > self.max_cash:
            raise ValueError("minimum_weight cannot exceed max_cash.")


@dataclass(frozen=True)
class PPOConfig:
    clip_epsilon: float = 0.20
    value_coefficient: float = 0.50
    entropy_coefficient: float = 0.002
    value_clip: float | None = 0.20
    max_gradient_norm: float = 0.50
    target_kl: float | None = 0.02


@dataclass(frozen=True)
class GlassoResult:
    covariance: np.ndarray
    precision: np.ndarray
    alpha: float
    ebic: float
    n_edges: int
    n_samples: int


@dataclass(frozen=True)
class GraphRepresentation:
    signed_partial_correlation: np.ndarray
    adjacency: np.ndarray
    edge_mask: np.ndarray


@dataclass(frozen=True)
class BootstrapResult:
    metric: str
    estimate: float
    bootstrap_mean: float
    standard_error: float
    confidence_low: float
    confidence_high: float
    probability_superior: float
    n_bootstrap: int
    block_length: int


@dataclass(frozen=True)
class PredictiveGraphResult:
    """Directed predictive graph, retaining separate lag channels when available."""

    signed_coefficients: np.ndarray
    adjacency: np.ndarray
    edge_mask: np.ndarray
    stability: np.ndarray
    selected_alphas: np.ndarray
    effective_samples: float
    detection_threshold: float
    lagged_signed_coefficients: np.ndarray | None = None
    lagged_adjacency: np.ndarray | None = None
    lagged_stability: np.ndarray | None = None
    lagged_p_values: np.ndarray | None = None
    lagged_null_thresholds: np.ndarray | None = None


# -----------------------------------------------------------------------------
# 2. Validation
# -----------------------------------------------------------------------------

def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch reproducibly."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = bool(deterministic)
        torch.backends.cudnn.benchmark = False


def _as_finite_array(
    values: ArrayLike,
    *,
    name: str,
    ndim: int | None = None,
    minimum_length: int | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; received {array.ndim}.")
    if minimum_length is not None and array.shape[0] < minimum_length:
        raise ValueError(f"{name} must contain at least {minimum_length} observations.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _constraint_bounds(
    n_assets: int,
    constraints: PortfolioConstraints,
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(n_assets, constraints.minimum_weight, dtype=float)
    upper = np.full(n_assets, constraints.maximum_risk_weight, dtype=float)
    upper[-1] = constraints.max_cash
    if lower.sum() > 1.0 + constraints.tolerance or upper.sum() < 1.0 - constraints.tolerance:
        raise ValueError("Portfolio constraints make the unit simplex infeasible.")
    return lower, upper


def validate_weights(
    weights: ArrayLike,
    *,
    constraints: PortfolioConstraints = PortfolioConstraints(),
    expected_assets: int | None = None,
) -> np.ndarray:
    vector = _as_finite_array(weights, name="weights", ndim=1)
    if expected_assets is not None and vector.size != expected_assets:
        raise ValueError(f"Expected {expected_assets} weights.")
    lower, upper = _constraint_bounds(vector.size, constraints)
    if np.any(vector < lower - constraints.tolerance) or np.any(vector > upper + constraints.tolerance):
        raise ValueError("A portfolio weight violates its bounds.")
    if abs(float(vector.sum()) - 1.0) > constraints.tolerance:
        raise ValueError("Portfolio weights must sum to one.")
    return vector


def _normalise_current_holdings(
    weights: ArrayLike,
    *,
    expected_assets: int,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Validate actual post-return holdings without imposing target caps.

    Portfolio constraints such as ``max_cash`` apply to a *new target
    allocation*. Actual holdings drift after asset returns and can therefore
    lie outside those target bounds until the next rebalance. Optimisers must
    retain those holdings when computing turnover rather than rejecting them.
    """
    vector = _as_finite_array(weights, name="current_weights", ndim=1)
    if vector.size != int(expected_assets):
        raise ValueError(f"Expected {expected_assets} current portfolio weights.")
    if np.any(vector < -tolerance):
        minimum = float(np.min(vector))
        raise ValueError(
            "Current portfolio holdings contain a materially negative weight "
            f"({minimum:.6g})."
        )
    vector = np.clip(vector, 0.0, None)
    total = float(vector.sum())
    if not np.isfinite(total) or total <= tolerance:
        raise ValueError("Current portfolio holdings do not have positive mass.")
    return vector / total


def project_long_only_weights(
    weights: ArrayLike,
    *,
    constraints: PortfolioConstraints = PortfolioConstraints(),
) -> np.ndarray:
    """Project arbitrary scores onto the constrained long-only simplex."""
    vector = _as_finite_array(weights, name="weights", ndim=1)
    lower, upper = _constraint_bounds(vector.size, constraints)
    
    left = float(np.min(vector - upper))
    right = float(np.max(vector - lower))
    for _ in range(100):
        midpoint = 0.5 * (left + right)
        projected = np.clip(vector - midpoint, lower, upper)
        if projected.sum() > 1.0:
            left = midpoint
        else:
            right = midpoint

    projected = np.clip(vector - 0.5 * (left + right), lower, upper)
    for _ in range(vector.size + 2):
        residual = 1.0 - float(projected.sum())
        if abs(residual) <= constraints.tolerance:
            break
        slack = upper - projected if residual > 0.0 else projected - lower
        free = slack > constraints.tolerance
        if not free.any():
            break
        allocation = slack[free] / slack[free].sum()
        projected[free] += np.sign(residual) * min(abs(residual), slack[free].sum()) * allocation
        projected = np.clip(projected, lower, upper)

    return projected


# -----------------------------------------------------------------------------
# 3. Data Preprocessing and Statistical Diagnostics
# -----------------------------------------------------------------------------

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    numeric = prices.astype(float)
    if (numeric <= 0.0).any().any():
        raise ValueError("Prices must be strictly positive.")
    return np.log(numeric / numeric.shift(1)).dropna()


def robust_scale_tensor(
    tensor: np.ndarray,
    training_tensor: np.ndarray,
    *,
    epsilon: float = 1e-8,
    clip: float | None = 20.0,
) -> np.ndarray:
    """Standardise avoiding look-ahead bias using training-only distributions."""
    values = np.asarray(tensor, dtype=float)
    training = np.asarray(training_tensor, dtype=float)
    
    median = np.median(training, axis=0, keepdims=True)
    q75 = np.percentile(training, 75, axis=0, keepdims=True)
    q25 = np.percentile(training, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    safe_iqr = np.where(iqr > epsilon, iqr, 1.0)
    
    scaled = (values - median) / safe_iqr
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    if clip is not None:
        scaled = np.clip(scaled, -clip, clip)
    return scaled


def build_causal_feature_tensor(
    log_returns: np.ndarray,
    *,
    window: int,
    include_volatility: bool = True,
    include_momentum: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Create fully causal, time-varying features for every point in each window.
    """
    returns = _as_finite_array(log_returns, name="log_returns", ndim=2)
    n_times, n_total_assets = returns.shape
    n_risk = n_total_assets - 1
    if window < 2 or n_times <= window:
        raise ValueError("window is incompatible with the return history.")

    risk = returns[:, :n_risk]
    names = ["log_return", "squared_return", "downside_squared"]
    feature_series: list[np.ndarray] = [
        risk,
        risk**2,
        np.minimum(risk, 0.0) ** 2,
    ]

    if include_momentum:
        for horizon in (5, 20):
            rolling = pd.DataFrame(risk).rolling(horizon, min_periods=1).sum().to_numpy()
            feature_series.append(rolling)
            names.append(f"momentum_{horizon}")
    if include_volatility:
        for horizon in (5, 20):
            rolling = (
                pd.DataFrame(risk)
                .rolling(horizon, min_periods=2)
                .std(ddof=1)
                .fillna(0.0)
                .to_numpy()
            )
            feature_series.append(rolling)
            names.append(f"volatility_{horizon}")

    stacked = np.stack(feature_series, axis=-1).astype(np.float32)
    tensor = np.zeros((n_times, n_risk, window, len(names)), dtype=np.float32)
    for t in range(window, n_times):
        tensor[t] = np.transpose(stacked[t - window : t], (1, 0, 2))
    return tensor, names


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (values.size - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def ljung_box_test(series: ArrayLike, lag: int = 20) -> tuple[float, float]:
    values = _as_finite_array(series, name="series", ndim=1, minimum_length=lag + 2)
    centred = values - values.mean()
    denominator = float(np.dot(centred, centred))
    if denominator <= 1e-20:
        return 0.0, 1.0
        
    correlations = [
        float(np.dot(centred[l:], centred[:-l])) / denominator
        for l in range(1, lag + 1)
    ]
    q_stat = values.size * (values.size + 2.0) * sum(
        c**2 / (values.size - l) for l, c in enumerate(correlations, start=1)
    )
    return float(q_stat), float(chi2.sf(q_stat, lag))


def run_statistical_tests(returns: pd.DataFrame, *, ljung_box_lag: int = 20) -> pd.DataFrame:
    rows = []
    jb_ps, lin_ps, sq_ps = [], [], []

    for column in returns.columns:
        values = _as_finite_array(returns[column].dropna().to_numpy(), name=str(column), ndim=1)
        jb = jarque_bera(values)
        _, lin_p = ljung_box_test(values, lag=ljung_box_lag)
        _, sq_p = ljung_box_test(values**2, lag=ljung_box_lag)
        
        jb_ps.append(float(jb.pvalue))
        lin_ps.append(lin_p)
        sq_ps.append(sq_p)
        
        rows.append({
            "Asset": str(column),
            "Mean": float(values.mean()),
            "Standard deviation": float(values.std(ddof=1)),
            "Skewness": float(skew(values, bias=False)),
            "Excess kurtosis": float(kurtosis(values, fisher=True, bias=False)),
            "Jarque-Bera p-value": float(jb.pvalue),
            "Ljung-Box p-value": lin_p,
            "Squared-return Ljung-Box p-value": sq_p,
        })

    result = pd.DataFrame(rows)
    result["JB_Adj_P"] = _holm_adjust(jb_ps)
    result["LB_Lin_Adj_P"] = _holm_adjust(lin_ps)
    result["LB_Squared_Adj_P"] = _holm_adjust(sq_ps)
    return result


# -----------------------------------------------------------------------------
# 4. Hidden Markov Model Utilities
# -----------------------------------------------------------------------------

def build_hmm_observations(
    risk_log_returns: np.ndarray,
    *,
    volatility_window: int = 20,
    use_log_volatility: bool = True,
    volatility_method: str = "ewma",
    ewma_halflife: float = 5.0,
) -> tuple[np.ndarray, int]:
    returns = _as_finite_array(risk_log_returns, name="risk_log_returns", ndim=2)
    market_return = returns.mean(axis=1)
    series = pd.Series(market_return)
    
    if volatility_method == "rolling":
        volatility = series.rolling(volatility_window, min_periods=volatility_window).std(ddof=1).to_numpy()
    else:
        min_periods = max(3, min(volatility_window, int(round(ewma_halflife))))
        volatility = series.ewm(halflife=ewma_halflife, adjust=False, min_periods=min_periods).std(bias=False).to_numpy()

    if use_log_volatility:
        volatility = np.log(np.maximum(volatility, 1e-12))
        
    observations = np.column_stack([market_return, volatility])
    finite_rows = np.flatnonzero(np.all(np.isfinite(observations), axis=1))
    return observations, int(finite_rows[0])


# -----------------------------------------------------------------------------
# 14. Dependency-free Gaussian HMM fallback
# -----------------------------------------------------------------------------

class _InternalGaussianHMM:
    """Small full-covariance Gaussian HMM used when hmmlearn is unavailable."""

    def __init__(
        self,
        n_components: int,
        *,
        n_iter: int = 300,
        tol: float = 1e-4,
        min_covar: float = 1e-5,
        transition_prior: float = 1.25,
        random_state: int = 0,
    ) -> None:
        self.n_components = int(n_components)
        self.n_iter = int(n_iter)
        self.tol = float(tol)
        self.min_covar = float(min_covar)
        self.transition_prior = float(transition_prior)
        self.random_state = int(random_state)
        self.covariance_type = "full"

    def _emission_log_prob(self, values: np.ndarray) -> np.ndarray:
        result = np.empty((values.shape[0], self.n_components), dtype=float)
        for state in range(self.n_components):
            result[:, state] = multivariate_normal.logpdf(
                values,
                mean=self.means_[state],
                cov=self.covars_[state],
                allow_singular=False,
            )
        return result

    def _forward_backward(self, values: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        log_emission = self._emission_log_prob(values)
        log_transition = np.log(np.clip(self.transmat_, 1e-12, None))
        log_start = np.log(np.clip(self.startprob_, 1e-12, None))
        length = values.shape[0]
        log_alpha = np.empty((length, self.n_components), dtype=float)
        log_alpha[0] = log_start + log_emission[0]
        for t in range(1, length):
            for state in range(self.n_components):
                log_alpha[t, state] = log_emission[t, state] + logsumexp(
                    log_alpha[t - 1] + log_transition[:, state]
                )
        log_likelihood = float(logsumexp(log_alpha[-1]))

        log_beta = np.zeros((length, self.n_components), dtype=float)
        for t in range(length - 2, -1, -1):
            for state in range(self.n_components):
                log_beta[t, state] = logsumexp(
                    log_transition[state] + log_emission[t + 1] + log_beta[t + 1]
                )
        log_gamma = log_alpha + log_beta - log_likelihood
        gamma = np.exp(log_gamma)
        gamma /= gamma.sum(axis=1, keepdims=True).clip(min=1e-12)

        xi_sum = np.zeros((self.n_components, self.n_components), dtype=float)
        for t in range(length - 1):
            log_xi = (
                log_alpha[t][:, None]
                + log_transition
                + log_emission[t + 1][None, :]
                + log_beta[t + 1][None, :]
                - log_likelihood
            )
            xi = np.exp(log_xi - logsumexp(log_xi))
            xi_sum += xi
        return log_likelihood, gamma, xi_sum

    def fit(self, values: np.ndarray) -> "_InternalGaussianHMM":
        from sklearn.cluster import KMeans

        data = np.asarray(values, dtype=float)
        if data.ndim != 2 or data.shape[0] < max(10, 3 * self.n_components):
            raise ValueError("The HMM training sample is too short.")

        rng = np.random.default_rng(self.random_state)
        labels = KMeans(
            n_clusters=self.n_components,
            n_init=10,
            random_state=self.random_state,
        ).fit_predict(data)

        self.startprob_ = np.full(self.n_components, 1.0 / self.n_components)
        self.transmat_ = np.full(
            (self.n_components, self.n_components),
            1.0 / self.n_components,
        )
        self.means_ = np.empty((self.n_components, data.shape[1]), dtype=float)
        self.covars_ = np.empty(
            (self.n_components, data.shape[1], data.shape[1]),
            dtype=float,
        )

        global_covariance = np.atleast_2d(np.cov(data, rowvar=False, ddof=1))
        global_covariance = 0.5 * (global_covariance + global_covariance.T)
        global_covariance += self.min_covar * np.eye(data.shape[1])

        for state in range(self.n_components):
            members = data[labels == state]
            if members.shape[0] < 2:
                self.means_[state] = data[rng.integers(0, data.shape[0])]
                self.covars_[state] = global_covariance
            else:
                self.means_[state] = members.mean(axis=0)
                covariance = np.atleast_2d(np.cov(members, rowvar=False, ddof=1))
                covariance = 0.5 * (covariance + covariance.T)
                self.covars_[state] = covariance + self.min_covar * np.eye(data.shape[1])

        history: list[float] = []
        converged = False
        for _ in range(self.n_iter):
            likelihood, gamma, xi_sum = self._forward_backward(data)
            if not np.isfinite(likelihood):
                raise FloatingPointError("The HMM log-likelihood is not finite.")

            history.append(float(likelihood))
            if len(history) >= 2:
                improvement = history[-1] - history[-2]
                relative_tolerance = self.tol * (1.0 + abs(history[-2]))
                if abs(improvement) <= relative_tolerance:
                    converged = True
                    break

            weights = gamma.sum(axis=0).clip(min=1e-8)
            self.startprob_ = gamma[0] + self.transition_prior
            self.startprob_ /= self.startprob_.sum()

            transition = xi_sum + self.transition_prior
            self.transmat_ = transition / transition.sum(axis=1, keepdims=True)
            self.means_ = (gamma.T @ data) / weights[:, None]

            for state in range(self.n_components):
                centred = data - self.means_[state]
                covariance = (
                    centred * gamma[:, state : state + 1]
                ).T @ centred / weights[state]
                covariance = 0.995 * covariance + 0.005 * global_covariance
                covariance = 0.5 * (covariance + covariance.T)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                self.covars_[state] = (
                    eigenvectors
                    * np.clip(eigenvalues, self.min_covar, None)
                ) @ eigenvectors.T

        final_score = float(self.score(data))
        self._xgat_converged = bool(converged)
        self._xgat_iterations = len(history)
        self._xgat_likelihood_history = tuple(history)
        self._training_score = final_score
        return self

    def score(self, values: np.ndarray) -> float:
        return float(self._forward_backward(np.asarray(values, dtype=float))[0])


def fit_hmm_with_restarts(
    observations: np.ndarray,
    *,
    n_components: int = 2,
    seeds: Iterable[int] | None = None,
    covariance_type: str = "full",
    n_iter: int = 1_000,
    tolerance: float = 1e-3,
    minimum_covariance: float = 1e-4,
    transition_prior: float = 1.50,
    robust_clip: float | None = 6.0,
) -> Any:
    """Fit a full-covariance Gaussian HMM with checked restarts."""
    if covariance_type != "full":
        raise ValueError("Only covariance_type='full' is supported.")

    values = _as_finite_array(observations, name="observations", ndim=2)
    if values.shape[0] < max(20, 5 * n_components):
        raise ValueError("The HMM training sample is too short.")

    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=1)
    scale = np.where(scale > 1e-10, scale, 1.0)
    standardised = (values - mean) / scale
    if robust_clip is not None:
        standardised = np.clip(standardised, -robust_clip, robust_clip)

    restart_seeds = list(seeds) if seeds is not None else [11, 23, 37, 53, 71]
    if not restart_seeds:
        raise ValueError("At least one HMM restart seed is required.")

    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        GaussianHMM = None

    def candidate_is_valid(model: Any, score: float) -> bool:
        if not np.isfinite(score):
            return False
        if not np.all(np.isfinite(np.asarray(model.means_, dtype=float))):
            return False
        transition = np.asarray(model.transmat_, dtype=float)
        start_probability = np.asarray(model.startprob_, dtype=float)
        if (
            transition.shape != (n_components, n_components)
            or start_probability.shape != (n_components,)
            or not np.all(np.isfinite(transition))
            or not np.all(np.isfinite(start_probability))
        ):
            return False
        if np.any(transition < 0.0) or np.any(start_probability < 0.0):
            return False
        if not np.allclose(transition.sum(axis=1), 1.0, atol=1e-6):
            return False
        if not np.isclose(start_probability.sum(), 1.0, atol=1e-6):
            return False
        for state in range(n_components):
            covariance = _hmm_state_covariance(
                model,
                state,
                standardised.shape[1],
            )
            if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
                return False
        return True

    best_model: Any | None = None
    best_score = -np.inf
    best_diagnostics: dict[str, Any] | None = None

    backends = ["hmmlearn", "internal"] if GaussianHMM is not None else ["internal"]
    for backend in backends:
        candidates_found = 0
        for restart_seed in restart_seeds:
            try:
                if backend == "hmmlearn":
                    model = GaussianHMM(
                        n_components=n_components,
                        covariance_type="full",
                        n_iter=n_iter,
                        tol=tolerance,
                        min_covar=minimum_covariance,
                        transmat_prior=np.full(
                            (n_components, n_components),
                            transition_prior,
                            dtype=float,
                        ),
                        startprob_prior=np.full(
                            n_components,
                            transition_prior,
                            dtype=float,
                        ),
                        random_state=int(restart_seed),
                        verbose=False,
                    )
                    hmm_logger = logging.getLogger("hmmlearn.base")
                    previous_level = hmm_logger.level
                    try:
                        hmm_logger.setLevel(logging.ERROR)
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message=r"Model is not converging.*",
                            )
                            model.fit(standardised)
                    finally:
                        hmm_logger.setLevel(previous_level)

                    history = tuple(
                        float(item)
                        for item in getattr(
                            getattr(model, "monitor_", None),
                            "history",
                            (),
                        )
                    )
                    final_delta = (
                        history[-1] - history[-2]
                        if len(history) >= 2
                        else float("nan")
                    )
                    decrease_tolerance = max(
                        tolerance,
                        1e-8 * (1.0 + abs(history[-1]))
                        if history
                        else tolerance,
                    )
                    if np.isfinite(final_delta) and final_delta < -decrease_tolerance:
                        continue
                    converged = bool(
                        getattr(getattr(model, "monitor_", None), "converged", False)
                    )
                    if not converged and np.isfinite(final_delta):
                        converged = abs(final_delta) <= decrease_tolerance
                    if not converged:
                        continue
                    iterations = len(history)
                else:
                    model = _InternalGaussianHMM(
                        n_components,
                        n_iter=min(int(n_iter), 300),
                        tol=tolerance,
                        min_covar=minimum_covariance,
                        transition_prior=transition_prior,
                        random_state=int(restart_seed),
                    )
                    model.fit(standardised)
                    history = tuple(
                        getattr(model, "_xgat_likelihood_history", ())
                    )
                    final_delta = (
                        history[-1] - history[-2]
                        if len(history) >= 2
                        else float("nan")
                    )
                    converged = bool(getattr(model, "_xgat_converged", False))
                    iterations = int(getattr(model, "_xgat_iterations", len(history)))
                    internal_tolerance = max(
                        10.0 * tolerance,
                        1e-8 * (1.0 + abs(history[-1]))
                        if history
                        else 10.0 * tolerance,
                    )
                    if not converged and np.isfinite(final_delta):
                        converged = abs(final_delta) <= internal_tolerance
                    if not converged:
                        continue

                score = float(model.score(standardised))
                if not candidate_is_valid(model, score):
                    continue

                candidates_found += 1
                if score > best_score:
                    best_model = model
                    best_score = score
                    best_diagnostics = {
                        "backend": backend,
                        "seed": int(restart_seed),
                        "converged": bool(converged),
                        "iterations": int(iterations),
                        "final_log_likelihood_change": float(final_delta),
                        "score": float(score),
                    }
            except (
                ValueError,
                RuntimeError,
                FloatingPointError,
                np.linalg.LinAlgError,
            ):
                continue

        if candidates_found > 0:
            break

    if best_model is None or best_diagnostics is None:
        raise RuntimeError("Every Gaussian HMM restart failed validation.")

    best_model._xgat_observation_mean = mean
    best_model._xgat_observation_scale = scale
    best_model._xgat_training_score = best_score
    best_model._xgat_hmm_diagnostics = best_diagnostics
    return best_model


def _hmm_state_covariance(model: Any, state: int, n_features: int) -> np.ndarray:
    covariances = np.asarray(model.covars_, dtype=float)
    cov_type = getattr(model, "covariance_type", None)

    if cov_type == "tied": candidate = covariances
    elif cov_type in {"full", "diag", "spherical"}: candidate = covariances[state]
    else: candidate = covariances[state] if covariances.ndim >= 3 else covariances

    if candidate.ndim == 0:
        covariance = np.eye(n_features, dtype=float) * float(candidate)
    elif candidate.ndim == 1:
        if candidate.size == 1: covariance = np.eye(n_features, dtype=float) * float(candidate[0])
        else: covariance = np.diag(candidate)
    elif candidate.shape == (n_features, n_features):
        covariance = candidate
    else:
        raise ValueError("Invalid HMM covariance shape.")

    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return (eigenvectors * np.clip(eigenvalues, 1e-8, None)) @ eigenvectors.T


def filtered_hmm_probabilities(
    model: Any,
    observations: np.ndarray,
    *,
    start_index: int = 0,
    initial_fill: str = "nan",
) -> np.ndarray:
    """Strictly causal Hamilton-filter probabilities."""
    values = np.asarray(observations, dtype=float)
    mean = getattr(model, "_xgat_observation_mean", np.zeros(values.shape[1]))
    scale = getattr(model, "_xgat_observation_scale", np.ones(values.shape[1]))
    transformed = (values - mean) / scale

    n_states = int(model.n_components)
    transition = np.clip(np.asarray(model.transmat_, dtype=float), 1e-12, None)
    transition /= transition.sum(axis=1, keepdims=True)
    start_prob = np.clip(np.asarray(model.startprob_, dtype=float), 1e-12, None)
    start_prob /= start_prob.sum()
    means = np.asarray(model.means_, dtype=float)

    filtered = np.full((values.shape[0], n_states), np.nan, dtype=float)
    if initial_fill == "start" and start_index > 0:
        filtered[:start_index] = start_prob

    log_transition = np.log(transition)
    prev_log_alpha: np.ndarray | None = None
    
    for t in range(start_index, values.shape[0]):
        log_emission = np.empty(n_states, dtype=float)
        for state in range(n_states):
            cov = _hmm_state_covariance(model, state, transformed.shape[1])
            log_emission[state] = multivariate_normal.logpdf(
                transformed[t], mean=means[state], cov=cov, allow_singular=False
            )

        if prev_log_alpha is None:
            log_pred = np.log(start_prob)
        else:
            log_pred = np.array([
                logsumexp(prev_log_alpha + log_transition[:, state])
                for state in range(n_states)
            ], dtype=float)
            
        log_alpha = log_pred + log_emission
        log_alpha -= logsumexp(log_alpha)
        filtered[t] = np.exp(log_alpha)
        prev_log_alpha = log_alpha
        
    return filtered


# -----------------------------------------------------------------------------
# 5. Covariance, Precision, and Graph Construction
# -----------------------------------------------------------------------------

def _stabilise_covariance(
    covariance: np.ndarray,
    *,
    minimum_eigenvalue: float = 1e-8,
    maximum_condition_number: float = 1e7,
) -> np.ndarray:
    matrix = _as_finite_array(covariance, name="covariance", ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("covariance must be square.")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    largest = max(float(np.max(eigenvalues)), minimum_eigenvalue)
    floor = max(minimum_eigenvalue, largest / maximum_condition_number)
    adjusted = (eigenvectors * np.clip(eigenvalues, floor, None)) @ eigenvectors.T
    return 0.5 * (adjusted + adjusted.T)


def estimate_covariance(returns: np.ndarray, *, method: str = "ledoit_wolf") -> np.ndarray:
    values = _as_finite_array(returns, name="returns", ndim=2, minimum_length=2)
    if method == "sample":
        covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    elif method == "ledoit_wolf":
        covariance = LedoitWolf().fit(values).covariance_
    else:
        raise ValueError("method must be 'sample' or 'ledoit_wolf'.")
    return _stabilise_covariance(covariance)


def compute_regime_weighted_covariance(
    returns: np.ndarray,
    state_probabilities: ArrayLike,
    *,
    minimum_effective_samples: float = 30.0,
    fallback_covariance: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    values = _as_finite_array(returns, name="returns", ndim=2, minimum_length=2)
    probabilities = _as_finite_array(state_probabilities, name="state_probs", ndim=1)
    
    weights = np.clip(probabilities, 0.0, None)
    fallback = estimate_covariance(values, method="ledoit_wolf") if fallback_covariance is None else fallback_covariance

    weight_sum = float(weights.sum())
    if weight_sum <= 1e-12:
        return fallback.copy(), 0.0
        
    normalised = weights / weight_sum
    sq_weight_sum = float(np.sum(normalised**2))
    eff_samples = float(1.0 / sq_weight_sum)
    
    weighted_mean = np.sum(values * normalised[:, None], axis=0)
    centred = values - weighted_mean
    denominator = 1.0 - sq_weight_sum
    if denominator <= 1e-12:
        return fallback.copy(), eff_samples
        
    weighted_covariance = (centred * normalised[:, None]).T @ centred
    weighted_covariance /= denominator
    weighted_covariance = 0.5 * (weighted_covariance + weighted_covariance.T)

    # Shrinkage towards unconditional if effective samples are too low
    if eff_samples < minimum_effective_samples:
        weight = np.clip(eff_samples / minimum_effective_samples, 0.0, 1.0)
        weighted_covariance = weight * weighted_covariance + (1.0 - weight) * fallback

    return _stabilise_covariance(weighted_covariance), eff_samples


def _covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    matrix = _stabilise_covariance(covariance, maximum_condition_number=1e4)
    std_dev = np.sqrt(np.clip(np.diag(matrix), 1e-12, None))
    correlation = matrix / np.outer(std_dev, std_dev)
    correlation = np.clip(correlation, -0.999, 0.999)
    np.fill_diagonal(correlation, 1.0)
    correlation = _stabilise_covariance(
        correlation, minimum_eigenvalue=1e-4, maximum_condition_number=1e4
    )
    scale = np.sqrt(np.clip(np.diag(correlation), 1e-12, None))
    correlation = correlation / np.outer(scale, scale)
    np.fill_diagonal(correlation, 1.0)
    return 0.5 * (correlation + correlation.T)


def fit_glasso_at_alpha(
    covariance: np.ndarray,
    *,
    n_samples: int,
    alpha: float,
    ebic_gamma: float = 0.50,
    edge_tolerance: float = 1e-7,
) -> GlassoResult:
    """Fit Graphical Lasso at a predeclared penalty and report its EBIC.

    Keeping the penalty fixed between structural refresh dates avoids repeated
    hyperparameter searches while preserving a causal rolling graph path.
    """
    if n_samples <= 1:
        raise ValueError("n_samples must exceed one.")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("alpha must be finite and positive.")
    if not 0.0 <= ebic_gamma <= 1.0:
        raise ValueError("ebic_gamma must lie in [0, 1].")

    empirical = _covariance_to_correlation(covariance)
    n_features = empirical.shape[0]
    identity = np.eye(n_features, dtype=float)
    # Retry with light shrinkage rather than failing a long replication on a
    # nearly singular rolling covariance matrix.
    for shrinkage in (0.0, 5e-3, 2e-2):
        candidate = (1.0 - shrinkage) * empirical + shrinkage * identity
        candidate = _stabilise_covariance(
            candidate,
            minimum_eigenvalue=max(1e-6, shrinkage * 1e-3),
            maximum_condition_number=1e6,
        )
        scale = np.sqrt(np.clip(np.diag(candidate), 1e-12, None))
        candidate = candidate / np.outer(scale, scale)
        np.fill_diagonal(candidate, 1.0)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                estimated_covariance, precision = graphical_lasso(
                    emp_cov=candidate, alpha=float(alpha), max_iter=300, tol=1e-3
                )
            estimated_covariance = _stabilise_covariance(
                estimated_covariance, maximum_condition_number=1e8
            )
            precision = _stabilise_covariance(
                precision, maximum_condition_number=1e8
            )
            if not np.all(np.isfinite(estimated_covariance)) or not np.all(np.isfinite(precision)):
                raise FloatingPointError("Graphical Lasso returned non-finite matrices.")
            sign, log_determinant = np.linalg.slogdet(precision)
            if sign <= 0 or not np.isfinite(log_determinant):
                raise np.linalg.LinAlgError("Graphical Lasso precision is not positive definite.")
            log_likelihood = 0.5 * n_samples * (log_determinant - np.trace(empirical @ precision))
            n_edges = int(np.triu(np.abs(precision) > edge_tolerance, k=1).sum())
            n_parameters = n_features + n_edges
            ebic = -2.0 * log_likelihood + n_parameters * np.log(n_samples) + 4.0 * ebic_gamma * n_edges * np.log(n_features)
            return GlassoResult(
                covariance=0.5 * (estimated_covariance + estimated_covariance.T),
                precision=0.5 * (precision + precision.T), alpha=float(alpha),
                ebic=float(ebic), n_edges=n_edges, n_samples=int(n_samples),
            )
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            continue

    # Valid conservative fallback: no conditional-dependence edges.
    diagonal = np.clip(np.diag(empirical), 1e-8, None)
    precision = np.diag(1.0 / diagonal)
    log_likelihood = 0.5 * n_samples * (np.linalg.slogdet(precision)[1] - np.trace(empirical @ precision))
    return GlassoResult(
        covariance=np.diag(diagonal), precision=precision, alpha=float(alpha),
        ebic=float(-2.0 * log_likelihood + n_features * np.log(n_samples)),
        n_edges=0, n_samples=int(n_samples),
    )


def fit_ebic_glasso(
    covariance: np.ndarray,
    *,
    n_samples: int,
    alphas: Iterable[float] | None = None,
    ebic_gamma: float = 0.50,
    edge_tolerance: float = 1e-7,
) -> GlassoResult:
    """Fit Graphical Lasso and select structural sparsity by EBIC."""
    if n_samples <= 1:
        raise ValueError("n_samples must exceed one.")
    if not 0.0 <= ebic_gamma <= 1.0:
        raise ValueError("ebic_gamma must lie in [0, 1].")
    alpha_grid = (
        np.asarray(list(alphas), dtype=float)
        if alphas is not None
        else np.geomspace(2e-4, 2e-1, 10)
    )
    if alpha_grid.size == 0 or np.any(~np.isfinite(alpha_grid)) or np.any(alpha_grid <= 0.0):
        raise ValueError("All Graphical Lasso penalties must be finite and positive.")

    best: GlassoResult | None = None
    for alpha in alpha_grid:
        try:
            candidate = fit_glasso_at_alpha(
                covariance,
                n_samples=n_samples,
                alpha=float(alpha),
                ebic_gamma=ebic_gamma,
                edge_tolerance=edge_tolerance,
            )
        except (
            ConvergenceWarning,
            FloatingPointError,
            ValueError,
            np.linalg.LinAlgError,
        ):
            continue
        if best is None or candidate.ebic < best.ebic:
            best = candidate

    if best is None:
        empirical = _covariance_to_correlation(covariance)
        precision = np.linalg.pinv(empirical)
        estimated_covariance = np.linalg.pinv(precision)
        n_edges = int(np.triu(np.abs(precision) > edge_tolerance, k=1).sum())
        best = GlassoResult(
            covariance=0.5 * (estimated_covariance + estimated_covariance.T),
            precision=0.5 * (precision + precision.T),
            alpha=float("nan"),
            ebic=float("nan"),
            n_edges=n_edges,
            n_samples=int(n_samples),
        )
    return best

def precision_to_graph(
    precision: np.ndarray,
    *,
    threshold: float = 0.03,
    include_self_loops: bool = True,
) -> GraphRepresentation:
    matrix = _as_finite_array(precision, name="precision", ndim=2)
    matrix = 0.5 * (matrix + matrix.T)
    diagonal = np.sqrt(np.diag(matrix))
    signed = -matrix / np.outer(diagonal, diagonal)
    signed = np.clip(0.5 * (signed + signed.T), -1.0, 1.0)
    np.fill_diagonal(signed, 0.0)
    
    mask = np.abs(signed) >= threshold
    np.fill_diagonal(mask, include_self_loops)
    adjacency = np.abs(signed) * mask
    if include_self_loops:
        np.fill_diagonal(adjacency, 1.0)
        
    return GraphRepresentation(
        signed_partial_correlation=signed,
        adjacency=adjacency,
        edge_mask=mask,
    )


# -----------------------------------------------------------------------------
# 6. Neural Building Blocks (X-GAT-DRL)
# -----------------------------------------------------------------------------


class DenseGATLayer(nn.Module):
    """Edge-conditioned GATv2 layer with a continuous statistical graph prior."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        negative_slope: float = 0.20,
        dropout: float = 0.0,
        prior_floor: float = 1e-4,
    ) -> None:
        super().__init__()
        if min(in_features, out_features) <= 0:
            raise ValueError("GAT dimensions must be positive.")
        self.query = nn.Linear(in_features, out_features, bias=False)
        self.key = nn.Linear(in_features, out_features, bias=False)
        self.positive_value = nn.Linear(in_features, out_features, bias=False)
        self.negative_value = nn.Linear(in_features, out_features, bias=False)
        self.edge_projection = nn.Linear(3, out_features, bias=False)
        self.attention_vector = nn.Parameter(torch.empty(out_features))
        self.prior_strength_raw = nn.Parameter(torch.tensor(0.0))
        self.negative_gate = nn.Parameter(torch.tensor(0.0))
        self.attention_temperature = nn.Parameter(torch.tensor(0.0))
        self.residual = (
            nn.Linear(in_features, out_features, bias=False)
            if in_features != out_features
            else nn.Identity()
        )
        self.normalisation = nn.LayerNorm(out_features)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.dropout = nn.Dropout(dropout)
        self.prior_floor = float(prior_floor)
        for module in (
            self.query,
            self.key,
            self.positive_value,
            self.negative_value,
            self.edge_projection,
        ):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
        nn.init.xavier_uniform_(self.attention_vector.view(1, -1), gain=1.0)

    def forward(self, node_features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        if node_features.ndim != 3:
            raise ValueError("node_features must have shape [batch, nodes, features].")
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0)
        batch, n_nodes, _ = node_features.shape
        if adjacency.size(0) == 1 and batch > 1:
            adjacency = adjacency.expand(batch, -1, -1)
        if adjacency.shape != (batch, n_nodes, n_nodes):
            raise ValueError("adjacency has an incompatible shape.")
        if not torch.isfinite(node_features).all() or not torch.isfinite(adjacency).all():
            raise ValueError("GAT inputs contain non-finite values.")

        adjacency = adjacency.clone()
        diagonal = torch.arange(n_nodes, device=adjacency.device)
        adjacency[:, diagonal, diagonal] = torch.where(
            adjacency[:, diagonal, diagonal].abs() < self.prior_floor,
            torch.ones_like(adjacency[:, diagonal, diagonal]),
            adjacency[:, diagonal, diagonal],
        )
        magnitude = adjacency.abs().clamp(max=1.0)
        sign = torch.sign(adjacency)
        edge_features = torch.stack(
            [magnitude, sign, torch.log1p(9.0 * magnitude) / math.log(10.0)], dim=-1
        )

        query = self.query(node_features).unsqueeze(2)
        key = self.key(node_features).unsqueeze(1)
        edge = self.edge_projection(edge_features)
        dynamic_pair = self.leaky_relu(query + key + edge)
        logits = torch.einsum("bijo,o->bij", dynamic_pair, self.attention_vector)
        prior_strength = F.softplus(self.prior_strength_raw) + 0.25
        logits = logits + prior_strength * torch.log(magnitude + self.prior_floor)
        
        temp = F.softplus(self.attention_temperature) + 0.05
        attention = F.softmax(logits / temp, dim=-1)
        attention = self.dropout(attention)

        positive = self.positive_value(node_features).unsqueeze(1).expand(-1, n_nodes, -1, -1)
        negative = self.negative_value(node_features).unsqueeze(1).expand(-1, n_nodes, -1, -1)
        negative_scale = torch.sigmoid(self.negative_gate)
        signed_messages = torch.where(
            (sign >= 0.0).unsqueeze(-1),
            positive,
            -negative_scale * negative,
        )
        aggregated = torch.sum(attention.unsqueeze(-1) * signed_messages, dim=2)
        output = self.normalisation(aggregated + self.residual(node_features))
        return F.silu(output)


class SpatioTemporalEncoder(nn.Module):
    """GRU temporal momentum encoder propagating risk spatially via GAT."""
    def __init__(self, num_assets: int, num_features: int, *, hidden_gru: int = 32, hidden_gat: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.num_assets = int(num_assets)
        self.gru = nn.GRU(input_size=num_features, hidden_size=hidden_gru, batch_first=True)
        self.gat1 = DenseGATLayer(hidden_gru, hidden_gat, dropout=dropout)
        self.gat2 = DenseGATLayer(hidden_gat, hidden_gat, dropout=dropout)

    def forward(self, sequence: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch, n_assets, lookback, n_features = sequence.shape
        self.gru.flatten_parameters()
        flattened = sequence.reshape(batch * n_assets, lookback, n_features)
        gru_output, _ = self.gru(flattened)
        temporal = gru_output[:, -1].reshape(batch, n_assets, -1)
        return self.gat2(self.gat1(temporal, adjacency), adjacency)


class BetaDirichletActor(nn.Module):
    """
    Bimodal Actor parametrizing a long-only portfolio on the continuous simplex.
    Employs a Beta distribution to cap Cash constraints and Dirichlet for risk-assets,
    mitigating policy collapse.
    """
    def __init__(self, state_dim: int, num_total_assets: int, *, max_cash: float = 0.30) -> None:
        super().__init__()
        self.num_risk = num_total_assets - 1
        self.max_cash = float(max_cash)

        self.backbone = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
        )
        
        self.risk_logits = nn.Linear(32, self.num_risk)
        self.risk_scale = nn.Linear(32, 1)
        self.cash_mean_logit = nn.Linear(32, 1)
        self.cash_scale = nn.Linear(32, 1)

        nn.init.zeros_(self.risk_logits.weight)
        nn.init.zeros_(self.risk_logits.bias)
        nn.init.zeros_(self.cash_mean_logit.weight)
        
        initial_unit_mean = (0.05) / self.max_cash
        initial_logit = math.log(initial_unit_mean / (1.0 - initial_unit_mean))
        nn.init.constant_(self.cash_mean_logit.bias, initial_logit)
        nn.init.zeros_(self.risk_scale.weight)
        nn.init.constant_(self.risk_scale.bias, 1.5)
        nn.init.zeros_(self.cash_scale.weight)
        nn.init.constant_(self.cash_scale.bias, 1.0)

    def _components(self, state: torch.Tensor) -> tuple[Dirichlet, Beta, torch.Tensor]:
        hidden = self.backbone(state)

        risk_target = F.softmax(self.risk_logits(hidden), dim=-1)
        risk_scale = 4.0 + torch.clamp(F.softplus(self.risk_scale(hidden)), max=50.0)
        risk_concentration = 0.05 + risk_scale * risk_target
        risk_distribution = Dirichlet(risk_concentration)

        cash_unit_mean = torch.clamp(torch.sigmoid(self.cash_mean_logit(hidden)), 1e-4, 1.0 - 1e-4)
        cash_total = 2.0 + torch.clamp(F.softplus(self.cash_scale(hidden)), max=50.0)
        cash_alpha = 0.20 + cash_unit_mean * cash_total
        cash_beta = 0.20 + (1.0 - cash_unit_mean) * cash_total
        cash_distribution = Beta(cash_alpha, cash_beta)

        return risk_distribution, cash_distribution, hidden

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        risk_distribution, cash_distribution, _ = self._components(state)
        raw_risk = risk_distribution.sample()
        raw_cash = cash_distribution.sample()
        
        cash = self.max_cash * raw_cash
        risk = raw_risk * (1.0 - cash)
        action = torch.cat([risk, cash], dim=-1)
        
        log_jacobian = torch.log(torch.tensor(self.max_cash, device=state.device)) + (self.num_risk - 1) * torch.log(torch.clamp(1.0 - cash.squeeze(-1), min=1e-8))
        log_probability = risk_distribution.log_prob(raw_risk) + cash_distribution.log_prob(raw_cash).squeeze(-1) - log_jacobian
        base_entropy = risk_distribution.entropy() + cash_distribution.entropy().squeeze(-1)
        
        return action, log_probability, base_entropy

    def evaluate(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        risk_distribution, cash_distribution, _ = self._components(state)
        cash = action[..., -1:]
        
        remaining = torch.clamp(1.0 - cash, min=1e-8)
        raw_cash = torch.clamp(cash / self.max_cash, 1e-6, 1.0 - 1e-6)
        raw_risk = torch.clamp(action[..., :-1] / remaining, min=1e-8)
        raw_risk = raw_risk / raw_risk.sum(dim=-1, keepdim=True)
        
        log_jacobian = torch.log(torch.tensor(self.max_cash, device=state.device)) + (self.num_risk - 1) * torch.log(torch.clamp(1.0 - cash.squeeze(-1), min=1e-8))
        log_probability = risk_distribution.log_prob(raw_risk) + cash_distribution.log_prob(raw_cash).squeeze(-1) - log_jacobian
        base_entropy = risk_distribution.entropy() + cash_distribution.entropy().squeeze(-1)
        
        return log_probability, base_entropy

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        risk_distribution, cash_distribution, _ = self._components(state)
        cash_mean = self.max_cash * cash_distribution.mean
        risk = risk_distribution.mean * (1.0 - cash_mean)
        return torch.cat([risk, cash_mean], dim=-1)


class Critic(nn.Module):
    """State-value network with Orthogonal Initialization and LayerNorm for stable PPO GAE."""
    def __init__(self, state_dim: int, hidden_dims: tuple[int, int] = (64, 32)) -> None:
        super().__init__()
        first_hidden, second_hidden = hidden_dims
        
        self.fc1 = nn.Linear(state_dim, first_hidden)
        self.ln1 = nn.LayerNorm(first_hidden)
        self.fc2 = nn.Linear(first_hidden, second_hidden)
        self.ln2 = nn.LayerNorm(second_hidden)
        self.output = nn.Linear(second_hidden, 1)

        # Orthogonal init stabilizes deep RL Value networks preventing negative R2
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.output.weight, gain=1.0)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        return self.output(x).squeeze(-1)




# -----------------------------------------------------------------------------
# 7. Baselines models
# -----------------------------------------------------------------------------

class GLASSOGATBenchmark(nn.Module):
    """Benchmark 3: graph-aware GRU-GAT without HMM regime probabilities."""
    def __init__(self, num_risk_assets: int, num_features: int, *, hidden_gru: int = 32, hidden_gat: int = 32, max_cash: float = 0.30, dropout: float = 0.0) -> None:
        super().__init__()
        self.encoder = SpatioTemporalEncoder(num_risk_assets, num_features, hidden_gru=hidden_gru, hidden_gat=hidden_gat, dropout=dropout)
        
        # BoundedAllocationHead handles determinism output mapping
        self.risk_logits = nn.Linear(num_risk_assets * hidden_gat, num_risk_assets)
        self.cash_logit = nn.Linear(num_risk_assets * hidden_gat, 1)
        self.max_cash = max_cash

    def forward(self, sequence: torch.Tensor, static_adjacency: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(sequence, static_adjacency).reshape(sequence.size(0), -1)
        risk_share = F.softmax(self.risk_logits(encoded), dim=-1)
        cash = torch.sigmoid(self.cash_logit(encoded)) * self.max_cash
        risk = risk_share * (1.0 - cash)
        return torch.cat([risk, cash], dim=-1)


class TCMACBenchmark(nn.Module):
    """Benchmark 4: task-context mutual actor-critic comparator."""
    def __init__(self, num_risk_assets: int, num_features: int, *, hidden_dim: int = 32, max_cash: float = 0.30, heat_kernel_theta: float = 2.0) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.heat_kernel_theta = float(heat_kernel_theta)

        self.task_gru = nn.GRU(num_features, hidden_dim, batch_first=True, bidirectional=True)
        self.task_attention = nn.Linear(hidden_dim * 2, 1)
        self.context_layer = nn.Linear(num_features, hidden_dim)
        self.context_attention = nn.Linear(hidden_dim, 1)
        fused_dim = hidden_dim * 3
        
        self.risk_logits = nn.Linear(fused_dim, num_risk_assets)
        self.cash_logit = nn.Linear(fused_dim, 1)
        self.max_cash = max_cash
        self.critic = Critic(fused_dim)
        self.discriminator = nn.Bilinear(hidden_dim * 2, hidden_dim, 1)

    def heat_kernel_adjacency(self, latest_features: torch.Tensor) -> torch.Tensor:
        normalised = latest_features / (latest_features.norm(dim=-1, keepdim=True) + 1e-8)
        distance = torch.cdist(normalised, normalised, p=2.0)
        adjacency = torch.exp(-(distance**2) / self.heat_kernel_theta)
        diagonal = torch.arange(adjacency.size(-1), device=adjacency.device)
        adjacency[:, diagonal, diagonal] = 1.0
        return adjacency

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, n_assets, lookback, n_features = sequence.shape
        self.task_gru.flatten_parameters()
        task_output, _ = self.task_gru(sequence.reshape(batch * n_assets, lookback, n_features))
        task_local = task_output[:, -1].reshape(batch, n_assets, self.hidden_dim * 2)
        task_weights = F.softmax(self.task_attention(task_local).squeeze(-1), dim=-1)
        task_global = torch.bmm(task_weights.unsqueeze(1), task_local).squeeze(1)

        latest = sequence[:, :, -1, :]
        adjacency = self.heat_kernel_adjacency(latest)
        context_local = F.relu(self.context_layer(latest))
        context_mapped = torch.matmul(adjacency, context_local)
        context_weights = F.softmax(self.context_attention(context_mapped).squeeze(-1), dim=-1)
        context_global = torch.bmm(context_weights.unsqueeze(1), context_mapped).squeeze(1)

        fused = torch.cat([task_global, context_global], dim=-1)
        
        risk_share = F.softmax(self.risk_logits(fused), dim=-1)
        cash = torch.sigmoid(self.cash_logit(fused)) * self.max_cash
        weights = torch.cat([risk_share * (1.0 - cash), cash], dim=-1)
        
        value = self.critic(fused)
        mutual_information_logits = self.discriminator(task_local, context_global.unsqueeze(1).expand(-1, n_assets, -1)).squeeze(-1)
        
        return weights, value, mutual_information_logits


class StatisticalJumpModel:
    """Benchmark 5a: Causal statistical jump model estimated by dynamic programming."""
    def __init__(self, n_components: int = 2, *, jump_penalty: float = 25.0, random_state: int = 0) -> None:
        self.n_components = int(n_components)
        self.jump_penalty = float(jump_penalty)
        self.random_state = int(random_state)
        self.centres_: np.ndarray | None = None

    def _initial_states(self, observations: np.ndarray) -> np.ndarray:
        centred = observations - observations.mean(axis=0)
        _, _, right_singular = np.linalg.svd(centred, full_matrices=False)
        score = centred @ right_singular[0]
        quantiles = np.quantile(score, np.linspace(0.0, 1.0, self.n_components + 1)[1:-1])
        return np.digitize(score, quantiles).astype(int)

    def fit_predict(self, observations: np.ndarray, *, max_iter: int = 100) -> np.ndarray:
        values = _as_finite_array(observations, name="observations", ndim=2, minimum_length=self.n_components)
        states = self._initial_states(values)
        centres = np.zeros((self.n_components, values.shape[1]), dtype=float)
        rng = np.random.default_rng(self.random_state)

        for _ in range(max_iter):
            for state in range(self.n_components):
                mask = states == state
                if mask.any(): centres[state] = values[mask].mean(axis=0)
                else: centres[state] = values[rng.integers(0, values.shape[0])]

            emission_cost = np.stack([0.5 * np.sum((values - centres[state]) ** 2, axis=1) for state in range(self.n_components)], axis=1)
            lattice = np.zeros_like(emission_cost)
            predecessor = np.zeros_like(emission_cost, dtype=int)
            lattice[0] = emission_cost[0]
            
            for t in range(1, values.shape[0]):
                for state in range(self.n_components):
                    trans_cost = lattice[t - 1] + self.jump_penalty
                    trans_cost[state] -= self.jump_penalty
                    predecessor[t, state] = int(np.argmin(trans_cost))
                    lattice[t, state] = emission_cost[t, state] + trans_cost[predecessor[t, state]]

            new_states = np.zeros(values.shape[0], dtype=int)
            new_states[-1] = int(np.argmin(lattice[-1]))
            for t in range(values.shape[0] - 2, -1, -1):
                new_states[t] = predecessor[t + 1, new_states[t + 1]]
                
            if np.array_equal(new_states, states):
                states = new_states
                break
            states = new_states

        for state in range(self.n_components):
            mask = states == state
            if mask.any(): centres[state] = values[mask].mean(axis=0)

        self.centres_ = centres.copy()
        return states.copy()

    def predict_state(self, observation: ArrayLike) -> int:
        value = _as_finite_array(observation, name="observation", ndim=1)
        if self.centres_ is None:
            raise RuntimeError("StatisticalJumpModel must be fitted before predict_state().")
        if value.size != self.centres_.shape[1]:
            raise ValueError(
                f"Expected an observation with {self.centres_.shape[1]} features; "
                f"received {value.size}."
            )
        return int(np.argmin(np.sum((self.centres_ - value) ** 2, axis=1)))


class ModelPredictiveControlOptimiser:
    """Benchmark 5b: Long-only mean-variance MPC with explicit turnover regularisation."""
    def __init__(self, n_assets: int, *, constraints: PortfolioConstraints = PortfolioConstraints(), risk_aversion: float = 5.0, turnover_penalty: float = 0.005) -> None:
        self.n_assets = int(n_assets)
        self.constraints = constraints
        self.risk_aversion = float(risk_aversion)
        self.turnover_penalty = float(turnover_penalty)

    def allocate(self, expected_returns: ArrayLike, covariance: np.ndarray, current_weights: ArrayLike) -> np.ndarray:
        mean = _as_finite_array(expected_returns, name="expected_returns", ndim=1)
        matrix = _as_finite_array(covariance, name="covariance", ndim=2)
        if mean.size != self.n_assets:
            raise ValueError(
                f"Expected {self.n_assets} expected returns; received {mean.size}."
            )
        if matrix.shape != (self.n_assets, self.n_assets):
            raise ValueError(
                "covariance must have shape "
                f"({self.n_assets}, {self.n_assets}); received {matrix.shape}."
            )

        # Current holdings are the drifted, pre-trade portfolio. They may
        # legitimately exceed a target bound (most visibly the cash cap) after
        # relative asset-price movements. Preserve them for turnover, while
        # projecting only the optimiser's feasible starting point.
        current = _normalise_current_holdings(
            current_weights,
            expected_assets=self.n_assets,
            tolerance=max(self.constraints.tolerance, 1e-10),
        )
        initial = project_long_only_weights(current, constraints=self.constraints)
        
        matrix = _stabilise_covariance(matrix)
        bounds = [(self.constraints.minimum_weight, self.constraints.maximum_risk_weight) for _ in range(self.n_assets - 1)] + [(self.constraints.minimum_weight, self.constraints.max_cash)]

        def objective(weights: np.ndarray) -> float:
            variance = float(weights @ matrix @ weights)
            turnover = 0.5 * float(np.abs(weights - current).sum())
            return float(-weights @ mean + 0.5 * self.risk_aversion * variance + self.turnover_penalty * turnover)

        result = minimize(
            objective, initial, method="SLSQP", bounds=bounds,
            constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
            options={"maxiter": 1_000, "ftol": 1e-10},
        )
        candidate = np.asarray(getattr(result, "x", initial), dtype=float)
        if (
            not bool(getattr(result, "success", False))
            or candidate.shape != initial.shape
            or not np.all(np.isfinite(candidate))
        ):
            warnings.warn(
                "SLSQP did not return a valid MPC solution; using the feasible "
                "projected current allocation.",
                RuntimeWarning,
                stacklevel=2,
            )
            candidate = initial
        allocation = project_long_only_weights(candidate, constraints=self.constraints)
        return validate_weights(
            allocation,
            constraints=self.constraints,
            expected_assets=self.n_assets,
        )


class JumpModelMPCBenchmark:
    """Benchmark 5: statistical jump-model regime detection plus MPC."""
    name = "JM-MPC"
    def __init__(self, n_assets: int, *, n_components: int = 2, jump_penalty: float = 25.0, constraints: PortfolioConstraints = PortfolioConstraints(), risk_aversion: float = 5.0, turnover_penalty: float = 0.005, random_state: int = 0) -> None:
        self.n_assets = int(n_assets)
        self.jump_model = StatisticalJumpModel(n_components, jump_penalty=jump_penalty, random_state=random_state)
        self.optimiser = ModelPredictiveControlOptimiser(n_assets, constraints=constraints, risk_aversion=risk_aversion, turnover_penalty=turnover_penalty)
        self.state_means_: list[np.ndarray] | None = None
        self.state_covariances_: list[np.ndarray] | None = None

    def fit(self, observations: np.ndarray, asset_returns: np.ndarray) -> "JumpModelMPCBenchmark":
        obs = _as_finite_array(observations, name="observations", ndim=2)
        returns = _as_finite_array(asset_returns, name="asset_returns", ndim=2)
        states = self.jump_model.fit_predict(obs)
        global_mean = returns.mean(axis=0)
        global_covariance = estimate_covariance(returns, method="ledoit_wolf")
        
        means, covariances = [], []
        for state in range(self.jump_model.n_components):
            mask = states == state
            if mask.sum() < max(5, self.n_assets + 1):
                means.append(global_mean.copy())
                covariances.append(global_covariance.copy())
            else:
                means.append(returns[mask].mean(axis=0))
                covariances.append(estimate_covariance(returns[mask], method="ledoit_wolf"))
        self.state_means_ = means
        self.state_covariances_ = covariances
        return self

    def allocate(self, observation: ArrayLike, current_weights: ArrayLike) -> np.ndarray:
        if self.state_means_ is None or self.state_covariances_ is None:
            raise RuntimeError("JumpModelMPCBenchmark must be fitted before allocate().")
        state = self.jump_model.predict_state(observation)
        return self.optimiser.allocate(self.state_means_[state], self.state_covariances_[state], current_weights)


# -----------------------------------------------------------------------------
# 8. PPO Training Utilities and Deflated Sharpe Ratio / MCS
# -----------------------------------------------------------------------------

def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_observations: int,
    skewness: float,
    kurtosis: float,
) -> float:
    """Probability that a non-annualised Sharpe exceeds its benchmark."""
    values = np.asarray([observed_sr, benchmark_sr, skewness, kurtosis], dtype=float)
    if n_observations < 3 or not np.all(np.isfinite(values)):
        return float("nan")
    denominator_variance = 1.0 - skewness * observed_sr + ((kurtosis - 1.0) / 4.0) * observed_sr**2
    if denominator_variance <= 0.0 or not np.isfinite(denominator_variance):
        return float("nan")
    statistic = (observed_sr - benchmark_sr) * math.sqrt(n_observations - 1) / math.sqrt(denominator_variance)
    return float(norm.cdf(statistic))


def expected_maximum_sharpe_ratio(
    trial_sharpes: ArrayLike,
    *,
    effective_trials: float | None = None,
) -> float:
    """Expected maximum zero-skill Sharpe for the tested model family."""
    values = np.asarray(trial_sharpes, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("trial_sharpes must contain at least one finite value.")
    n_trials = float(values.size if effective_trials is None else effective_trials)
    if not np.isfinite(n_trials) or n_trials < 1.0:
        raise ValueError("effective_trials must be finite and at least 1.")
    if n_trials <= 1.0 or values.size <= 1:
        return 0.0
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation <= 0.0 or not np.isfinite(standard_deviation):
        return 0.0
    euler_gamma = 0.5772156649015329
    first = np.clip(1.0 - 1.0 / n_trials, 1e-12, 1.0 - 1e-12)
    second = np.clip(1.0 - 1.0 / (n_trials * math.e), 1e-12, 1.0 - 1e-12)
    return float(standard_deviation * ((1.0 - euler_gamma) * norm.ppf(first) + euler_gamma * norm.ppf(second)))


def deflated_sharpe_ratio(
    observed_sr: float,
    trial_sharpes: ArrayLike,
    n_observations: int,
    skewness: float,
    kurtosis: float,
    *,
    effective_trials: float | None = None,
    return_benchmark: bool = False,
) -> float | tuple[float, float]:
    """DSR using frequency-consistent Sharpe values and multiplicity correction."""
    benchmark_sr = expected_maximum_sharpe_ratio(trial_sharpes, effective_trials=effective_trials)
    probability = probabilistic_sharpe_ratio(
        observed_sr, benchmark_sr, n_observations, skewness, kurtosis
    )
    return (probability, benchmark_sr) if return_benchmark else probability


def compute_model_confidence_set(
    losses: np.ndarray,
    alpha: float = 0.10,
    block_length: int = 20,
    bootstraps: int = 1000,
    random_state: int = 42
) -> list[int]:
    """
    Compute the Model Confidence Set
    Given an [N_obs, M_models] matrix of losses, returns the indices of the surviving models.
    """
    rng = np.random.default_rng(random_state)
    n_obs, n_models = losses.shape
    active_models = list(range(n_models))

    for step in range(n_models - 1):
        m = len(active_models)
        current_losses = losses[:, active_models]
        mean_losses = np.mean(current_losses, axis=0)
        
        # Block Bootstrap
        bootstrap_means = np.zeros((bootstraps, m))
        n_blocks = math.ceil(n_obs / block_length)
        
        for b in range(bootstraps):
            starts = rng.integers(0, n_obs - block_length + 1, size=n_blocks)
            indices = np.concatenate([np.arange(start, start + block_length) for start in starts])[:n_obs]
            bootstrap_means[b, :] = np.mean(current_losses[indices, :], axis=0)

        t_stats = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                if i != j:
                    diff_mean = mean_losses[i] - mean_losses[j]
                    diff_boot = bootstrap_means[:, i] - bootstrap_means[:, j]
                    diff_var = np.var(diff_boot, ddof=1)
                    t_stats[i, j] = diff_mean / np.sqrt(diff_var) if diff_var > 1e-12 else 0.0

        t_max_models = np.max(t_stats, axis=1)
        t_max_overall = np.max(t_max_models)
        worst_model_idx = np.argmax(t_max_models)

        t_max_boot = np.zeros(bootstraps)
        for b in range(bootstraps):
            boot_t_stats = np.zeros((m, m))
            for i in range(m):
                for j in range(m):
                    if i != j:
                        diff_boot_b = bootstrap_means[b, i] - bootstrap_means[b, j]
                        diff_mean_H0 = diff_boot_b - (mean_losses[i] - mean_losses[j]) 
                        diff_var = np.var(bootstrap_means[:, i] - bootstrap_means[:, j], ddof=1)
                        boot_t_stats[i, j] = diff_mean_H0 / np.sqrt(diff_var) if diff_var > 1e-12 else 0.0
            t_max_boot[b] = np.max(boot_t_stats)

        p_value = np.mean(t_max_boot >= t_max_overall)
        if p_value < alpha:
            active_models.pop(worst_model_idx)
        else:
            break

    return active_models


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    terminal_flags: Sequence[bool],
    next_value: float,
    gamma: float = 0.99,
    tau: float = 0.95,
    *,
    normalise: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalised Advantage Estimation with correct truncation bootstrapping."""
    reward_array = _as_finite_array(rewards, name="rewards", ndim=1, minimum_length=1)
    value_array = _as_finite_array(values, name="values", ndim=1, minimum_length=1)
    terminal_array = np.asarray(terminal_flags, dtype=bool).reshape(-1)
    if reward_array.size != value_array.size or terminal_array.size != reward_array.size:
        raise ValueError("rewards, values, and terminal_flags must have equal length.")
    if not 0.0 <= gamma <= 1.0 or not 0.0 <= tau <= 1.0:
        raise ValueError("gamma and tau must lie in [0, 1].")

    advantages = np.zeros(reward_array.size, dtype=np.float32)
    gae = 0.0
    for index in range(reward_array.size - 1, -1, -1):
        continuation = 0.0 if terminal_array[index] else 1.0
        following_value = float(next_value) if index == reward_array.size - 1 else value_array[index + 1]
        delta = reward_array[index] + gamma * following_value * continuation - value_array[index]
        gae = delta + gamma * tau * continuation * gae
        advantages[index] = gae

    returns = advantages + value_array.astype(np.float32)
    advantage_tensor = torch.as_tensor(advantages, dtype=torch.float32)
    if normalise and advantages.size > 1:
        advantage_tensor = (advantage_tensor - advantage_tensor.mean()) / (
            advantage_tensor.std(unbiased=False) + 1e-8
        )
    return torch.as_tensor(returns, dtype=torch.float32), advantage_tensor


def ppo_update_batch(
    agent: XGATDRLAgent,
    optimiser: torch.optim.Optimizer | Mapping[str, torch.optim.Optimizer],
    states: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    actions: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    *,
    config: PPOConfig = PPOConfig(),
) -> dict[str, float | bool]:
    """Perform one PPO minibatch update with Huber-stabilized Critic."""
    log_probability, entropy, values, *_ = agent.evaluate_actions_extended(
        *states, actions
    )
    
    ratio = torch.exp(log_probability - old_log_probabilities)
    unclipped_objective = ratio * advantages
    clipped_objective = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantages
    policy_loss = -torch.minimum(unclipped_objective, clipped_objective).mean()

    # Huber loss limits the influence of large residuals.
    if config.value_clip is None:
        value_loss = F.huber_loss(values, returns, delta=1.0)
    else:
        clipped_values = old_values + torch.clamp(values - old_values, -config.value_clip, config.value_clip)
        value_loss_unclipped = F.huber_loss(values, returns, delta=1.0, reduction='none')
        value_loss_clipped = F.huber_loss(clipped_values, returns, delta=1.0, reduction='none')
        value_loss = torch.maximum(value_loss_unclipped, value_loss_clipped).mean()

    entropy_bonus = entropy.mean()
    loss = policy_loss + config.value_coefficient * value_loss - config.entropy_coefficient * entropy_bonus

    active_optimisers = (
        list(optimiser.values()) if isinstance(optimiser, Mapping) else [optimiser]
    )
    for active_optimiser in active_optimisers:
        active_optimiser.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(
        [parameter for parameter in agent.parameters() if parameter.requires_grad],
        config.max_gradient_norm,
    )
    for active_optimiser in active_optimisers:
        active_optimiser.step()

    with torch.no_grad():
        approximate_kl = torch.mean((torch.exp(log_probability - old_log_probabilities) - 1.0) - (log_probability - old_log_probabilities))
        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > config.clip_epsilon).float())
        stop_for_kl = bool(config.target_kl is not None and float(approximate_kl) > config.target_kl)

    return {
        "Loss": float(loss.item()),
        "Policy loss": float(policy_loss.item()),
        "Value loss": float(value_loss.item()),
        "Base entropy": float(entropy_bonus.item()),
        "Approximate KL": float(approximate_kl.item()),
        "Clip fraction": float(clip_fraction.item()),
        "Gradient norm": float(gradient_norm.item()),
        "Stop for KL": stop_for_kl,
    }


def ppo_update_step(
    agent: XGATDRLAgent,
    optimiser: torch.optim.Optimizer,
    states: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    actions: torch.Tensor,
    old_log_probs: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    hyperparameters: Mapping[str, Any],
) -> dict[str, float | bool]:
    with torch.no_grad():
        _, _, old_values, *_ = agent.evaluate_actions_extended(*states, actions)
        
    config = PPOConfig(
        clip_epsilon=float(hyperparameters.get("clip_epsilon", 0.20)),
        value_coefficient=float(hyperparameters.get("value_coefficient", 0.50)),
        entropy_coefficient=float(hyperparameters.get("entropy_coefficient", 0.002)),
        value_clip=hyperparameters.get("value_clip"),
        max_gradient_norm=float(hyperparameters.get("max_gradient_norm", 0.50)),
        target_kl=hyperparameters.get("target_kl"),
    )
    diagnostics = ppo_update_batch(
        agent, optimiser, states, actions, old_log_probs, old_values, returns, advantages, config=config
    )
    return {**diagnostics, "loss": diagnostics["Loss"], "actor_loss": diagnostics["Policy loss"], "critic_loss": diagnostics["Value loss"], "entropy": diagnostics["Base entropy"]}


# -----------------------------------------------------------------------------
# 9. Explainability and Graph-Ablation Utilities
# -----------------------------------------------------------------------------

def evaluate_edge_ablation(
    agent: XGATDRLAgent,
    sequence: torch.Tensor,
    risk_adjacency: torch.Tensor,
    predictive_adjacency: torch.Tensor,
    external_state: torch.Tensor,
    *,
    edge: tuple[int, int],
    graph: Literal["risk", "predictive"] = "risk",
    predictive_lag: int | None = None,
    base_weights: torch.Tensor | None = None,
    cash_bounds: torch.Tensor | None = None,
) -> float:
    """Measure the L1 policy shift after removing one graph edge.

    Risk edges are removed symmetrically.  Predictive edges are directed and
    are removed from every lag unless ``predictive_lag`` selects one channel.
    """
    if graph not in {"risk", "predictive"}:
        raise ValueError("graph must be 'risk' or 'predictive'.")
    if risk_adjacency.ndim == 2:
        risk_adjacency = risk_adjacency.unsqueeze(0)
    if predictive_adjacency.ndim == 3:
        predictive_adjacency = predictive_adjacency.unsqueeze(1)
    target, source = edge
    was_training = agent.training
    try:
        agent.eval()
        with torch.no_grad():
            baseline = agent.deterministic_action(
                sequence,
                risk_adjacency,
                predictive_adjacency,
                external_state,
                base_weights,
                cash_bounds,
            )
            ablated_risk = risk_adjacency.clone()
            ablated_predictive = predictive_adjacency.clone()
            if graph == "risk":
                ablated_risk[:, target, source] = 0.0
                ablated_risk[:, source, target] = 0.0
            elif predictive_lag is None:
                ablated_predictive[:, :, target, source] = 0.0
            else:
                if not 0 <= predictive_lag < ablated_predictive.size(1):
                    raise ValueError("predictive_lag is outside the available lag channels.")
                ablated_predictive[:, predictive_lag, target, source] = 0.0
            changed = agent.deterministic_action(
                sequence,
                ablated_risk,
                ablated_predictive,
                external_state,
                base_weights,
                cash_bounds,
            )
    finally:
        agent.train(was_training)
    return float(torch.abs(baseline - changed).sum(dim=-1).mean().cpu())


# -----------------------------------------------------------------------------
# 10. Visualisation
# -----------------------------------------------------------------------------

def _save_or_close(figure: plt.Figure, save_path: PathLike | None) -> None:
    if not bool(getattr(figure, "_xgat_layout_done", False)):
        figure.tight_layout()
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=200)
    plt.close(figure)

def plot_cumulative_wealth(
    returns_by_method: Mapping[str, ArrayLike],
    *,
    title: str = "Out-of-sample cumulative wealth",
    save_path: PathLike | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    for method, returns in returns_by_method.items():
        values = _as_finite_array(returns, name=str(method), ndim=1)
        style = METHOD_VISUAL_STYLES.get(str(method), {}) if "METHOD_VISUAL_STYLES" in globals() else {}
        axis.plot(
            np.exp(np.cumsum(values)),
            label=method,
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 2.0),
        )
    axis.set_title(title)
    axis.set_xlabel("Time step")
    axis.set_ylabel("Portfolio wealth")
    axis.legend(frameon=False, ncol=2)
    _save_or_close(figure, save_path)

def plot_drawdowns(
    returns_by_method: Mapping[str, ArrayLike],
    *,
    title: str = "Out-of-sample drawdowns",
    save_path: PathLike | None = None,
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5))
    for method, returns in returns_by_method.items():
        values = _as_finite_array(returns, name=str(method), ndim=1)
        wealth = np.concatenate([[1.0], np.exp(np.cumsum(values))])
        peaks = np.maximum.accumulate(wealth)
        style = METHOD_VISUAL_STYLES.get(str(method), {}) if "METHOD_VISUAL_STYLES" in globals() else {}
        axis.plot(
            wealth / peaks - 1.0,
            label=method,
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 1.8),
        )
    axis.axhline(0.0, linewidth=0.8, color="0.4")
    axis.set_title(title)
    axis.set_xlabel("Time step")
    axis.set_ylabel("Drawdown")
    axis.legend(frameon=False, ncol=2)
    _save_or_close(figure, save_path)

def plot_allocation_area(weights: np.ndarray, asset_names: Sequence[str], *, save_path: PathLike | None = None) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    axis.stackplot(np.arange(weights.shape[0]), weights.T, labels=list(asset_names), alpha=0.85)
    axis.set_title("Dynamic portfolio allocation")
    axis.set_xlabel("Time step")
    axis.set_ylabel("Portfolio weight")
    axis.set_ylim(0.0, 1.0)
    axis.margins(x=0)
    axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    _save_or_close(figure, save_path)

def plot_network_topology(
    adjacency: np.ndarray,
    asset_names: Sequence[str],
    *,
    title: str = "Graph topology",
    save_path: PathLike | None = None,
) -> dict[int, np.ndarray]:
    """Plot a stable and readable signed network topology.

    Node positions are deterministic and shared across scenarios when the
    number of assets is unchanged. Solid edges denote positive coefficients;
    dashed edges denote negative coefficients. Edge width is proportional to
    absolute coefficient magnitude.
    """
    matrix = _as_finite_array(adjacency, name="adjacency", ndim=2).copy()
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] != len(asset_names):
        raise ValueError("adjacency and asset_names are inconsistent.")
    np.fill_diagonal(matrix, 0.0)
    directed = not np.allclose(matrix, matrix.T, atol=1e-8)
    graph_type = nx.DiGraph if directed else nx.Graph
    graph = nx.from_numpy_array(matrix, create_using=graph_type)

    # A circular layout makes node positions directly comparable across
    # scenarios and avoids labels being clipped inside small nodes.
    layout = nx.circular_layout(graph)
    figure, axis = plt.subplots(figsize=(8.5, 7.2))
    node_size = max(1_500, 250 * max(len(str(name)) for name in asset_names))
    nx.draw_networkx_nodes(
        graph,
        layout,
        node_size=node_size,
        ax=axis,
        node_color="#4C72B0",
        edgecolors="black",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(
        graph,
        layout,
        labels={index: str(name) for index, name in enumerate(asset_names)},
        ax=axis,
        font_color="white",
        font_weight="bold",
        font_size=9,
    )

    positive = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data.get("weight", 0.0) >= 0.0
    ]
    negative = [
        (u, v)
        for u, v, data in graph.edges(data=True)
        if data.get("weight", 0.0) < 0.0
    ]
    for edges, colour, line_style in (
        (positive, "0.20", "solid"),
        (negative, "0.55", "dashed"),
    ):
        widths = [
            0.9 + 5.0 * abs(float(graph.edges[edge].get("weight", 0.0)))
            for edge in edges
        ]
        edge_kwargs: dict[str, Any] = {
            "G": graph,
            "pos": layout,
            "edgelist": edges,
            "width": widths,
            "alpha": 0.80,
            "ax": axis,
            "edge_color": colour,
            "style": line_style,
            "arrows": directed,
            "connectionstyle": "arc3,rad=0.08" if directed else "arc3",
        }
        if directed:
            edge_kwargs["arrowsize"] = 18
            edge_kwargs["min_source_margin"] = 18
            edge_kwargs["min_target_margin"] = 18
        nx.draw_networkx_edges(**edge_kwargs)

    axis.set_title(title)
    axis.text(
        0.5,
        -0.04,
        "Solid: positive relation; dashed: negative relation; width: absolute magnitude",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=8.5,
    )
    axis.margins(0.22)
    axis.axis("off")
    _save_or_close(figure, save_path)
    return layout


def plot_pareto_frontier(
    returns: ArrayLike,
    cvars: ArrayLike,
    turnovers: ArrayLike,
    *,
    labels: Sequence[str],
    save_path: PathLike | None = None,
) -> None:
    """Plot the return-CVaR trade-off and connect non-dominated methods."""
    return_array = _as_finite_array(returns, name="returns", ndim=1)
    cvar_array = _as_finite_array(cvars, name="cvars", ndim=1)
    turnover_array = _as_finite_array(turnovers, name="turnovers", ndim=1)
    if not (
        return_array.size
        == cvar_array.size
        == turnover_array.size
        == len(labels)
    ):
        raise ValueError("Pareto inputs and labels must have equal length.")

    efficient = np.ones(return_array.size, dtype=bool)
    for index in range(return_array.size):
        dominates = (
            (return_array >= return_array[index])
            & (cvar_array >= cvar_array[index])
            & (
                (return_array > return_array[index])
                | (cvar_array > cvar_array[index])
            )
        )
        efficient[index] = not bool(np.any(dominates))

    figure, axis = plt.subplots(figsize=(9, 6))
    maximum_turnover = max(float(np.max(turnover_array)), 1e-8)
    for index, label in enumerate(labels):
        style = METHOD_VISUAL_STYLES.get(str(label), {}) if "METHOD_VISUAL_STYLES" in globals() else {}
        marker_size = 70.0 + 180.0 * float(turnover_array[index]) / maximum_turnover
        axis.scatter(
            cvar_array[index],
            return_array[index],
            s=marker_size,
            color=style.get("color"),
            marker=style.get("marker", "o"),
            edgecolors="black",
            linewidths=0.6,
            alpha=0.90,
            zorder=3,
        )
        axis.annotate(
            str(label),
            (cvar_array[index], return_array[index]),
            xytext=(6, 6),
            textcoords="offset points",
        )

    frontier_indices = np.flatnonzero(efficient)
    if frontier_indices.size >= 2:
        order = frontier_indices[np.argsort(cvar_array[frontier_indices])]
        axis.plot(
            cvar_array[order],
            return_array[order],
            color="0.25",
            linewidth=1.2,
            linestyle="--",
            label="Non-dominated frontier",
            zorder=2,
        )
        axis.legend(frameon=False)
    axis.set_xlabel("Daily CVaR 95% (higher / closer to zero is better)")
    axis.set_ylabel("Annualised return")
    axis.set_title("Return-CVaR trade-off; marker size represents turnover")
    axis.grid(True, linestyle=":", alpha=0.35)
    _save_or_close(figure, save_path)


def plot_return_cvar_facets(
    summary: pd.DataFrame,
    *,
    scenario_order: Sequence[str],
    method_order: Sequence[str],
    save_path: PathLike | None = None,
) -> plt.Figure:
    """Facet the IQM return--CVaR frontier by scenario.

    The input must contain one row per scenario and method with the same
    estimators used in the main performance table.
    """
    required = {
        "Scenario",
        "Method",
        "Annualised return",
        "Daily CVaR 95%",
        "Mean turnover",
    }
    if not required.issubset(summary.columns):
        raise ValueError(f"summary must contain {sorted(required)}")

    active = [
        scenario
        for scenario in scenario_order
        if bool((summary["Scenario"].astype(str) == str(scenario)).any())
    ]
    if not active:
        raise ValueError("No requested scenarios are available.")

    figure, axes = plt.subplots(
        1,
        len(active),
        figsize=(6.2 * len(active), 5.3),
        squeeze=False,
    )
    for axis, scenario in zip(axes.ravel(), active):
        frame = summary.loc[
            summary["Scenario"].astype(str) == str(scenario)
        ].copy()
        frame["Method"] = pd.Categorical(
            frame["Method"].astype(str),
            categories=list(method_order),
            ordered=True,
        )
        frame = frame.sort_values("Method").dropna(
            subset=["Annualised return", "Daily CVaR 95%", "Mean turnover"]
        )
        if frame.empty:
            axis.set_visible(False)
            continue

        returns = frame["Annualised return"].to_numpy(dtype=float)
        cvars = frame["Daily CVaR 95%"].to_numpy(dtype=float)
        turnovers = frame["Mean turnover"].to_numpy(dtype=float)
        labels = frame["Method"].astype(str).tolist()
        maximum_turnover = max(float(np.max(turnovers)), 1e-8)

        efficient = np.ones(frame.shape[0], dtype=bool)
        for index in range(frame.shape[0]):
            dominates = (
                (returns >= returns[index])
                & (cvars >= cvars[index])
                & ((returns > returns[index]) | (cvars > cvars[index]))
            )
            efficient[index] = not bool(np.any(dominates))

        for index, label in enumerate(labels):
            style = METHOD_VISUAL_STYLES.get(label, {})
            marker_size = 70.0 + 190.0 * math.sqrt(
                max(float(turnovers[index]), 0.0) / maximum_turnover
            )
            axis.scatter(
                cvars[index],
                returns[index],
                s=marker_size,
                color=style.get("color"),
                marker=style.get("marker", "o"),
                edgecolors="black",
                linewidths=0.6,
                alpha=0.92,
                zorder=3,
            )
            axis.annotate(
                label,
                (cvars[index], returns[index]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8.5,
            )

        frontier = np.flatnonzero(efficient)
        if frontier.size >= 2:
            order = frontier[np.argsort(cvars[frontier])]
            axis.plot(
                cvars[order],
                returns[order],
                color="0.25",
                linewidth=1.2,
                linestyle="--",
                zorder=2,
            )

        axis.axhline(0.0, color="0.5", linewidth=0.8, linestyle=":")
        axis.set_title(str(scenario).replace("_", " ").title())
        axis.set_xlabel("Daily CVaR 95% (closer to zero is better)")
        axis.set_ylabel("Annualised return")
        axis.grid(True, linestyle=":", alpha=0.30)

    figure.suptitle(
        "Return--CVaR trade-off; marker area represents mean turnover"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure._xgat_layout_done = True
    _save_or_close(figure, save_path)
    return figure


def plot_graph_recovery_diagnostics(
    summary: pd.DataFrame,
    *,
    negative_control: str = "covariance_only",
    positive_scenarios: Sequence[str] = ("graph_predictive", "tail_stress"),
    save_path: PathLike | None = None,
) -> plt.Figure:
    """Plot false discoveries separately from positive-edge recovery.

    Overall accuracy is intentionally not used as the headline metric because
    it is inflated when the graph contains many true non-edges.
    """
    required = {
        "Scenario",
        "Metric",
        "Estimate",
        "Confidence low",
        "Confidence high",
    }
    if not required.issubset(summary.columns):
        raise ValueError(f"summary must contain {sorted(required)}")

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.9), squeeze=False)
    left, right = axes.ravel()

    control = summary.loc[
        (summary["Scenario"].astype(str) == str(negative_control))
        & (summary["Metric"] == "Predictive graph false-positive rate")
    ]
    if not control.empty:
        row = control.iloc[0]
        estimate = float(row["Estimate"])
        low = float(row["Confidence low"])
        high = float(row["Confidence high"])
        left.errorbar(
            estimate,
            0,
            xerr=np.array([[max(0.0, estimate - low)], [max(0.0, high - estimate)]]),
            fmt="o",
            capsize=5,
            linewidth=1.5,
        )
        left.set_yticks([0], labels=[str(negative_control).replace("_", " ").title()])
    else:
        left.text(0.5, 0.5, "No negative-control summary", ha="center", va="center")
        left.set_yticks([])
    left.set_xlim(0.0, 1.0)
    left.set_xlabel("False-positive rate")
    left.set_title("Negative-control false discoveries")
    left.grid(axis="x", alpha=0.25)

    metric_labels = [
        ("Predictive graph precision", "Precision"),
        ("Predictive graph recall", "Recall"),
        ("Predictive graph F1", "F1"),
    ]
    rows: list[tuple[str, str, float, float, float]] = []
    for scenario in positive_scenarios:
        for metric, label in metric_labels:
            frame = summary.loc[
                (summary["Scenario"].astype(str) == str(scenario))
                & (summary["Metric"] == metric)
            ]
            if frame.empty:
                continue
            row = frame.iloc[0]
            rows.append(
                (
                    str(scenario).replace("_", " ").title(),
                    label,
                    float(row["Estimate"]),
                    float(row["Confidence low"]),
                    float(row["Confidence high"]),
                )
            )

    positions = np.arange(len(rows))[::-1]
    for position, (scenario, label, estimate, low, high) in zip(positions, rows):
        right.errorbar(
            estimate,
            position,
            xerr=np.array([[max(0.0, estimate - low)], [max(0.0, high - estimate)]]),
            fmt="o",
            capsize=4,
            linewidth=1.4,
        )
    right.set_yticks(
        positions,
        labels=[f"{scenario}: {label}" for scenario, label, *_ in rows],
    )
    right.set_xlim(0.0, 1.0)
    right.set_xlabel("Recovery metric")
    right.set_title("Recovery when directed spillovers are present")
    right.grid(axis="x", alpha=0.25)

    figure.suptitle("Directed predictive-graph recovery")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure._xgat_layout_done = True
    _save_or_close(figure, save_path)
    return figure


def plot_metric_distributions(
    experiment_metrics: pd.DataFrame,
    *,
    metric: str,
    method_column: str = "Method",
    save_path: PathLike | None = None,
) -> None:
    """Show replication distributions using fixed colour- and marker-safe styles."""
    if metric not in experiment_metrics or method_column not in experiment_metrics:
        raise ValueError("experiment_metrics lacks the requested columns.")
    methods = list(dict.fromkeys(experiment_metrics[method_column].astype(str)))
    data = [
        experiment_metrics.loc[
            experiment_metrics[method_column].astype(str) == method, metric
        ]
        .dropna()
        .to_numpy(dtype=float)
        for method in methods
    ]
    figure, axis = plt.subplots(figsize=(max(10, 1.15 * len(methods)), 6))
    boxes = axis.boxplot(
        data,
        tick_labels=methods,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.3},
    )
    rng = np.random.default_rng(42)
    hatches = ("", "//", "..", "xx", "\\", "++", "oo", "**")
    for index, (method, values, box) in enumerate(
        zip(methods, data, boxes["boxes"]), start=1
    ):
        style = METHOD_VISUAL_STYLES.get(method, {}) if "METHOD_VISUAL_STYLES" in globals() else {}
        colour = style.get("color", "0.7")
        box.set_facecolor(colour)
        box.set_alpha(0.22)
        box.set_edgecolor(colour)
        box.set_hatch(hatches[(index - 1) % len(hatches)])
        if values.size:
            jitter = rng.normal(index, 0.035, size=values.size)
            axis.scatter(
                jitter,
                values,
                s=24,
                alpha=0.65,
                color=colour,
                marker=style.get("marker", "o"),
                edgecolors="none",
            )
    axis.set_title(f"Distribution of {metric} across independent markets and policy runs")
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", linestyle=":", alpha=0.30)
    _save_or_close(figure, save_path)

def plot_metric_forest(
    comparison_table: pd.DataFrame,
    *,
    metric_column: str = "Metric",
    estimate_column: str = "Estimate",
    low_column: str = "Confidence interval low",
    high_column: str = "Confidence interval high",
    title: str = "Paired performance differences (95% Bootstrap CI)",
    save_path: PathLike | None = None,
) -> None:
    """Plot bootstrap metric differences with confidence intervals."""
    table = comparison_table.reset_index(drop=True)
    positions = np.arange(table.shape[0])
    estimates = table[estimate_column].to_numpy(dtype=float)
    lower = estimates - table[low_column].to_numpy(dtype=float)
    upper = table[high_column].to_numpy(dtype=float) - estimates
    
    figure, axis = plt.subplots(figsize=(9, max(4, 0.6 * table.shape[0])))
    axis.errorbar(estimates, positions, xerr=np.vstack([lower, upper]), fmt="o", capsize=3, color="navy")
    axis.axvline(0.0, linestyle="--", linewidth=1, color="red")
    axis.set_yticks(positions, labels=table[metric_column].astype(str))
    axis.set_xlabel("Difference; Positive favours X-GAT-DRL")
    axis.set_title(title)
    axis.invert_yaxis()
    _save_or_close(figure, save_path)


# -----------------------------------------------------------------------------
# 11. Constraint, graph-selection, and policy extensions
# -----------------------------------------------------------------------------




@dataclass(frozen=True)
class RiskBudgetConfig:
    cvar_limit: float = 0.025
    drawdown_limit: float = 0.25
    turnover_limit: float = 0.03
    hhi_limit: float = 0.45
    multiplier_learning_rate: float = 0.025
    maximum_multiplier: float = 50.0
    active_constraints: tuple[str, ...] = ("cvar", "drawdown")

    def __post_init__(self) -> None:
        valid = {"cvar", "drawdown", "turnover", "hhi"}
        if min(self.cvar_limit, self.drawdown_limit, self.turnover_limit, self.hhi_limit) < 0.0:
            raise ValueError("Risk limits must be non-negative.")
        if self.multiplier_learning_rate <= 0.0:
            raise ValueError("multiplier_learning_rate must be positive.")
        if self.maximum_multiplier <= 0.0:
            raise ValueError("maximum_multiplier must be positive.")
        if not self.active_constraints:
            raise ValueError("At least one risk constraint must be active.")
        unknown = set(self.active_constraints) - valid
        if unknown:
            raise ValueError(f"Unknown active constraints: {sorted(unknown)}")


class LagrangeController:
    """Projected dual updates for predeclared empirical risk constraints."""

    names = ("cvar", "drawdown", "turnover", "hhi")

    def __init__(self, config: RiskBudgetConfig = RiskBudgetConfig()) -> None:
        self.config = config
        self.active_names = tuple(config.active_constraints)
        self.values = {name: 0.0 for name in self.names}

    def state_dict(self) -> dict[str, float]:
        return dict(self.values)

    def load_state_dict(self, state: Mapping[str, float]) -> None:
        for name in self.names:
            value = state.get(name, 0.0) if name in self.active_names else 0.0
            self.values[name] = float(np.clip(value, 0.0, self.config.maximum_multiplier))

    def update(self, realised: Mapping[str, float], limits: Mapping[str, float] | None = None) -> dict[str, float]:
        active_limits = {
            "cvar": self.config.cvar_limit,
            "drawdown": self.config.drawdown_limit,
            "turnover": self.config.turnover_limit,
            "hhi": self.config.hhi_limit,
        }
        if limits is not None:
            active_limits.update({key: float(value) for key, value in limits.items() if key in active_limits})

        for name in self.names:
            if name not in self.active_names:
                self.values[name] = 0.0
                continue
            limit = max(float(active_limits[name]), 1e-12)
            normalised_violation = float(realised.get(name, 0.0)) / limit - 1.0
            updated = self.values[name] + self.config.multiplier_learning_rate * normalised_violation
            self.values[name] = float(np.clip(updated, 0.0, self.config.maximum_multiplier))
        return self.state_dict()

    def penalty(self, costs: Mapping[str, float]) -> float:
        return float(
            sum(
                self.values[name] * max(0.0, float(costs.get(name, 0.0)))
                for name in self.active_names
            )
        )


def regime_dependent_limits(
    crisis_probability: float,
    base: RiskBudgetConfig = RiskBudgetConfig(),
) -> dict[str, float]:
    """Interpolate predeclared normal/crisis limits without discontinuities."""
    p = float(np.clip(crisis_probability, 0.0, 1.0))
    return {
        "cvar": (1.0 - p) * base.cvar_limit + p * 0.75 * base.cvar_limit,
        "drawdown": (1.0 - p) * base.drawdown_limit + p * 0.80 * base.drawdown_limit,
        "turnover": (1.0 - p) * base.turnover_limit + p * 0.80 * base.turnover_limit,
        "hhi": (1.0 - p) * base.hhi_limit + p * 0.90 * base.hhi_limit,
    }


def cash_bounds_from_signals(
    crisis_probability: float,
    drawdown: float,
    predicted_cvar: float,
    uncertainty: float,
    *,
    normal_minimum: float = 0.0,
    crisis_minimum: float = 0.35,
    normal_maximum: float = 0.35,
    maximum_cash: float = 1.0,
) -> tuple[float, float]:
    """Produce smooth feasible cash bounds from observable state variables."""
    p = float(np.clip(crisis_probability, 0.0, 1.0))
    dd = float(np.clip(abs(min(drawdown, 0.0)), 0.0, 1.0))
    tail = float(np.clip(abs(min(predicted_cvar, 0.0)) / 0.05, 0.0, 1.0))
    unc = float(np.clip(uncertainty, 0.0, 1.0))

    lower_signal = 0.50 * p + 0.20 * dd + 0.20 * tail + 0.10 * unc
    upper_signal = 0.65 * p + 0.15 * dd + 0.10 * tail + 0.10 * unc
    lower = normal_minimum + lower_signal * (crisis_minimum - normal_minimum)
    upper = normal_maximum + upper_signal * (maximum_cash - normal_maximum)
    lower = float(np.clip(lower, 0.0, maximum_cash))
    upper = float(np.clip(max(upper, lower + 1e-4), lower, maximum_cash))
    return lower, upper


@dataclass(frozen=True)
class StableGraphRepresentation:
    signed_partial_correlation: np.ndarray
    adjacency: np.ndarray
    edge_mask: np.ndarray
    stability: np.ndarray


def _moving_block_indices(
    length: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if length < 2:
        raise ValueError("At least two observations are required.")
    block = int(np.clip(block_length, 1, length))
    starts = rng.integers(0, length, size=int(math.ceil(length / block)))
    pieces = [np.mod(np.arange(start, start + block), length) for start in starts]
    return np.concatenate(pieces)[:length]


def stability_selected_glasso_graph(
    returns: np.ndarray,
    state_probabilities: ArrayLike | None = None,
    *,
    minimum_effective_samples: float = 30.0,
    ebic_gamma: float = 0.50,
    threshold: float = 0.05,
    bootstrap_replicates: int = 24,
    block_length: int = 20,
    selection_probability: float = 0.65,
    maximum_degree: int | None = None,
    maximum_density: float = 0.60,
    random_state: int = 0,
    alpha: float | None = None,
) -> tuple[GlassoResult, StableGraphRepresentation, float]:
    """Estimate a signed partial-correlation graph with block-bootstrap stability selection."""
    values = _as_finite_array(returns, name="returns", ndim=2, minimum_length=5)
    if not 0.0 < selection_probability <= 1.0:
        raise ValueError("selection_probability must lie in (0, 1].")
    if not 0.0 < maximum_density <= 1.0:
        raise ValueError("maximum_density must lie in (0, 1].")

    probabilities = None if state_probabilities is None else _as_finite_array(
        state_probabilities, name="state_probabilities", ndim=1, minimum_length=values.shape[0]
    )
    if probabilities is not None and probabilities.shape[0] != values.shape[0]:
        raise ValueError("state_probabilities and returns must have equal length.")

    fallback = estimate_covariance(values, method="ledoit_wolf")
    if probabilities is None:
        covariance = fallback
        effective = float(values.shape[0])
    else:
        covariance, effective = compute_regime_weighted_covariance(
            values,
            probabilities,
            minimum_effective_samples=minimum_effective_samples,
            fallback_covariance=fallback,
        )
    effective_n = max(2, int(round(effective)))
    glasso = (
        fit_ebic_glasso(
            covariance,
            n_samples=effective_n,
            ebic_gamma=ebic_gamma,
        )
        if alpha is None
        else fit_glasso_at_alpha(
            covariance,
            n_samples=effective_n,
            alpha=float(alpha),
            ebic_gamma=ebic_gamma,
        )
    )
    base = precision_to_graph(glasso.precision, threshold=threshold, include_self_loops=False)

    rng = np.random.default_rng(random_state)
    counts = np.zeros_like(base.edge_mask, dtype=float)
    successful = 0
    if int(bootstrap_replicates) <= 0:
        counts = base.edge_mask.astype(float)
        successful = 1
    for _ in range(max(0, int(bootstrap_replicates))):
        indices = _moving_block_indices(values.shape[0], block_length, rng)
        sampled_returns = values[indices]
        try:
            if probabilities is None:
                sampled_covariance = estimate_covariance(sampled_returns, method="ledoit_wolf")
                sampled_effective = float(sampled_returns.shape[0])
            else:
                sampled_covariance, sampled_effective = compute_regime_weighted_covariance(
                    sampled_returns,
                    probabilities[indices],
                    minimum_effective_samples=minimum_effective_samples,
                    fallback_covariance=fallback,
                )
            sampled_glasso = (
                fit_glasso_at_alpha(
                    sampled_covariance,
                    n_samples=max(2, int(round(sampled_effective))),
                    alpha=float(glasso.alpha),
                    ebic_gamma=ebic_gamma,
                )
                if np.isfinite(glasso.alpha) and glasso.alpha > 0.0
                else fit_ebic_glasso(
                    sampled_covariance,
                    n_samples=max(2, int(round(sampled_effective))),
                    ebic_gamma=ebic_gamma,
                )
            )
            sampled_graph = precision_to_graph(
                sampled_glasso.precision,
                threshold=threshold,
                include_self_loops=False,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            continue
        counts += sampled_graph.edge_mask.astype(float)
        successful += 1

    stability = counts / max(1, successful)
    signed = np.asarray(base.signed_partial_correlation, dtype=float)
    scores = np.abs(signed) * stability
    n_assets = signed.shape[0]
    candidates = [
        (float(scores[i, j]), i, j)
        for i in range(n_assets)
        for j in range(i + 1, n_assets)
        if stability[i, j] >= selection_probability and abs(signed[i, j]) >= threshold
    ]
    candidates.sort(reverse=True)
    maximum_edges = max(1, int(math.floor(maximum_density * n_assets * (n_assets - 1) / 2.0)))
    degree_limit = n_assets - 1 if maximum_degree is None else int(np.clip(maximum_degree, 1, n_assets - 1))
    degree = np.zeros(n_assets, dtype=int)
    mask = np.zeros((n_assets, n_assets), dtype=bool)
    accepted = 0
    for _, i, j in candidates:
        if accepted >= maximum_edges:
            break
        if degree[i] >= degree_limit or degree[j] >= degree_limit:
            continue
        mask[i, j] = mask[j, i] = True
        degree[i] += 1
        degree[j] += 1
        accepted += 1

    np.fill_diagonal(mask, True)
    weighted_signed = signed * stability
    adjacency = np.where(mask, np.abs(weighted_signed), 0.0)
    np.fill_diagonal(adjacency, 1.0)
    np.fill_diagonal(weighted_signed, 0.0)
    return glasso, StableGraphRepresentation(weighted_signed, adjacency, mask, stability), effective


def sparsify_signed_graph(
    signed_adjacency: np.ndarray,
    *,
    threshold: float = 0.05,
    maximum_degree: int | None = None,
    maximum_density: float = 0.60,
    directed: bool = False,
    include_self_loops: bool = True,
) -> np.ndarray:
    """Apply magnitude, density, and degree constraints without selecting zeros."""
    matrix = _as_finite_array(signed_adjacency, name="signed_adjacency", ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("signed_adjacency must be square.")
    if threshold < 0.0 or not 0.0 < maximum_density <= 1.0:
        raise ValueError("Invalid graph sparsification settings.")
    n_assets = matrix.shape[0]
    values = matrix.copy()
    np.fill_diagonal(values, 0.0)
    if not directed:
        values = 0.5 * (values + values.T)

    effective_threshold = max(float(threshold), 1e-12)
    candidates: list[tuple[float, int, int]] = []
    if directed:
        for target in range(n_assets):
            for source in range(n_assets):
                magnitude = abs(float(values[target, source]))
                if target != source and np.isfinite(magnitude) and magnitude > effective_threshold:
                    candidates.append((magnitude, target, source))
        possible_edges = n_assets * (n_assets - 1)
    else:
        for first in range(n_assets):
            for second in range(first + 1, n_assets):
                magnitude = abs(float(values[first, second]))
                if np.isfinite(magnitude) and magnitude > effective_threshold:
                    candidates.append((magnitude, first, second))
        possible_edges = n_assets * (n_assets - 1) // 2
    candidates.sort(reverse=True)

    maximum_edges = int(math.floor(maximum_density * possible_edges))
    degree_limit = n_assets - 1 if maximum_degree is None else int(
        np.clip(maximum_degree, 1, n_assets - 1)
    )
    selected = np.zeros_like(values)
    incoming = np.zeros(n_assets, dtype=int)
    undirected_degree = np.zeros(n_assets, dtype=int)
    accepted = 0
    for _, first, second in candidates:
        if accepted >= maximum_edges:
            break
        if directed:
            if incoming[first] >= degree_limit:
                continue
            selected[first, second] = values[first, second]
            incoming[first] += 1
        else:
            if undirected_degree[first] >= degree_limit or undirected_degree[second] >= degree_limit:
                continue
            selected[first, second] = selected[second, first] = values[first, second]
            undirected_degree[first] += 1
            undirected_degree[second] += 1
        accepted += 1
    if include_self_loops:
        np.fill_diagonal(selected, 1.0)
    return selected


def _lagged_design(returns: np.ndarray, maximum_lag: int) -> tuple[np.ndarray, np.ndarray]:
    values = _as_finite_array(returns, name="returns", ndim=2, minimum_length=maximum_lag + 8)
    if maximum_lag < 1:
        raise ValueError("maximum_lag must be positive.")
    design = np.concatenate(
        [values[maximum_lag - lag : values.shape[0] - lag] for lag in range(1, maximum_lag + 1)],
        axis=1,
    )
    targets = values[maximum_lag:]
    return design, targets


def _weighted_standardise(
    design: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalised = np.clip(weights, 0.0, None)
    if normalised.sum() <= 1e-12:
        normalised = np.ones_like(normalised)
    normalised /= normalised.sum()
    design_mean = np.sum(design * normalised[:, None], axis=0)
    target_mean = np.sum(targets * normalised[:, None], axis=0)
    design_scale = np.sqrt(np.sum((design - design_mean) ** 2 * normalised[:, None], axis=0))
    target_scale = np.sqrt(np.sum((targets - target_mean) ** 2 * normalised[:, None], axis=0))
    design_scale = np.where(design_scale > 1e-8, design_scale, 1.0)
    target_scale = np.where(target_scale > 1e-8, target_scale, 1.0)
    return (
        (design - design_mean) / design_scale,
        (targets - target_mean) / target_scale,
        design_scale,
        target_scale,
    )


def _fit_sparse_lag_coefficients(
    returns: np.ndarray,
    *,
    maximum_lag: int,
    sample_weights: np.ndarray | None,
    alphas: np.ndarray,
    l1_ratio: float,
    ebic_gamma: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    design, targets = _lagged_design(returns, maximum_lag)
    n_observations, n_predictors = design.shape
    if sample_weights is None:
        weights = np.ones(n_observations, dtype=float)
    else:
        raw_weights = _as_finite_array(sample_weights, name="sample_weights", ndim=1)
        if raw_weights.size != returns.shape[0]:
            raise ValueError("sample_weights and returns must have equal time length.")
        weights = np.asarray(raw_weights[maximum_lag:], dtype=float)
    weights = np.clip(weights, 0.0, None)
    if weights.sum() <= 1e-12:
        weights = np.ones_like(weights)
    normalised_weights = weights / weights.sum()
    effective_samples = float(1.0 / np.sum(normalised_weights**2))
    standard_design, standard_targets, design_scale, target_scale = _weighted_standardise(
        design, targets, normalised_weights
    )
    square_root_weights = np.sqrt(normalised_weights * n_observations)
    weighted_design = standard_design * square_root_weights[:, None]

    n_assets = targets.shape[1]
    coefficients = np.zeros((n_assets, maximum_lag, n_assets), dtype=float)
    selected_alphas = np.full(n_assets, np.nan, dtype=float)
    for target in range(n_assets):
        weighted_target = standard_targets[:, target] * square_root_weights
        best_score = float("inf")
        best_coefficients: np.ndarray | None = None
        best_alpha = float("nan")
        for alpha in alphas:
            model = ElasticNet(
                alpha=float(alpha),
                l1_ratio=float(l1_ratio),
                fit_intercept=False,
                max_iter=3_000,
                tol=1e-5,
                selection="cyclic",
            )
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=ConvergenceWarning)
                    model.fit(weighted_design, weighted_target)
            except ConvergenceWarning:
                continue
            residual = weighted_target - model.predict(weighted_design)
            residual_sum = max(float(np.dot(residual, residual)), 1e-12)
            nonzero = int(np.count_nonzero(np.abs(model.coef_) > 1e-8))
            bic = n_observations * math.log(residual_sum / n_observations)
            bic += nonzero * math.log(max(2, n_observations))
            bic += 2.0 * ebic_gamma * nonzero * math.log(max(2, n_predictors))
            if bic < best_score:
                best_score = bic
                best_coefficients = model.coef_.copy()
                best_alpha = float(alpha)
        if best_coefficients is None:
            continue
        raw = best_coefficients.reshape(maximum_lag, n_assets)
        raw = raw * (target_scale[target] / design_scale.reshape(maximum_lag, n_assets))
        coefficients[target] = raw
        selected_alphas[target] = best_alpha
    return coefficients, selected_alphas, effective_samples



def _block_permutation_indices(
    length: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute contiguous blocks while preserving within-block serial order."""
    block = max(1, min(int(block_length), int(length)))
    blocks = [np.arange(start, min(length, start + block), dtype=int) for start in range(0, length, block)]
    order = rng.permutation(len(blocks))
    return np.concatenate([blocks[index] for index in order])[:length]


def _benjamini_hochberg_mask(p_values: np.ndarray, level: float, valid: np.ndarray) -> np.ndarray:
    """Return the Benjamini-Hochberg rejection mask on a supplied valid set."""
    output = np.zeros_like(p_values, dtype=bool)
    flat_indices = np.flatnonzero(valid.ravel())
    if flat_indices.size == 0:
        return output
    values = np.asarray(p_values, dtype=float).ravel()[flat_indices]
    order = np.argsort(values)
    ordered = values[order]
    critical = float(level) * np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    accepted = np.flatnonzero(ordered <= critical)
    if accepted.size == 0:
        return output
    cutoff = ordered[accepted[-1]]
    output.ravel()[flat_indices[values <= cutoff]] = True
    return output


def _fixed_alpha_lag_coefficients(
    design: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
    *,
    maximum_lag: int,
    selected_alphas: np.ndarray,
    l1_ratio: float,
    permuted_design: np.ndarray | None = None,
) -> np.ndarray:
    """Fit lag coefficients using fixed target-specific penalties."""
    normalised_weights = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if normalised_weights.sum() <= 1e-12:
        normalised_weights = np.ones_like(normalised_weights)
    normalised_weights /= normalised_weights.sum()
    standard_design, standard_targets, design_scale, target_scale = _weighted_standardise(
        design, targets, normalised_weights
    )
    active_design = standard_design if permuted_design is None else np.asarray(permuted_design, dtype=float)
    square_root_weights = np.sqrt(normalised_weights * design.shape[0])
    weighted_design = active_design * square_root_weights[:, None]
    n_assets = targets.shape[1]
    coefficients = np.zeros((n_assets, maximum_lag, n_assets), dtype=float)
    finite_alphas = selected_alphas[np.isfinite(selected_alphas) & (selected_alphas > 0.0)]
    fallback_alpha = float(np.median(finite_alphas)) if finite_alphas.size else 0.01
    for target in range(n_assets):
        alpha = float(selected_alphas[target]) if np.isfinite(selected_alphas[target]) and selected_alphas[target] > 0.0 else fallback_alpha
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=float(l1_ratio),
            fit_intercept=False,
            max_iter=3_000,
            tol=1e-5,
            selection="cyclic",
        )
        weighted_target = standard_targets[:, target] * square_root_weights
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("error", category=ConvergenceWarning)
                model.fit(weighted_design, weighted_target)
        except ConvergenceWarning:
            continue
        raw = model.coef_.reshape(maximum_lag, n_assets)
        raw = raw * (target_scale[target] / design_scale.reshape(maximum_lag, n_assets))
        coefficients[target] = raw
    return coefficients


def stability_selected_predictive_graph(
    returns: np.ndarray,
    state_probabilities: ArrayLike | None = None,
    *,
    maximum_lag: int = 3,
    alphas: Iterable[float] | None = None,
    l1_ratio: float = 0.80,
    ebic_gamma: float = 0.50,
    coefficient_threshold: float = 0.01,
    bootstrap_replicates: int = 0,
    block_length: int = 20,
    selection_probability: float = 0.60,
    maximum_in_degree: int | None = 3,
    maximum_density: float = 0.30,
    random_state: int = 0,
    soft_selection: bool = False,
    minimum_stability_weight: float = 0.25,
    null_replicates: int = 0,
    false_discovery_rate: float = 0.10,
    null_quantile: float = 0.95,
) -> PredictiveGraphResult:
    """Estimate a lag-specific directed graph with stability and null calibration.
    """
    values = _as_finite_array(returns, name="returns", ndim=2, minimum_length=maximum_lag + 12)
    if not 0.0 <= false_discovery_rate <= 1.0:
        raise ValueError("false_discovery_rate must lie in [0, 1].")
    if not 0.5 < null_quantile < 1.0:
        raise ValueError("null_quantile must lie in (0.5, 1).")
    probabilities = None
    if state_probabilities is not None:
        probabilities = _as_finite_array(state_probabilities, name="state_probabilities", ndim=1)
        if probabilities.size != values.shape[0]:
            raise ValueError("state_probabilities and returns must have equal length.")
    alpha_grid = np.asarray(
        tuple(alphas) if alphas is not None else np.logspace(-3.0, -0.35, 10), dtype=float
    )
    coefficient_cube, selected_alphas, effective = _fit_sparse_lag_coefficients(
        values,
        maximum_lag=maximum_lag,
        sample_weights=probabilities,
        alphas=alpha_grid,
        l1_ratio=l1_ratio,
        ebic_gamma=ebic_gamma,
    )
    lagged_signed = np.transpose(coefficient_cube, (1, 0, 2))
    for lag in range(maximum_lag):
        np.fill_diagonal(lagged_signed[lag], 0.0)
    maximum_absolute = float(np.max(np.abs(lagged_signed))) if lagged_signed.size else 0.0
    if maximum_absolute > 1.0:
        lagged_signed = lagged_signed / maximum_absolute

    detection_threshold = float(max(1e-8, coefficient_threshold))
    counts = np.zeros_like(lagged_signed, dtype=float)
    successful = 0
    rng = np.random.default_rng(random_state)
    if int(bootstrap_replicates) <= 0:
        counts = (np.abs(lagged_signed) > detection_threshold).astype(float)
        successful = 1
    else:
        subsample_length = int(
            np.clip(round(0.70 * values.shape[0]), maximum_lag + 12, values.shape[0])
        )
        for _ in range(int(bootstrap_replicates)):
            maximum_start = values.shape[0] - subsample_length
            start = int(rng.integers(0, maximum_start + 1)) if maximum_start > 0 else 0
            indices = np.arange(start, start + subsample_length, dtype=int)
            sampled_values = values[indices]
            sampled_probabilities = None if probabilities is None else probabilities[indices]
            try:
                sampled_cube, _, _ = _fit_sparse_lag_coefficients(
                    sampled_values,
                    maximum_lag=maximum_lag,
                    sample_weights=sampled_probabilities,
                    alphas=alpha_grid,
                    l1_ratio=l1_ratio,
                    ebic_gamma=ebic_gamma,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
            sampled_lagged = np.transpose(sampled_cube, (1, 0, 2))
            for lag in range(maximum_lag):
                np.fill_diagonal(sampled_lagged[lag], 0.0)
            counts += (np.abs(sampled_lagged) > detection_threshold).astype(float)
            successful += 1
    lagged_stability = counts / max(1, successful)
    lagged_weighted = lagged_signed * lagged_stability

    n_assets = values.shape[1]
    off_diagonal = np.ones_like(lagged_signed, dtype=bool)
    for lag in range(maximum_lag):
        np.fill_diagonal(off_diagonal[lag], False)
    lagged_p_values = np.ones_like(lagged_signed, dtype=float)
    lagged_null_thresholds = np.full_like(lagged_signed, detection_threshold, dtype=float)
    fdr_mask = off_diagonal.copy()

    if int(null_replicates) > 0:
        design, targets = _lagged_design(values, maximum_lag)
        if probabilities is None:
            target_weights = np.ones(targets.shape[0], dtype=float)
        else:
            target_weights = np.asarray(probabilities[maximum_lag:], dtype=float)
        normalised = np.clip(target_weights, 0.0, None)
        if normalised.sum() <= 1e-12:
            normalised = np.ones_like(normalised)
        normalised /= normalised.sum()
        standard_design, _, _, _ = _weighted_standardise(design, targets, normalised)
        null_coefficients: list[np.ndarray] = []
        source_columns = [
            np.asarray([lag * n_assets + source for lag in range(maximum_lag)], dtype=int)
            for source in range(n_assets)
        ]
        for _ in range(int(null_replicates)):
            permuted = standard_design.copy()
            for source, columns in enumerate(source_columns):
                indices = _block_permutation_indices(design.shape[0], block_length, rng)
                permuted[:, columns] = standard_design[indices][:, columns]
            cube = _fixed_alpha_lag_coefficients(
                design,
                targets,
                target_weights,
                maximum_lag=maximum_lag,
                selected_alphas=selected_alphas,
                l1_ratio=l1_ratio,
                permuted_design=permuted,
            )
            null_lagged = np.abs(np.transpose(cube, (1, 0, 2)))
            for lag in range(maximum_lag):
                np.fill_diagonal(null_lagged[lag], 0.0)
            null_coefficients.append(null_lagged)
        if null_coefficients:
            null_array = np.stack(null_coefficients, axis=0)
            observed_absolute = np.abs(lagged_signed)
            lagged_null_thresholds = np.quantile(null_array, null_quantile, axis=0)
            null_mean = np.mean(null_array, axis=0)
            null_scale = np.std(null_array, axis=0, ddof=1 if null_array.shape[0] > 1 else 0)
            z_score = (observed_absolute - null_mean) / np.maximum(null_scale, 1e-8)
            lagged_p_values = norm.sf(z_score)
            fdr_mask = np.zeros_like(off_diagonal, dtype=bool)
            for target in range(n_assets):
                target_valid = off_diagonal[:, target, :]
                target_mask = _benjamini_hochberg_mask(
                    lagged_p_values[:, target, :],
                    false_discovery_rate,
                    target_valid,
                )
                fdr_mask[:, target, :] = target_mask

    lagged_adjacencies: list[np.ndarray] = []
    for lag in range(maximum_lag):
        stability_limit = minimum_stability_weight if soft_selection else selection_probability
        magnitude_limit = np.maximum(detection_threshold, lagged_null_thresholds[lag])
        selected = (
            (lagged_stability[lag] >= stability_limit)
            & (np.abs(lagged_signed[lag]) > magnitude_limit)
            & fdr_mask[lag]
        )
        candidate_values = lagged_weighted[lag] if soft_selection else lagged_signed[lag]
        candidate = np.where(selected, candidate_values, 0.0)
        lagged_adjacencies.append(
            sparsify_signed_graph(
                candidate,
                threshold=1e-12,
                maximum_degree=maximum_in_degree,
                maximum_density=maximum_density,
                directed=True,
                include_self_loops=True,
            )
        )
    lagged_adjacency = np.stack(lagged_adjacencies, axis=0)

    lag_discount = 1.0 / np.arange(1, maximum_lag + 1, dtype=float)
    signed = np.einsum("ltk,l->tk", lagged_weighted, lag_discount)
    aggregate_candidate = np.einsum("ltk,l->tk", lagged_adjacency, lag_discount)
    np.fill_diagonal(signed, 0.0)
    adjacency = sparsify_signed_graph(
        aggregate_candidate,
        threshold=1e-12,
        maximum_degree=maximum_in_degree,
        maximum_density=maximum_density,
        directed=True,
        include_self_loops=True,
    )
    aggregate_stability = np.max(lagged_stability, axis=0)
    return PredictiveGraphResult(
        signed_coefficients=signed,
        adjacency=adjacency,
        edge_mask=np.abs(adjacency) > 1e-12,
        stability=aggregate_stability,
        selected_alphas=selected_alphas,
        effective_samples=effective,
        detection_threshold=detection_threshold,
        lagged_signed_coefficients=lagged_weighted,
        lagged_adjacency=lagged_adjacency,
        lagged_stability=lagged_stability,
        lagged_p_values=lagged_p_values,
        lagged_null_thresholds=lagged_null_thresholds,
    )


class _CausalTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.convolution = nn.Conv1d(
            channels, channels, kernel_size=3, dilation=dilation, padding=self.padding
        )
        self.normalisation = nn.GroupNorm(1, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        output = self.convolution(values)
        if self.padding:
            output = output[..., :-self.padding]
        output = self.dropout(F.silu(self.normalisation(output)))
        return output + residual


class AssetTemporalEncoder(nn.Module):
    """GRU, LSTM, or hybrid causal TCN plus decay-attention LSTM."""

    def __init__(
        self,
        num_features: int,
        hidden_dim: int,
        *,
        mode: Literal["gru", "lstm", "hybrid"] = "hybrid",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in {"gru", "lstm", "hybrid"}:
            raise ValueError("Unknown temporal encoder mode.")
        self.mode = mode
        self.hidden_dim = int(hidden_dim)
        if mode == "gru":
            self.recurrent = nn.GRU(num_features, hidden_dim, batch_first=True)
        else:
            self.recurrent = nn.LSTM(num_features, hidden_dim, batch_first=True)
        if mode == "hybrid":
            self.tcn_input = nn.Conv1d(num_features, hidden_dim, kernel_size=1)
            self.tcn_blocks = nn.ModuleList(
                [_CausalTemporalBlock(hidden_dim, dilation, dropout) for dilation in (1, 2, 4)]
            )
            self.query = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.decay_scale = nn.Linear(hidden_dim, 1)
            self.fusion_gate = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence.ndim != 4:
            raise ValueError("sequence must have shape [batch, assets, lookback, features].")
        batch, assets, lookback, features = sequence.shape
        flat = sequence.reshape(batch * assets, lookback, features)
        self.recurrent.flatten_parameters()
        recurrent_output, _ = self.recurrent(flat)
        if self.mode in {"gru", "lstm"}:
            representation = recurrent_output[:, -1]
            confidence = torch.ones((batch * assets, 1), device=flat.device, dtype=flat.dtype)
            return representation.reshape(batch, assets, -1), confidence.reshape(batch, assets, 1)

        query = self.query(recurrent_output[:, -1:])
        scores = torch.sum(self.key(recurrent_output) * query, dim=-1) / math.sqrt(self.hidden_dim)
        distance = torch.arange(lookback - 1, -1, -1, device=flat.device, dtype=flat.dtype)
        scale = F.softplus(self.decay_scale(recurrent_output)).squeeze(-1) + 1e-3
        scores = scores - distance.unsqueeze(0) / scale.square().clamp_min(1e-4)
        attention = F.softmax(scores, dim=-1)
        long_memory = torch.sum(attention.unsqueeze(-1) * recurrent_output, dim=1)

        tcn = self.tcn_input(flat.transpose(1, 2))
        for block in self.tcn_blocks:
            tcn = block(tcn)
        short_memory = tcn[..., -1]
        gate = torch.sigmoid(self.fusion_gate(torch.cat([short_memory, long_memory], dim=-1)))
        fused = self.output_norm(gate * short_memory + (1.0 - gate) * long_memory)
        # Low attention entropy means a more decisive temporal representation.
        entropy = -(attention * torch.log(attention.clamp_min(1e-8))).sum(dim=-1)
        confidence = 1.0 - entropy / math.log(max(2, lookback))
        return fused.reshape(batch, assets, -1), confidence.reshape(batch, assets, 1)


class DifferentiablePortfolioLayer(nn.Module):
    """Unrolled mirror-descent allocator with regime-conditional risk aversion."""

    def __init__(self, num_risk_assets: int, *, steps: int = 5) -> None:
        super().__init__()
        self.num_risk_assets = int(num_risk_assets)
        self.steps = int(steps)
        
        # Base parameters
        self.risk_aversion_base = nn.Parameter(torch.tensor(0.0))
        self.turnover_raw = nn.Parameter(torch.tensor(-1.5))
        self.step_raw = nn.Parameter(torch.tensor(-1.0))
        self.cash_head = nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
        
        # Dynamic regime mapping for risk aversion
        self.regime_risk_modifier = nn.Linear(2, 1)
        nn.init.zeros_(self.regime_risk_modifier.weight)
        nn.init.zeros_(self.regime_risk_modifier.bias)

    def forward(
        self,
        expected_returns: torch.Tensor,
        risk_scale: torch.Tensor,
        risk_adjacency: torch.Tensor,
        current_weights: torch.Tensor,
        cash_bounds: torch.Tensor,
        crisis_probability: torch.Tensor,
        regime_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        batch, assets = expected_returns.shape
        if assets != self.num_risk_assets:
            raise ValueError("Unexpected number of risky assets.")
        
        if risk_adjacency.ndim == 2:
            risk_adjacency = risk_adjacency.unsqueeze(0)
            
        graph = 0.5 * (risk_adjacency + risk_adjacency.transpose(-1, -2))
        eye = torch.eye(assets, device=graph.device, dtype=graph.dtype).expand(batch, -1, -1)
        off_graph = graph - torch.diag_embed(torch.diagonal(graph, dim1=-2, dim2=-1))
        factor = eye + 0.20 * off_graph
        covariance = factor @ factor.transpose(-1, -2)
        diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(1e-6).sqrt()
        correlation = covariance / (diagonal.unsqueeze(-1) * diagonal.unsqueeze(-2)).clamp_min(1e-6)
        volatility = F.softplus(risk_scale) + 0.05
        covariance = volatility.unsqueeze(-1) * correlation * volatility.unsqueeze(-2)

        low = cash_bounds[:, :1].clamp(0.0, 1.0)
        high = torch.maximum(cash_bounds[:, 1:2].clamp(0.0, 1.0), low + 1e-5)
        downside_signal = torch.relu(-expected_returns).mean(dim=-1, keepdim=True)
        dispersion = expected_returns.std(dim=-1, keepdim=True, unbiased=False)
        cash_unit = torch.sigmoid(
            self.cash_head(torch.cat([crisis_probability, regime_uncertainty, downside_signal, dispersion], dim=-1))
        )
        cash = low + (high - low) * cash_unit

        previous_risk = current_weights[:, :assets].clamp_min(1e-8)
        previous_share = previous_risk / previous_risk.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        weights = previous_share
        
        # Dynamic Risk Aversion: Adapts to HMM crisis signals
        regime_features = torch.cat([crisis_probability, regime_uncertainty], dim=-1)
        risk_aversion = F.softplus(self.risk_aversion_base + self.regime_risk_modifier(regime_features)) + 0.05
        
        turnover_penalty = F.softplus(self.turnover_raw)
        step = 0.02 + 0.25 * torch.sigmoid(self.step_raw)
        
        for _ in range(self.steps):
            marginal_risk = torch.bmm(covariance, weights.unsqueeze(-1)).squeeze(-1)
            turnover_gradient = torch.tanh((weights - previous_share) / 0.05)
            gradient = expected_returns - risk_aversion * marginal_risk - turnover_penalty * turnover_gradient
            logits = torch.log(weights.clamp_min(1e-8)) + step * gradient
            weights = F.softmax(logits, dim=-1)
            
        risky_weights = weights * (1.0 - cash)
        return torch.cat([risky_weights, cash], dim=-1)


class MultiGraphSpatioTemporalEncoder(nn.Module):
    """Temporal and predictive return experts with an independent risk expert."""

    def __init__(
        self,
        num_assets: int,
        num_features: int,
        *,
        hidden_gru: int = 32,
        hidden_gat: int = 32,
        dropout: float = 0.0,
        use_risk_graph: bool = True,
        use_predictive_graph: bool = True,
        temporal_mode: Literal["gru", "lstm", "hybrid"] = "hybrid",
        predictive_lags: int = 3,
        external_state_dim: int = 0,
    ) -> None:
        super().__init__()
        if min(num_assets, num_features, hidden_gru, hidden_gat, predictive_lags) <= 0:
            raise ValueError("Encoder dimensions must be positive.")
        self.num_assets = int(num_assets)
        self.hidden_gat = int(hidden_gat)
        self.predictive_lags = int(predictive_lags)
        self.use_risk_graph = bool(use_risk_graph)
        self.use_predictive_graph = bool(use_predictive_graph)
        self.temporal = AssetTemporalEncoder(
            num_features, hidden_gru, mode=temporal_mode, dropout=dropout
        )
        self.temporal_projection = nn.Linear(hidden_gru, hidden_gat)
        self.risk_gat = DenseGATLayer(hidden_gru, hidden_gat, dropout=dropout)
        self.predictive_gats = nn.ModuleList(
            [DenseGATLayer(hidden_gru, hidden_gat, dropout=dropout) for _ in range(self.predictive_lags)]
        )
        self.lag_embeddings = nn.Parameter(torch.zeros(self.predictive_lags, hidden_gru))
        nn.init.normal_(self.lag_embeddings, mean=0.0, std=0.02)
        self.lag_attention = nn.Sequential(
            nn.Linear(hidden_gat + 1, hidden_gat), nn.SiLU(), nn.Linear(hidden_gat, 1)
        )
        self.external_projection = (
            nn.Sequential(nn.Linear(external_state_dim, hidden_gat), nn.SiLU())
            if external_state_dim > 0
            else None
        )
        gate_input = 2 * hidden_gat + 5 + (hidden_gat if external_state_dim > 0 else 0)
        self.return_gate = nn.Sequential(
            nn.Linear(gate_input, hidden_gat),
            nn.SiLU(),
            nn.Linear(hidden_gat, 2),
        )
        self.gate_temperature = nn.Parameter(torch.tensor(0.0))
        self.plain_norm = nn.LayerNorm(hidden_gat)
        self.risk_norm = nn.LayerNorm(hidden_gat)
        self.predictive_norm = nn.LayerNorm(hidden_gat)

    @property
    def output_features(self) -> int:
        return 3 * self.hidden_gat

    @staticmethod
    def _node_confidence(adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim == 2:
            adjacency = adjacency.unsqueeze(0)
        assets = adjacency.size(-1)
        eye = torch.eye(assets, device=adjacency.device, dtype=torch.bool)
        magnitude = adjacency.abs().masked_fill(eye.unsqueeze(0), 0.0)
        nonzero = magnitude > 1e-8
        density = nonzero.float().mean(dim=-1, keepdim=True)
        strength = magnitude.sum(dim=-1, keepdim=True) / nonzero.sum(dim=-1, keepdim=True).clamp_min(1)
        concentration = magnitude.square().sum(dim=-1, keepdim=True) / magnitude.sum(dim=-1, keepdim=True).square().clamp_min(1e-8)
        return torch.cat([density, strength, concentration], dim=-1)

    def _prepare_predictive(self, adjacency: torch.Tensor) -> torch.Tensor:
        if adjacency.ndim == 3:
            adjacency = adjacency.unsqueeze(1)
        if adjacency.ndim != 4:
            raise ValueError("predictive_adjacency must have [batch, lag, asset, asset] shape.")
        if adjacency.size(1) < self.predictive_lags:
            padding = adjacency[:, -1:].expand(-1, self.predictive_lags - adjacency.size(1), -1, -1)
            adjacency = torch.cat([adjacency, padding], dim=1)
        return adjacency[:, : self.predictive_lags]

    def forward(
        self,
        sequence: torch.Tensor,
        risk_adjacency: torch.Tensor,
        predictive_adjacency: torch.Tensor,
        external_state: torch.Tensor | None = None,
        *,
        return_gates: bool = False,
        return_experts: bool = False,
    ) -> Any:
        temporal_hidden, temporal_confidence = self.temporal(sequence)
        plain = self.plain_norm(self.temporal_projection(temporal_hidden))
        risk = (
            self.risk_norm(self.risk_gat(temporal_hidden, risk_adjacency))
            if self.use_risk_graph
            else plain
        )

        lagged_adjacency = self._prepare_predictive(predictive_adjacency)
        if self.use_predictive_graph:
            lag_embeddings: list[torch.Tensor] = []
            lag_confidences: list[torch.Tensor] = []
            for lag_index, gat in enumerate(self.predictive_gats):
                lag_hidden = temporal_hidden + self.lag_embeddings[lag_index].view(1, 1, -1)
                graph = lagged_adjacency[:, lag_index]
                embedding = gat(lag_hidden, graph)
                confidence = self._node_confidence(graph)[..., 1:2]
                lag_embeddings.append(embedding)
                lag_confidences.append(confidence)
            stacked_embeddings = torch.stack(lag_embeddings, dim=2)
            stacked_confidence = torch.stack(lag_confidences, dim=2)
            lag_logits = self.lag_attention(
                torch.cat([stacked_embeddings, stacked_confidence], dim=-1)
            ).squeeze(-1)
            lag_weights = F.softmax(lag_logits, dim=2)
            predictive = self.predictive_norm(
                torch.sum(lag_weights.unsqueeze(-1) * stacked_embeddings, dim=2)
            )
        else:
            predictive = plain
            lag_weights = torch.zeros(
                sequence.size(0), self.num_assets, self.predictive_lags,
                dtype=sequence.dtype, device=sequence.device,
            )
            lag_weights[..., 0] = 1.0

        risk_confidence = self._node_confidence(risk_adjacency)
        predictive_aggregate = lagged_adjacency.abs().amax(dim=1)
        predictive_confidence = self._node_confidence(predictive_aggregate)
        temporal_predictive_distance = 1.0 - F.cosine_similarity(
            plain, predictive, dim=-1
        ).unsqueeze(-1)
        return_confidence_features = torch.cat(
            [temporal_confidence, predictive_confidence, temporal_predictive_distance], dim=-1
        )
        gate_inputs = [plain, predictive, return_confidence_features]
        if self.external_projection is not None:
            if external_state is None:
                raise ValueError("external_state is required by this encoder.")
            external = self.external_projection(external_state).unsqueeze(1).expand(-1, self.num_assets, -1)
            gate_inputs.append(external)
        if self.use_predictive_graph:
            return_logits = self.return_gate(torch.cat(gate_inputs, dim=-1))
            assets = predictive_aggregate.size(-1)
            identity = torch.eye(assets, dtype=torch.bool, device=predictive_aggregate.device)
            predictive_available = (
                predictive_aggregate.abs().masked_fill(identity.unsqueeze(0), 0.0).sum(dim=-1) > 1e-8
            )
            return_logits[..., 1] = return_logits[..., 1].masked_fill(
                ~predictive_available, -1e9
            )
            # Temperature scaling for sharp expert selection
            temp = F.softplus(self.gate_temperature) + 0.05
            return_gates = F.softmax(return_logits / temp, dim=-1)
        else:
            return_gates = torch.zeros(
                sequence.size(0), self.num_assets, 2,
                dtype=sequence.dtype, device=sequence.device,
            )
            return_gates[..., 0] = 1.0

        temporal_gate = return_gates[..., 0:1]
        predictive_gate = return_gates[..., 1:2]
        risk_usage = torch.ones_like(temporal_gate) if self.use_risk_graph else torch.zeros_like(temporal_gate)
        route_gates = torch.cat([temporal_gate, risk_usage, predictive_gate], dim=-1)
        encoded = torch.cat(
            [temporal_gate * plain, risk, predictive_gate * predictive], dim=-1
        )
        diagnostics = {
            "lag_weights": lag_weights,
            "temporal_confidence": temporal_confidence.squeeze(-1),
            "risk_confidence": risk_confidence[..., 1],
            "predictive_confidence": predictive_confidence[..., 1],
            "return_gate_entropy": -(
                return_gates * torch.log(return_gates.clamp_min(1e-8))
            ).sum(dim=-1),
        }
        if return_experts:
            return encoded, route_gates, (plain, risk, predictive), diagnostics
        return (encoded, route_gates) if return_gates else encoded


class AdaptiveBetaDirichletActor(nn.Module):
    """Residual logistic-normal policy centred on an economic optimiser base."""

    def __init__(
        self,
        state_dim: int,
        num_total_assets: int,
        *,
        maximum_cash: float = 1.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or num_total_assets < 2:
            raise ValueError("Actor dimensions are invalid.")
        self.num_risk = int(num_total_assets - 1)
        self.maximum_cash = float(maximum_cash)
        self.backbone = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
            nn.SiLU(),
        )
        latent_risk = max(1, self.num_risk - 1)
        self.risk_residual = nn.Linear(64, latent_risk)
        self.risk_log_std = nn.Linear(64, latent_risk)
        self.cash_residual = nn.Linear(64, 1)
        self.cash_log_std = nn.Linear(64, 1)
        self.residual_gate = nn.Linear(64, 1)
        for module in (self.risk_residual, self.cash_residual):
            nn.init.orthogonal_(module.weight, gain=0.02)
            nn.init.zeros_(module.bias)
        nn.init.constant_(self.risk_log_std.bias, -1.2)
        nn.init.constant_(self.cash_log_std.bias, -1.2)
        nn.init.zeros_(self.residual_gate.weight)
        nn.init.zeros_(self.residual_gate.bias)

    def _defaults(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch = state.shape[0]
        cash = torch.full((batch, 1), min(0.05, self.maximum_cash), device=state.device, dtype=state.dtype)
        risk = torch.full((batch, self.num_risk), 1.0 / self.num_risk, device=state.device, dtype=state.dtype)
        return torch.cat([risk * (1.0 - cash), cash], dim=-1), torch.tensor(
            [0.0, self.maximum_cash], device=state.device, dtype=state.dtype
        ).expand(batch, -1)

    @staticmethod
    def _alr(simplex: torch.Tensor) -> torch.Tensor:
        if simplex.size(-1) == 1:
            return torch.zeros((*simplex.shape[:-1], 1), dtype=simplex.dtype, device=simplex.device)
        reference = simplex[..., -1:].clamp_min(1e-8)
        return torch.log(simplex[..., :-1].clamp_min(1e-8)) - torch.log(reference)

    @staticmethod
    def _inverse_alr(latent: torch.Tensor, n_components: int) -> torch.Tensor:
        if n_components == 1:
            return torch.ones((*latent.shape[:-1], 1), dtype=latent.dtype, device=latent.device)
        logits = torch.cat([latent, torch.zeros_like(latent[..., :1])], dim=-1)
        return F.softmax(logits, dim=-1)

    def _components(
        self,
        state: torch.Tensor,
        base_weights: torch.Tensor | None,
        cash_bounds: torch.Tensor | None,
        anchor_enabled: bool,
    ) -> tuple[Independent, Independent, torch.Tensor, torch.Tensor, torch.Tensor]:
        default_base, default_bounds = self._defaults(state)
        base = default_base if base_weights is None else base_weights
        bounds = default_bounds if cash_bounds is None else cash_bounds
        hidden = self.backbone(state)
        low = torch.clamp(bounds[..., :1], 0.0, self.maximum_cash)
        high = torch.maximum(torch.clamp(bounds[..., 1:2], 0.0, self.maximum_cash), low + 1e-5)
        width = high - low
        base_cash = torch.clamp(base[..., -1:], low + 1e-6, high - 1e-6)
        base_risk = torch.clamp(base[..., :-1], min=1e-8)
        base_risk = base_risk / base_risk.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        residual_scale = 0.05 + 0.95 * torch.sigmoid(self.residual_gate(hidden))
        free_risk = 2.0 * torch.tanh(self.risk_residual(hidden))
        free_cash = 2.0 * torch.tanh(self.cash_residual(hidden))
        if anchor_enabled:
            base_risk_mean = self._alr(base_risk)
            base_unit = torch.clamp((base_cash - low) / width, 1e-5, 1.0 - 1e-5)
            base_cash_mean = torch.log(base_unit) - torch.log1p(-base_unit)
            risk_mean = base_risk_mean + residual_scale * free_risk
            cash_mean = base_cash_mean + residual_scale * free_cash
            base_reliance = 1.0 - residual_scale.squeeze(-1)
        else:
            risk_mean = free_risk
            cash_mean = free_cash
            base_reliance = torch.zeros(state.size(0), device=state.device, dtype=state.dtype)
        risk_std = torch.exp(torch.clamp(self.risk_log_std(hidden), -4.0, 0.50))
        cash_std = torch.exp(torch.clamp(self.cash_log_std(hidden), -4.0, 0.50))
        return (
            Independent(Normal(risk_mean, risk_std), 1),
            Independent(Normal(cash_mean, cash_std), 1),
            low,
            width,
            base_reliance,
        )

    def _transform(self, risk_latent: torch.Tensor, cash_latent: torch.Tensor, low: torch.Tensor, width: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        risk_share = self._inverse_alr(risk_latent, self.num_risk)
        cash_unit = torch.sigmoid(cash_latent)
        cash = low + width * cash_unit
        action = torch.cat([risk_share * (1.0 - cash), cash], dim=-1)
        log_jacobian = torch.log(width.squeeze(-1).clamp_min(1e-12))
        log_jacobian += torch.log(cash_unit.squeeze(-1).clamp_min(1e-12))
        log_jacobian += torch.log((1.0 - cash_unit.squeeze(-1)).clamp_min(1e-12))
        log_jacobian += torch.log(risk_share.clamp_min(1e-12)).sum(dim=-1)
        log_jacobian += (self.num_risk - 1) * torch.log((1.0 - cash.squeeze(-1)).clamp_min(1e-12))
        return action, log_jacobian

    def sample(self, state: torch.Tensor, base_weights: torch.Tensor | None = None, cash_bounds: torch.Tensor | None = None, *, anchor_enabled: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        risk_distribution, cash_distribution, low, width, base_reliance = self._components(state, base_weights, cash_bounds, anchor_enabled)
        risk_latent = risk_distribution.rsample()
        cash_latent = cash_distribution.rsample()
        action, log_jacobian = self._transform(risk_latent, cash_latent, low, width)
        log_probability = risk_distribution.log_prob(risk_latent) + cash_distribution.log_prob(cash_latent) - log_jacobian
        entropy = risk_distribution.entropy() + cash_distribution.entropy()
        return action, log_probability, entropy, base_reliance

    def evaluate(self, state: torch.Tensor, action: torch.Tensor, base_weights: torch.Tensor | None = None, cash_bounds: torch.Tensor | None = None, *, anchor_enabled: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        risk_distribution, cash_distribution, low, width, base_reliance = self._components(state, base_weights, cash_bounds, anchor_enabled)
        cash = torch.clamp(action[..., -1:], low + 1e-7, low + width - 1e-7)
        cash_unit = torch.clamp((cash - low) / width, 1e-6, 1.0 - 1e-6)
        cash_latent = torch.log(cash_unit) - torch.log1p(-cash_unit)
        risk_share = torch.clamp(action[..., :-1] / (1.0 - cash).clamp_min(1e-8), min=1e-8)
        risk_share = risk_share / risk_share.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        risk_latent = self._alr(risk_share)
        _, log_jacobian = self._transform(risk_latent, cash_latent, low, width)
        log_probability = risk_distribution.log_prob(risk_latent) + cash_distribution.log_prob(cash_latent) - log_jacobian
        return log_probability, risk_distribution.entropy() + cash_distribution.entropy(), base_reliance

    def deterministic(self, state: torch.Tensor, base_weights: torch.Tensor | None = None, cash_bounds: torch.Tensor | None = None, *, anchor_enabled: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        risk_distribution, cash_distribution, low, width, base_reliance = self._components(state, base_weights, cash_bounds, anchor_enabled)
        action, _ = self._transform(risk_distribution.base_dist.loc, cash_distribution.base_dist.loc, low, width)
        return action, base_reliance


class DistributionalCritic(nn.Module):
    """Reward, tail-quantile, and constraint-value heads."""

    def __init__(
        self,
        state_dim: int,
        *,
        hidden_dims: tuple[int, int] = (128, 64),
        quantile_levels: Sequence[float] = (0.01, 0.05, 0.10, 0.50, 0.90),
        cost_names: Sequence[str] = ("cvar", "drawdown", "turnover", "hhi"),
    ) -> None:
        super().__init__()
        first, second = hidden_dims
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, first),
            nn.LayerNorm(first),
            nn.SiLU(),
            nn.Linear(first, second),
            nn.LayerNorm(second),
            nn.SiLU(),
        )
        levels = tuple(float(value) for value in quantile_levels)
        self.value_head = nn.Linear(second, 1)
        self.quantile_head = nn.Linear(second, len(levels))
        self.cost_names = tuple(str(name) for name in cost_names)
        self.cost_heads = nn.ModuleDict({name: nn.Linear(second, 1) for name in self.cost_names})
        self.register_buffer("quantile_levels", torch.tensor(levels, dtype=torch.float32))
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.orthogonal_(self.quantile_head.weight, gain=0.5)
        nn.init.zeros_(self.value_head.bias)
        nn.init.zeros_(self.quantile_head.bias)
        for head in self.cost_heads.values():
            nn.init.orthogonal_(head.weight, gain=0.5)
            nn.init.zeros_(head.bias)

    def _hidden(self, state: torch.Tensor) -> torch.Tensor:
        return self.backbone(state)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self._hidden(state)
        raw = self.quantile_head(hidden)
        first = raw[..., :1]
        increments = F.softplus(raw[..., 1:])
        ordered_quantiles = torch.cat([first, first + torch.cumsum(increments, dim=-1)], dim=-1)
        return self.value_head(hidden).squeeze(-1), ordered_quantiles

    def cost_values(self, state: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self._hidden(state)
        return {name: self.cost_heads[name](hidden).squeeze(-1) for name in self.cost_names}


def quantile_huber_loss(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    quantile_levels: torch.Tensor,
    *,
    kappa: float = 1.0,
) -> torch.Tensor:
    target = targets.unsqueeze(-1)
    error = target - predicted
    absolute = error.abs()
    huber = torch.where(
        absolute <= kappa,
        0.5 * error.square(),
        kappa * (absolute - 0.5 * kappa),
    )
    weight = torch.abs(quantile_levels.view(1, -1) - (error.detach() < 0.0).float())
    return (weight * huber / kappa).mean()



class XGATDRLAgent(nn.Module):
    """Hybrid temporal/graph residual PPO policy with an economic optimiser base."""

    def __init__(
        self,
        num_risk_assets: int,
        num_features: int,
        external_state_dim: int,
        *,
        hidden_gru: int = 32,
        hidden_gat: int = 32,
        max_cash: float = 1.0,
        dropout: float = 0.0,
        use_risk_graph: bool = True,
        use_predictive_graph: bool = True,
        use_anchor: bool = False,
        temporal_mode: Literal["gru", "lstm", "hybrid"] = "hybrid",
        predictive_lags: int = 3,
        use_optimizer: bool = True,
    ) -> None:
        super().__init__()
        self.num_risk_assets = int(num_risk_assets)
        self.external_state_dim = int(external_state_dim)
        self.maximum_cash = float(max_cash)
        self.use_benchmark_anchor = bool(use_anchor)
        self.use_optimizer = bool(use_optimizer)
        self.use_risk_graph = bool(use_risk_graph)
        self.use_predictive_graph = bool(use_predictive_graph)
        # The actor is centred on the differentiable optimiser whenever it is active.
        self.use_anchor = bool(use_anchor or use_optimizer)
        self.encoder = MultiGraphSpatioTemporalEncoder(
            num_risk_assets,
            num_features,
            hidden_gru=hidden_gru,
            hidden_gat=hidden_gat,
            dropout=dropout,
            use_risk_graph=use_risk_graph,
            use_predictive_graph=use_predictive_graph,
            temporal_mode=temporal_mode,
            predictive_lags=predictive_lags,
            external_state_dim=external_state_dim,
        )
        self.temporal_return_head = nn.Linear(hidden_gat, 1)
        self.predictive_residual_head = nn.Linear(hidden_gat, 1)
        self.risk_scale_head = nn.Linear(hidden_gat, 1)
        for head in (self.temporal_return_head, self.predictive_residual_head, self.risk_scale_head):
            nn.init.orthogonal_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)
        encoded_dim = num_risk_assets * self.encoder.output_features
        expert_dim = 3 * num_risk_assets
        state_dim = encoded_dim + expert_dim + external_state_dim
        self.encoded_norm = nn.LayerNorm(encoded_dim)
        self.expert_norm = nn.LayerNorm(expert_dim)
        self.portfolio_layer = DifferentiablePortfolioLayer(num_risk_assets)
        self.actor = AdaptiveBetaDirichletActor(state_dim, num_risk_assets + 1, maximum_cash=max_cash)
        self.critic = DistributionalCritic(state_dim)

    def build_state_extended(self, sequence: torch.Tensor, risk_adjacency: torch.Tensor, predictive_adjacency: torch.Tensor, external_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        encoded, route_gates, experts, diagnostics = self.encoder(
            sequence, risk_adjacency, predictive_adjacency, external_state,
            return_experts=True,
        )
        plain, risk, predictive = experts
        temporal_prediction = self.temporal_return_head(plain).squeeze(-1)
        predictive_residual = self.predictive_residual_head(predictive).squeeze(-1)
        predictive_full_return = temporal_prediction + predictive_residual
        risk_prediction = F.softplus(self.risk_scale_head(risk).squeeze(-1))
        combined_return = (
            route_gates[..., 0] * temporal_prediction
            + route_gates[..., 2] * predictive_full_return
        )
        flattened = self.encoded_norm(encoded.reshape(sequence.size(0), -1))
        expert_summary = self.expert_norm(
            torch.cat([temporal_prediction, predictive_residual, risk_prediction], dim=-1)
        )
        state = torch.cat([flattened, expert_summary, external_state], dim=-1)
        predictions = {
            "temporal_return": temporal_prediction,
            "predictive_residual": predictive_residual,
            "predictive_return": predictive_full_return,
            "combined_return": combined_return,
            "risk_scale": risk_prediction,
            **diagnostics,
        }
        return state, route_gates, predictions

    def _optimizer_base(self, predictions: Mapping[str, torch.Tensor], risk_adjacency: torch.Tensor, external_state: torch.Tensor, cash_bounds: torch.Tensor) -> torch.Tensor:
        current = external_state[:, : self.num_risk_assets + 1]
        # External layout ends with regime change and entropy; crisis probability is
        # the last regime probability immediately before those two fields.
        crisis = external_state[:, -3:-2]
        uncertainty = external_state[:, -1:]
        active_risk_adjacency = risk_adjacency
        if not self.use_risk_graph:
            identity = torch.eye(
                self.num_risk_assets,
                dtype=risk_adjacency.dtype,
                device=risk_adjacency.device,
            )
            active_risk_adjacency = identity.expand(risk_adjacency.size(0), -1, -1)
        return self.portfolio_layer(
            predictions["combined_return"],
            predictions["risk_scale"],
            active_risk_adjacency,
            current,
            cash_bounds,
            crisis,
            uncertainty,
        )

    def _active_base(self, predictions: Mapping[str, torch.Tensor], risk_adjacency: torch.Tensor, external_state: torch.Tensor, supplied_base: torch.Tensor | None, cash_bounds: torch.Tensor) -> torch.Tensor:
        if self.use_optimizer:
            optimizer_base = self._optimizer_base(predictions, risk_adjacency, external_state, cash_bounds)
            if self.use_benchmark_anchor and supplied_base is not None:
                return 0.25 * supplied_base + 0.75 * optimizer_base
            return optimizer_base
        if supplied_base is not None:
            return supplied_base
        batch = external_state.size(0)
        cash = torch.full((batch, 1), min(0.05, self.maximum_cash), device=external_state.device, dtype=external_state.dtype)
        risk = torch.full((batch, self.num_risk_assets), 1.0 / self.num_risk_assets, device=external_state.device, dtype=external_state.dtype)
        return torch.cat([risk * (1.0 - cash), cash], dim=-1)

    def build_state(self, sequence: torch.Tensor, risk_adjacency: torch.Tensor, predictive_adjacency: torch.Tensor, external_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        state, route_gates, _ = self.build_state_extended(sequence, risk_adjacency, predictive_adjacency, external_state)
        return state, route_gates

    def sample_action(self, sequence: torch.Tensor, risk_adjacency: torch.Tensor, predictive_adjacency: torch.Tensor, external_state: torch.Tensor, base_weights: torch.Tensor | None = None, cash_bounds: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if cash_bounds is None:
            cash_bounds = torch.tensor([0.0, self.maximum_cash], device=external_state.device, dtype=external_state.dtype).expand(external_state.size(0), -1)
        state, _, predictions = self.build_state_extended(sequence, risk_adjacency, predictive_adjacency, external_state)
        base = self._active_base(predictions, risk_adjacency, external_state, base_weights, cash_bounds)
        action, log_probability, entropy, _ = self.actor.sample(state, base, cash_bounds, anchor_enabled=self.use_anchor)
        value, _ = self.critic(state)
        return action, log_probability, entropy, value

    def sample_action_with_costs(self, sequence: torch.Tensor, risk_adjacency: torch.Tensor, predictive_adjacency: torch.Tensor, external_state: torch.Tensor, base_weights: torch.Tensor | None = None, cash_bounds: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if cash_bounds is None:
            cash_bounds = torch.tensor([0.0, self.maximum_cash], device=external_state.device, dtype=external_state.dtype).expand(external_state.size(0), -1)
        state, _, predictions = self.build_state_extended(sequence, risk_adjacency, predictive_adjacency, external_state)
        base = self._active_base(predictions, risk_adjacency, external_state, base_weights, cash_bounds)
        action, log_probability, entropy, _ = self.actor.sample(state, base, cash_bounds, anchor_enabled=self.use_anchor)
        value, _ = self.critic(state)
        return action, log_probability, entropy, value, self.critic.cost_values(state)

    def evaluate_actions_extended(self, sequence: torch.Tensor, risk_adjacency: torch.Tensor, predictive_adjacency: torch.Tensor, external_state: torch.Tensor, base_weights: torch.Tensor, cash_bounds: torch.Tensor, actions: torch.Tensor) -> tuple[Any, ...]:
        state, route_gates, predictions = self.build_state_extended(sequence, risk_adjacency, predictive_adjacency, external_state)
        base = self._active_base(predictions, risk_adjacency, external_state, base_weights, cash_bounds)
        predictions["optimizer_base"] = base
        log_probability, entropy, base_reliance = self.actor.evaluate(state, actions, base, cash_bounds, anchor_enabled=self.use_anchor)
        value, quantiles = self.critic(state)
        return log_probability, entropy, value, quantiles, base_reliance, route_gates, self.critic.cost_values(state), predictions, state

    def deterministic_action(self, sequence: torch.Tensor, risk_adjacency: torch.Tensor, predictive_adjacency: torch.Tensor, external_state: torch.Tensor, base_weights: torch.Tensor | None = None, cash_bounds: torch.Tensor | None = None) -> torch.Tensor:
        if cash_bounds is None:
            cash_bounds = torch.tensor([0.0, self.maximum_cash], device=external_state.device, dtype=external_state.dtype).expand(external_state.size(0), -1)
        state, _, predictions = self.build_state_extended(sequence, risk_adjacency, predictive_adjacency, external_state)
        base = self._active_base(predictions, risk_adjacency, external_state, base_weights, cash_bounds)
        action, _ = self.actor.deterministic(state, base, cash_bounds, anchor_enabled=self.use_anchor)
        return action

    def deterministic_base(self, sequence: torch.Tensor, risk_adjacency: torch.Tensor, predictive_adjacency: torch.Tensor, external_state: torch.Tensor, base_weights: torch.Tensor | None = None, cash_bounds: torch.Tensor | None = None) -> torch.Tensor:
        if cash_bounds is None:
            cash_bounds = torch.tensor([0.0, self.maximum_cash], device=external_state.device, dtype=external_state.dtype).expand(external_state.size(0), -1)
        _, _, predictions = self.build_state_extended(sequence, risk_adjacency, predictive_adjacency, external_state)
        return self._active_base(predictions, risk_adjacency, external_state, base_weights, cash_bounds)

    def predict_quantiles(self, sequence: torch.Tensor, risk_adjacency: torch.Tensor, predictive_adjacency: torch.Tensor, external_state: torch.Tensor) -> torch.Tensor:
        state, _, _ = self.build_state_extended(sequence, risk_adjacency, predictive_adjacency, external_state)
        _, quantiles = self.critic(state)
        return quantiles

    def predict_values(self, sequence: torch.Tensor, risk_adjacency: torch.Tensor, predictive_adjacency: torch.Tensor, external_state: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        state, _, _ = self.build_state_extended(sequence, risk_adjacency, predictive_adjacency, external_state)
        value, _ = self.critic(state)
        return value, self.critic.cost_values(state)

    def encoder_parameters(self) -> list[nn.Parameter]:
        return list(self.encoder.parameters())

    def policy_parameters(self) -> list[nn.Parameter]:
        return (
            list(self.actor.parameters())
            + list(self.portfolio_layer.parameters())
            + list(self.encoded_norm.parameters())
            + list(self.expert_norm.parameters())
            + list(self.temporal_return_head.parameters())
            + list(self.predictive_residual_head.parameters())
            + list(self.risk_scale_head.parameters())
        )

    def critic_parameters(self) -> list[nn.Parameter]:
        return list(self.critic.parameters())

    def set_encoder_trainable(self, trainable: bool) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(bool(trainable))

    def optimisation_parameter_groups(
        self,
        *,
        actor_learning_rate: float,
        critic_learning_rate: float,
        encoder_learning_rate: float | None = None,
    ) -> list[dict[str, Any]]:
        return [
            {
                "params": self.encoder_parameters(),
                "lr": float(actor_learning_rate if encoder_learning_rate is None else encoder_learning_rate),
            },
            {"params": self.policy_parameters(), "lr": float(actor_learning_rate)},
            {"params": self.critic_parameters(), "lr": float(critic_learning_rate)},
        ]



def behaviour_cloning_loss(
    agent: XGATDRLAgent,
    states: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    teacher_actions: torch.Tensor,
) -> torch.Tensor:
    sequence, risk_adjacency, predictive_adjacency, external, base_weights, cash_bounds = states
    state, _, predictions = agent.build_state_extended(
        sequence, risk_adjacency, predictive_adjacency, external
    )
    active_base = agent._active_base(
        predictions,
        risk_adjacency,
        external,
        base_weights,
        cash_bounds,
    )
    log_probability, _, _ = agent.actor.evaluate(
        state,
        teacher_actions,
        active_base,
        cash_bounds,
        anchor_enabled=agent.use_anchor,
    )
    return -log_probability.mean()


def compute_mmd_loss(y_true: torch.Tensor, y_pred_samples: torch.Tensor, sigma: float = 0.04) -> torch.Tensor:
    """Biased squared MMD with an inverse-multiquadric kernel."""
    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    if y_true.ndim != 2 or y_pred_samples.ndim != 2:
        raise ValueError("MMD inputs must have shape [samples, features].")
    if y_true.size(1) != y_pred_samples.size(1):
        raise ValueError("MMD inputs must have the same feature dimension.")

    scale = 2.0 * float(sigma) ** 2

    def kernel(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        squared_distance = torch.cdist(first, second).square()
        return scale / (scale + squared_distance)

    true_kernel = kernel(y_true, y_true)
    predicted_kernel = kernel(y_pred_samples, y_pred_samples)
    cross_kernel = kernel(y_true, y_pred_samples)
    return true_kernel.mean() + predicted_kernel.mean() - 2.0 * cross_kernel.mean()

def hybrid_pretraining_loss(
    agent: XGATDRLAgent,
    states: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    realised_asset_returns: torch.Tensor,
    future_risk_targets: torch.Tensor,
    previous_weights: torch.Tensor,
    *,
    target_scale: float = 1.0,
    transaction_cost: float = 0.0,
    slippage_coefficient: float = 0.0,
    impact_coefficient: float = 0.0,
    downside_penalty: float = 0.10,
    concentration_penalty: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    sequence, risk_adjacency, predictive_adjacency, external, supplied_base, cash_bounds = states
    state, gates, predictions = agent.build_state_extended(
        sequence, risk_adjacency, predictive_adjacency, external
    )
    base = agent._active_base(predictions, risk_adjacency, external, supplied_base, cash_bounds)
    action, _ = agent.actor.deterministic(state, base, cash_bounds, anchor_enabled=agent.use_anchor)
    
    target = realised_asset_returns[..., : agent.num_risk_assets] * float(target_scale)
    temporal_error = F.smooth_l1_loss(predictions["temporal_return"], target, reduction="none")
    residual_target = target - predictions["temporal_return"].detach()
    predictive_error = F.smooth_l1_loss(
        predictions["predictive_residual"], residual_target, reduction="none"
    )
    risk_target = future_risk_targets * float(target_scale)
    risk_error = F.smooth_l1_loss(predictions["risk_scale"], risk_target, reduction="none")

    predictive_full_error = F.smooth_l1_loss(
        predictions["predictive_return"], target, reduction="none"
    )
    
    mmd_temporal_loss = compute_mmd_loss(target, predictions["temporal_return"])
    mmd_predictive_loss = compute_mmd_loss(target, predictions["predictive_return"])

    gate_scores = -torch.stack(
        [temporal_error.detach(), predictive_full_error.detach()], dim=-1
    )
    predictive_available = gates[..., 2] > 1e-8
    gate_scores[..., 1] = gate_scores[..., 1].masked_fill(~predictive_available, -1e9)
    gate_target = F.softmax(gate_scores / 0.25, dim=-1)
    return_gates = gates[..., (0, 2)].clamp_min(1e-8)
    gate_loss = -(gate_target * torch.log(return_gates)).sum(dim=-1).mean()

    growth = torch.sum(action * torch.exp(realised_asset_returns), dim=-1).clamp_min(1e-12)
    gross_return = torch.log(growth)
    turnover = 0.5 * torch.sum(torch.abs(action - previous_weights), dim=-1)
    cost = (
        float(transaction_cost) * turnover
        + float(slippage_coefficient) * torch.sqrt(turnover + 1e-12)
        + float(impact_coefficient) * turnover.square()
    )
    net_return = gross_return - cost
    allocation_loss = (
        -net_return.mean()
        + float(downside_penalty) * torch.relu(-net_return).square().mean()
        + float(concentration_penalty) * action.square().sum(dim=-1).mean()
    )
    
    expert_loss = (
        temporal_error.mean() + predictive_error.mean()
        + predictive_full_error.mean() + risk_error.mean()
        + 0.5 * (mmd_temporal_loss + mmd_predictive_loss) 
    )
    total = expert_loss + 0.10 * gate_loss + allocation_loss
    
    return total, {
        "Pretrain loss": float(total.detach()),
        "Pretrain expert loss": float(expert_loss.detach()),
        "Pretrain gate loss": float(gate_loss.detach()),
        "Pretrain allocation loss": float(allocation_loss.detach()),
        "Pretrain MMD loss": float((mmd_temporal_loss + mmd_predictive_loss).detach()),
    }
    
    
def constrained_ppo_update_batch(
    agent: XGATDRLAgent,
    optimiser: torch.optim.Optimizer | Mapping[str, torch.optim.Optimizer],
    states: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    actions: torch.Tensor,
    old_log_probabilities: torch.Tensor,
    old_values: torch.Tensor,
    returns: torch.Tensor,
    advantages: torch.Tensor,
    *,
    config: PPOConfig = PPOConfig(),
    quantile_targets: torch.Tensor | None = None,
    quantile_coefficient: float = 0.10,
    kl_coefficient: float = 0.0,
    old_cost_values: Mapping[str, torch.Tensor] | None = None,
    cost_returns: Mapping[str, torch.Tensor] | None = None,
    cost_value_coefficient: float = 0.25,
    realised_asset_returns: torch.Tensor | None = None,
    previous_weights: torch.Tensor | None = None,
    direct_utility_coefficient: float = 0.0,
    expert_auxiliary_coefficient: float = 0.0,
    gate_auxiliary_coefficient: float = 0.0,
    auxiliary_target_scale: float = 1.0,
    transaction_cost: float = 0.0,
    slippage_coefficient: float = 0.0,
    impact_coefficient: float = 0.0,
    direct_downside_penalty: float = 0.10,
    direct_concentration_penalty: float = 0.01,
) -> dict[str, float | bool]:
    (
        log_probability,
        entropy,
        values,
        quantiles,
        anchor,
        route_gates,
        predicted_cost_values,
        predictions,
        state,
    ) = agent.evaluate_actions_extended(*states, actions)
    old_log_probabilities = old_log_probabilities.reshape_as(log_probability).detach()
    old_values = old_values.reshape_as(values).detach()
    returns = returns.reshape_as(values).detach()
    advantages = advantages.reshape_as(values).detach()
    tensors_to_check = [
        log_probability,
        entropy,
        values,
        quantiles,
        old_log_probabilities,
        old_values,
        returns,
        advantages,
    ]
    tensors_to_check.extend(predicted_cost_values.values())
    if not all(torch.isfinite(tensor).all() for tensor in tensors_to_check):
        raise FloatingPointError("PPO tensors contain non-finite values.")

    log_ratio = torch.clamp(log_probability - old_log_probabilities, -20.0, 20.0)
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - config.clip_epsilon, 1.0 + config.clip_epsilon) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()

    if config.value_clip is None:
        value_loss = F.huber_loss(values, returns, delta=1.0)
    else:
        clipped_values = old_values + torch.clamp(values - old_values, -config.value_clip, config.value_clip)
        value_loss = torch.maximum(
            F.huber_loss(values, returns, delta=1.0, reduction="none"),
            F.huber_loss(clipped_values, returns, delta=1.0, reduction="none"),
        ).mean()

    active_quantile_targets = returns if quantile_targets is None else quantile_targets
    quantile_loss = quantile_huber_loss(quantiles, active_quantile_targets, agent.critic.quantile_levels)

    cost_value_loss = torch.zeros((), dtype=values.dtype, device=values.device)
    if cost_returns:
        losses = []
        for name, target in cost_returns.items():
            if name not in predicted_cost_values:
                continue
            prediction = predicted_cost_values[name]
            target = target.reshape_as(prediction).detach()
            if old_cost_values is not None and name in old_cost_values and config.value_clip is not None:
                old_prediction = old_cost_values[name].reshape_as(prediction).detach()
                clipped_prediction = old_prediction + torch.clamp(
                    prediction - old_prediction, -config.value_clip, config.value_clip
                )
                item = torch.maximum(
                    F.huber_loss(prediction, target, delta=1.0, reduction="none"),
                    F.huber_loss(clipped_prediction, target, delta=1.0, reduction="none"),
                ).mean()
            else:
                item = F.huber_loss(prediction, target, delta=1.0)
            losses.append(item)
        if losses:
            cost_value_loss = torch.stack(losses).mean()

    direct_utility_loss = torch.zeros((), dtype=values.dtype, device=values.device)
    if realised_asset_returns is not None and previous_weights is not None and direct_utility_coefficient > 0.0:
        active_base = predictions.get("optimizer_base", states[4])
        mean_action, _ = agent.actor.deterministic(
            state,
            active_base,
            states[5],
            anchor_enabled=agent.use_anchor,
        )
        gross_growth = torch.sum(mean_action * torch.exp(realised_asset_returns), dim=-1).clamp_min(1e-12)
        gross_return = torch.log(gross_growth)
        turnover = 0.5 * torch.sum(torch.abs(mean_action - previous_weights), dim=-1)
        execution_cost = (
            float(transaction_cost) * turnover
            + float(slippage_coefficient) * torch.sqrt(turnover.clamp_min(0.0) + 1e-12)
            + float(impact_coefficient) * turnover.square()
        )
        net_return = gross_return - execution_cost
        downside = torch.relu(-net_return).square()
        concentration = mean_action.square().sum(dim=-1)
        direct_utility_loss = (
            -net_return.mean()
            + float(direct_downside_penalty) * downside.mean()
            + float(direct_concentration_penalty) * concentration.mean()
        )

    expert_loss = torch.zeros((), dtype=values.dtype, device=values.device)
    gate_loss = torch.zeros((), dtype=values.dtype, device=values.device)
    if realised_asset_returns is not None and (expert_auxiliary_coefficient > 0.0 or gate_auxiliary_coefficient > 0.0):
        target = realised_asset_returns[..., : agent.num_risk_assets] * float(auxiliary_target_scale)
        risk_target = target.abs()
        temporal_error = F.smooth_l1_loss(predictions["temporal_return"], target, reduction="none")
        residual_target = target - predictions["temporal_return"].detach()
        predictive_residual_error = F.smooth_l1_loss(
            predictions["predictive_residual"], residual_target, reduction="none"
        )
        predictive_full_error = F.smooth_l1_loss(
            predictions["predictive_return"], target, reduction="none"
        )
        risk_error = F.smooth_l1_loss(predictions["risk_scale"], risk_target, reduction="none")
        combined_error = F.smooth_l1_loss(predictions["combined_return"], target)
        expert_loss = (
            temporal_error.mean() + predictive_residual_error.mean()
            + predictive_full_error.mean() + risk_error.mean() + combined_error
        )
        if gate_auxiliary_coefficient > 0.0:
            gate_scores = -torch.stack(
                [temporal_error.detach(), predictive_full_error.detach()], dim=-1
            )
            predictive_available = route_gates[..., 2] > 1e-8
            gate_scores[..., 1] = gate_scores[..., 1].masked_fill(~predictive_available, -1e9)
            gate_target = F.softmax(gate_scores / 0.25, dim=-1)
            return_gates = route_gates[..., (0, 2)].clamp_min(1e-8)
            gate_loss = -(gate_target * torch.log(return_gates)).sum(dim=-1).mean()

    entropy_bonus = entropy.mean()
    kl_penalty = 0.5 * log_ratio.square().mean()
    loss = (
        policy_loss
        + config.value_coefficient * value_loss
        + float(quantile_coefficient) * quantile_loss
        + float(cost_value_coefficient) * cost_value_loss
        + float(direct_utility_coefficient) * direct_utility_loss
        + float(expert_auxiliary_coefficient) * expert_loss
        + float(gate_auxiliary_coefficient) * gate_loss
        + float(kl_coefficient) * kl_penalty
        - config.entropy_coefficient * entropy_bonus
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("PPO loss is not finite.")

    active_optimisers = list(optimiser.values()) if isinstance(optimiser, Mapping) else [optimiser]
    for active_optimiser in active_optimisers:
        active_optimiser.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(
        [parameter for parameter in agent.parameters() if parameter.requires_grad],
        config.max_gradient_norm,
    )
    for active_optimiser in active_optimisers:
        active_optimiser.step()

    with torch.no_grad():
        approximate_kl = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio)
        clip_fraction = torch.mean((torch.abs(ratio - 1.0) > config.clip_epsilon).float())
        stop_for_kl = bool(config.target_kl is not None and float(approximate_kl) > 1.5 * config.target_kl)
    return {
        "Loss": float(loss.item()),
        "Policy loss": float(policy_loss.item()),
        "Value loss": float(value_loss.item()),
        "Quantile loss": float(quantile_loss.item()),
        "Cost value loss": float(cost_value_loss.item()),
        "Direct utility loss": float(direct_utility_loss.item()),
        "Expert auxiliary loss": float(expert_loss.item()),
        "Gate auxiliary loss": float(gate_loss.item()),
        "Base entropy": float(entropy_bonus.item()),
        "Approximate KL": float(approximate_kl.item()),
        "KL coefficient": float(kl_coefficient),
        "Clip fraction": float(clip_fraction.item()),
        "Gradient norm": float(gradient_norm.item()),
        "Mean optimiser-base reliance": float(anchor.mean().item()),
        "Mean temporal-return gate": float(route_gates[..., 0].mean().item()),
        "Mean risk-expert usage": float(route_gates[..., 1].mean().item()),
        "Mean predictive-return gate": float(route_gates[..., 2].mean().item()),
        "Stop for KL": stop_for_kl,
    }


def critic_regression_update_batch(
    agent: XGATDRLAgent,
    optimiser: torch.optim.Optimizer,
    states: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    returns: torch.Tensor,
    *,
    quantile_targets: torch.Tensor | None = None,
    quantile_coefficient: float = 0.10,
    cost_returns: Mapping[str, torch.Tensor] | None = None,
    cost_value_coefficient: float = 0.25,
    max_gradient_norm: float = 0.50,
) -> dict[str, float]:
    """Perform a critic-only regression step on a detached policy state."""
    sequence, risk_adjacency, predictive_adjacency, external_state, _, _ = states
    with torch.no_grad():
        state, _, _ = agent.build_state_extended(
            sequence, risk_adjacency, predictive_adjacency, external_state
        )
    state = state.detach()
    values, quantiles = agent.critic(state)
    targets = returns.reshape_as(values).detach()
    value_loss = F.huber_loss(values, targets, delta=1.0)
    active_quantile_targets = targets if quantile_targets is None else quantile_targets.detach()
    quantile_loss = quantile_huber_loss(
        quantiles, active_quantile_targets, agent.critic.quantile_levels
    )
    predicted_costs = agent.critic.cost_values(state)
    cost_loss = torch.zeros((), dtype=values.dtype, device=values.device)
    if cost_returns:
        items = [
            F.huber_loss(predicted_costs[name], target.reshape_as(predicted_costs[name]).detach(), delta=1.0)
            for name, target in cost_returns.items()
            if name in predicted_costs
        ]
        if items:
            cost_loss = torch.stack(items).mean()
    loss = value_loss + float(quantile_coefficient) * quantile_loss + float(cost_value_coefficient) * cost_loss
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = nn.utils.clip_grad_norm_(agent.critic.parameters(), float(max_gradient_norm))
    optimiser.step()
    return {
        "Critic-only loss": float(loss.detach()),
        "Critic-only value loss": float(value_loss.detach()),
        "Critic-only quantile loss": float(quantile_loss.detach()),
        "Critic-only cost loss": float(cost_loss.detach()),
        "Critic-only gradient norm": float(gradient_norm),
    }

# -----------------------------------------------------------------------------
# 12. Hierarchical evaluation and scenario-level figures
# -----------------------------------------------------------------------------

METHOD_VISUAL_STYLES: dict[str, dict[str, Any]] = {
    "X-GAT-DRL": {"color": "#000000", "linestyle": "-", "marker": "o", "linewidth": 2.6},
    "1/N": {"color": "#E69F00", "linestyle": "--", "marker": "s", "linewidth": 1.6},
    "GMV": {"color": "#0072B2", "linestyle": ":", "marker": "^", "linewidth": 1.6},
    "HMM-GMV": {"color": "#56B4E9", "linestyle": "-.", "marker": "D", "linewidth": 1.6},
    "GLASSO-GAT": {"color": "#009E73", "linestyle": "-", "marker": "P", "linewidth": 1.8},
    "TC-MAC": {"color": "#D55E00", "linestyle": "--", "marker": "X", "linewidth": 1.6},
    "JM-MPC": {"color": "#CC79A7", "linestyle": ":", "marker": "*", "linewidth": 1.6},
}

def interquartile_mean(values: ArrayLike) -> float:
    """Exact empirical mean over the central 50% of probability mass."""
    array = np.sort(_as_finite_array(values, name="values", ndim=1, minimum_length=1))
    lower = 0.25 * array.size
    upper = 0.75 * array.size
    indices = np.arange(array.size, dtype=float)
    weights = np.maximum(
        0.0,
        np.minimum(indices + 1.0, upper) - np.maximum(indices, lower),
    )
    return float(np.dot(weights, array) / max(upper - lower, 1e-12))


def hierarchical_metric_summary(
    metrics: pd.DataFrame,
    *,
    metric_columns: Sequence[str],
    scenario_column: str = "Scenario",
    method_column: str = "Method",
    market_column: str = "Market seed",
    policy_column: str = "Policy seed",
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    random_state: int = 0,
) -> pd.DataFrame:
    required = {scenario_column, method_column, market_column, policy_column, *metric_columns}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Missing hierarchical columns: {missing}")
    rng = np.random.default_rng(random_state)
    alpha = 0.5 * (1.0 - confidence_level)
    rows: list[dict[str, Any]] = []
    grouped = metrics.groupby([scenario_column, method_column], sort=False, dropna=False)
    for (scenario, method), group in grouped:
        for metric in metric_columns:
            market_values = (
                group.groupby(market_column, sort=False)[metric]
                .mean()
                .dropna()
                .to_numpy(dtype=float)
            )
            if market_values.size == 0:
                continue
            bootstrap = np.empty(max(1, int(n_bootstrap)), dtype=float)
            for index in range(bootstrap.size):
                draw = rng.choice(market_values, size=market_values.size, replace=True)
                bootstrap[index] = interquartile_mean(draw)
            rows.append(
                {
                    "Scenario": scenario,
                    "Method": method,
                    "Metric": metric,
                    "Independent markets": int(market_values.size),
                    "Policy runs per market": int(group[policy_column].nunique()),
                    "Mean": float(market_values.mean()),
                    "Median": float(np.median(market_values)),
                    "IQM": interquartile_mean(market_values),
                    "Standard error": float(bootstrap.std(ddof=1)) if bootstrap.size > 1 else float("nan"),
                    "Confidence low": float(np.quantile(bootstrap, alpha)),
                    "Confidence high": float(np.quantile(bootstrap, 1.0 - alpha)),
                }
            )
    return pd.DataFrame(rows)


def _sign_flip_p_value(differences: np.ndarray, *, repetitions: int, rng: np.random.Generator) -> float:
    values = _as_finite_array(differences, name="differences", ndim=1, minimum_length=1)
    observed = abs(interquartile_mean(values))
    if values.size <= 20:
        total = 1 << values.size
        if total <= max(2, repetitions):
            statistics = np.empty(total, dtype=float)
            for mask in range(total):
                signs = np.array([1.0 if mask & (1 << bit) else -1.0 for bit in range(values.size)])
                statistics[mask] = abs(interquartile_mean(signs * values))
            return float((1.0 + np.sum(statistics >= observed)) / (statistics.size + 1.0))
    count = 0
    repetitions = max(1, int(repetitions))
    for _ in range(repetitions):
        signs = rng.choice((-1.0, 1.0), size=values.size)
        count += int(abs(interquartile_mean(signs * values)) >= observed)
    return float((count + 1.0) / (repetitions + 1.0))


def _holm_adjust_local(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (values.size - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def hierarchical_paired_comparisons(
    metrics: pd.DataFrame,
    *,
    strategy: str,
    benchmarks: Sequence[str],
    metric_directions: Mapping[str, bool],
    scenario_column: str = "Scenario",
    method_column: str = "Method",
    market_column: str = "Market seed",
    policy_column: str = "Policy seed",
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    sign_flip_repetitions: int = 10_000,
    random_state: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    alpha = 0.5 * (1.0 - confidence_level)
    market_level = (
        metrics.groupby([scenario_column, method_column, market_column], sort=False)[
            list(metric_directions)
        ]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, Any]] = []
    for scenario in market_level[scenario_column].drop_duplicates():
        scenario_table = market_level.loc[market_level[scenario_column] == scenario]
        for metric, higher_is_better in metric_directions.items():
            strategy_table = scenario_table.loc[
                scenario_table[method_column] == strategy, [market_column, metric]
            ].rename(columns={metric: "strategy"})
            for benchmark in benchmarks:
                benchmark_table = scenario_table.loc[
                    scenario_table[method_column] == benchmark, [market_column, metric]
                ].rename(columns={metric: "benchmark"})
                paired = strategy_table.merge(benchmark_table, on=market_column, how="inner")
                if paired.empty:
                    continue
                sign = 1.0 if higher_is_better else -1.0
                differences = sign * (
                    paired["strategy"].to_numpy(dtype=float)
                    - paired["benchmark"].to_numpy(dtype=float)
                )
                bootstrap = np.empty(max(1, int(n_bootstrap)), dtype=float)
                for index in range(bootstrap.size):
                    draw = rng.choice(differences, size=differences.size, replace=True)
                    bootstrap[index] = interquartile_mean(draw)
                rows.append(
                    {
                        "Scenario": scenario,
                        "Metric": metric,
                        "Benchmark": benchmark,
                        "Independent paired markets": int(differences.size),
                        "Direction-adjusted effect": interquartile_mean(differences),
                        "Effect estimator": "IQM",
                        "Confidence low": float(np.quantile(bootstrap, alpha)),
                        "Confidence high": float(np.quantile(bootstrap, 1.0 - alpha)),
                        "Probability superior": float(np.mean(bootstrap > 0.0)),
                        "Sign-flip p-value": _sign_flip_p_value(
                            differences,
                            repetitions=sign_flip_repetitions,
                            rng=rng,
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["Holm-adjusted p-value"] = np.nan
        for (_, metric), index in result.groupby(["Scenario", "Metric"]).groups.items():
            result.loc[index, "Holm-adjusted p-value"] = _holm_adjust_local(
                result.loc[index, "Sign-flip p-value"].to_numpy(dtype=float)
            )
    return result


def plot_grouped_scenario_boxplots(
    metrics: pd.DataFrame,
    *,
    metric: str,
    method_order: Sequence[str],
    scenario_order: Sequence[str],
    save_path: PathLike | None = None,
) -> plt.Figure:
    """Draw scenario-dodged boxplots inside each method group."""
    required = {"Method", "Scenario", metric}
    if not required.issubset(metrics.columns):
        raise ValueError(f"metrics must contain {sorted(required)}")
    figure, axis = plt.subplots(figsize=(max(10.0, 1.55 * len(method_order)), 6.4))
    offsets = np.linspace(-0.28, 0.28, max(1, len(scenario_order)))
    width = min(0.22, 0.72 / max(1, len(scenario_order)))
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
    handles = []
    for scenario_index, scenario in enumerate(scenario_order):
        data = []
        positions = []
        for method_index, method in enumerate(method_order):
            values = metrics.loc[
                (metrics["Method"] == method) & (metrics["Scenario"] == scenario), metric
            ].dropna().to_numpy(dtype=float)
            data.append(values if values.size else np.asarray([np.nan]))
            positions.append(method_index + offsets[scenario_index])
        colour = palette[scenario_index % len(palette)]
        box = axis.boxplot(
            data,
            positions=positions,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
        )
        for patch in box["boxes"]:
            patch.set_facecolor(colour)
            patch.set_alpha(0.58)
        handles.append(plt.Line2D([0], [0], color=colour, linewidth=8, alpha=0.58, label=str(scenario).replace("_", " ").title()))
    axis.set_xticks(np.arange(len(method_order)))
    axis.set_xticklabels(method_order, rotation=25, ha="right")
    axis.set_ylabel(metric)
    axis.set_title(f"{metric} across scenarios")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(handles=handles, title="Scenario", frameon=False, ncol=min(3, len(handles)))
    figure.tight_layout()
    _save_or_close(figure, save_path)
    return figure


def plot_wealth_facets(
    wealth_paths: Mapping[str, Mapping[str, np.ndarray]],
    *,
    scenario_order: Sequence[str],
    method_order: Sequence[str],
    save_path: PathLike | None = None,
) -> plt.Figure:
    """Facet cumulative-wealth bands by scenario."""
    active = [scenario for scenario in scenario_order if scenario in wealth_paths]
    if not active:
        raise ValueError("No requested scenarios are available.")
    columns = 2 if len(active) > 1 else 1
    rows = int(math.ceil(len(active) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(7.2 * columns, 4.8 * rows), squeeze=False)
    for axis, scenario in zip(axes.ravel(), active):
        for method in method_order:
            if method not in wealth_paths[scenario]:
                continue
            paths = np.asarray(wealth_paths[scenario][method], dtype=float)
            centre = np.median(paths, axis=0)
            low = np.quantile(paths, 0.10, axis=0)
            high = np.quantile(paths, 0.90, axis=0)
            style = METHOD_VISUAL_STYLES.get(method, {})
            axis.plot(centre, label=method, **{k: v for k, v in style.items() if k != "marker"})
            axis.fill_between(np.arange(centre.size), low, high, color=style.get("color"), alpha=0.10)
        axis.set_title(str(scenario).replace("_", " ").title())
        axis.set_xlabel("Out-of-sample step")
        axis.set_ylabel("Portfolio wealth")
        axis.grid(alpha=0.22)
    for axis in axes.ravel()[len(active):]:
        axis.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=max(1, len(labels)), frameon=False,
    )
    figure.tight_layout(rect=(0, 0.16, 1, 1))
    figure._xgat_layout_done = True
    _save_or_close(figure, save_path)
    return figure


def plot_drawdown_facets(
    wealth_paths: Mapping[str, Mapping[str, np.ndarray]],
    *,
    scenario_order: Sequence[str],
    method_order: Sequence[str],
    save_path: PathLike | None = None,
) -> plt.Figure:
    """Facet median drawdown paths by scenario."""
    active = [scenario for scenario in scenario_order if scenario in wealth_paths]
    columns = 2 if len(active) > 1 else 1
    rows = int(math.ceil(len(active) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(7.2 * columns, 4.8 * rows), squeeze=False)
    for axis, scenario in zip(axes.ravel(), active):
        for method in method_order:
            if method not in wealth_paths[scenario]:
                continue
            paths = np.asarray(wealth_paths[scenario][method], dtype=float)
            drawdowns = paths / np.maximum.accumulate(paths, axis=1) - 1.0
            centre = np.median(drawdowns, axis=0)
            low = np.quantile(drawdowns, 0.10, axis=0)
            high = np.quantile(drawdowns, 0.90, axis=0)
            style = METHOD_VISUAL_STYLES.get(method, {})
            axis.plot(centre, label=method, **{k: v for k, v in style.items() if k != "marker"})
            axis.fill_between(np.arange(centre.size), low, high, color=style.get("color"), alpha=0.10)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(str(scenario).replace("_", " ").title())
        axis.set_xlabel("Out-of-sample step")
        axis.set_ylabel("Drawdown")
        axis.grid(alpha=0.22)
    for axis in axes.ravel()[len(active):]:
        axis.set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=max(1, len(labels)), frameon=False,
    )
    figure.tight_layout(rect=(0, 0.16, 1, 1))
    figure._xgat_layout_done = True
    _save_or_close(figure, save_path)
    return figure


def plot_accuracy_precision_intervals(
    summary: pd.DataFrame,
    *,
    accuracy_metric: str,
    precision_metric: str,
    title: str | None = None,
    save_path: PathLike | None = None,
) -> plt.Figure:
    """Plot accuracy and precision estimates with confidence intervals."""
    required = {"Scenario", "Metric", "Estimate", "Confidence low", "Confidence high"}
    if not required.issubset(summary.columns):
        raise ValueError(f"summary must contain {sorted(required)}")
    metrics = [accuracy_metric, precision_metric]
    labels = ["Accuracy", "Precision"]
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), squeeze=False)
    for axis, metric, label in zip(axes.ravel(), metrics, labels):
        frame = summary.loc[summary["Metric"] == metric].reset_index(drop=True)
        y = np.arange(frame.shape[0])
        estimates = frame["Estimate"].to_numpy(dtype=float)
        lower = frame["Confidence low"].to_numpy(dtype=float)
        upper = frame["Confidence high"].to_numpy(dtype=float)
        axis.errorbar(
            estimates, y,
            xerr=np.vstack([estimates - lower, upper - estimates]),
            fmt="o", capsize=4, linewidth=1.4,
        )
        axis.set_yticks(y)
        axis.set_yticklabels(
            frame["Scenario"].astype(str).str.replace("_", " ").str.title()
        )
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel(label)
        axis.grid(axis="x", alpha=0.25)
    if title:
        figure.suptitle(title)
        figure.tight_layout(rect=(0, 0, 1, 0.94))
    else:
        figure.tight_layout()
    figure._xgat_layout_done = True
    _save_or_close(figure, save_path)
    return figure


def plot_graph_recovery_intervals(
    summary: pd.DataFrame,
    *,
    save_path: PathLike | None = None,
) -> plt.Figure:
    """Plot predictive-graph accuracy and precision intervals."""
    return plot_accuracy_precision_intervals(
        summary,
        accuracy_metric="Predictive graph accuracy",
        precision_metric="Predictive graph precision",
        title="Predictive graph recovery",
        save_path=save_path,
    )


def plot_scenario_metric_intervals(
    summary: pd.DataFrame,
    *,
    metric: str,
    save_path: PathLike | None = None,
) -> None:
    table = summary.loc[summary["Metric"] == metric].copy()
    if table.empty:
        raise ValueError(f"No summary rows are available for {metric}.")
    scenarios = list(dict.fromkeys(table["Scenario"].astype(str)))
    figure, axes = plt.subplots(
        1, len(scenarios), figsize=(6.0 * len(scenarios), 5.5), squeeze=False, sharey=True
    )
    for column, scenario in enumerate(scenarios):
        axis = axes[0, column]
        subset = table.loc[table["Scenario"].astype(str) == scenario].reset_index(drop=True)
        positions = np.arange(subset.shape[0])
        for position, row in subset.iterrows():
            style = METHOD_VISUAL_STYLES.get(str(row["Method"]), {})
            estimate = float(row["IQM"])
            lower = estimate - float(row["Confidence low"])
            upper = float(row["Confidence high"]) - estimate
            axis.errorbar(
                estimate,
                position,
                xerr=np.array([[max(0.0, lower)], [max(0.0, upper)]]),
                fmt=style.get("marker", "o"),
                color=style.get("color", "black"),
                capsize=3,
                markersize=6,
            )
        axis.set_yticks(positions, labels=subset["Method"].astype(str))
        axis.axvline(0.0, color="0.5", linewidth=0.8, linestyle=":")
        axis.set_title(scenario.replace("_", " ").title())
        axis.set_xlabel(metric)
        axis.invert_yaxis()
    figure.suptitle(f"{metric}: policy seeds averaged within each market")
    _save_or_close(figure, save_path)


def plot_wealth_bands_by_scenario(
    wealth_paths: Mapping[str, Mapping[str, np.ndarray]],
    *,
    scenario: str,
    save_path: PathLike | None = None,
) -> None:
    if scenario not in wealth_paths:
        raise ValueError(f"Unknown scenario: {scenario}")
    figure, axis = plt.subplots(figsize=(11, 6))
    for method, paths in wealth_paths[scenario].items():
        matrix = _as_finite_array(paths, name=f"{scenario}-{method}", ndim=2)
        centre = np.median(matrix, axis=0)
        low = np.quantile(matrix, 0.10, axis=0)
        high = np.quantile(matrix, 0.90, axis=0)
        style = METHOD_VISUAL_STYLES.get(method, {})
        x_axis = np.arange(matrix.shape[1])
        axis.plot(
            x_axis,
            centre,
            label=method,
            color=style.get("color"),
            linestyle=style.get("linestyle", "-"),
            linewidth=style.get("linewidth", 1.7),
        )
        axis.fill_between(x_axis, low, high, color=style.get("color"), alpha=0.10)
    axis.set_title(f"Out-of-sample wealth by market: {scenario.replace('_', ' ').title()}")
    axis.set_xlabel("Out-of-sample step")
    axis.set_ylabel("Portfolio wealth")
    axis.legend(frameon=False, ncol=2)
    _save_or_close(figure, save_path)

