"""Comparative simulations for X-GAT-DRL and controlled benchmarks.

The file handles market generation, training, portfolio accounting, validation,
and result aggregation. Reusable components are imported from
``xgat_drl_code.py`` through a checked public API.
"""

from __future__ import annotations

import argparse
import copy
import gc
import gzip
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
import pickle
import shutil
import sysconfig
import tempfile

# Bound native BLAS/OpenMP pools before NumPy, scikit-learn, and PyTorch are
# imported.  This makes runtime reproducible and prevents oversubscription or
# interpreter-shutdown stalls on CPU-only research servers.
_native_thread_count = os.getenv("XGAT_NATIVE_THREADS", "1")
for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = _native_thread_count

import sys
import time
import traceback
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

LOGGER = logging.getLogger("xgat.simulations")


CHECKPOINT_SCHEMA_VERSION = 1


REQUIRED_CORE_API = (
    "PortfolioConstraints",
    "PPOConfig",
    "RiskBudgetConfig",
    "LagrangeController",
    "seed_everything",
    "validate_weights",
    "project_long_only_weights",
    "robust_scale_tensor",
    "build_causal_feature_tensor",
    "build_hmm_observations",
    "fit_hmm_with_restarts",
    "filtered_hmm_probabilities",
    "estimate_covariance",
    "compute_regime_weighted_covariance",
    "fit_glasso_at_alpha",
    "stability_selected_glasso_graph",
    "stability_selected_predictive_graph",
    "sparsify_signed_graph",
    "XGATDRLAgent",
    "GLASSOGATBenchmark",
    "TCMACBenchmark",
    "ModelPredictiveControlOptimiser",
    "JumpModelMPCBenchmark",
    "compute_gae",
    "constrained_ppo_update_batch",
    "critic_regression_update_batch",
    "behaviour_cloning_loss",
    "hybrid_pretraining_loss",
    "cash_bounds_from_signals",
    "regime_dependent_limits",
    "probabilistic_sharpe_ratio",
    "deflated_sharpe_ratio",
    "compute_model_confidence_set",
    "run_statistical_tests",
    "hierarchical_metric_summary",
    "hierarchical_paired_comparisons",
    "plot_scenario_metric_intervals",
    "plot_wealth_bands_by_scenario",
    "plot_cumulative_wealth",
    "plot_drawdowns",
    "plot_allocation_area",
    "plot_network_topology",
    "plot_pareto_frontier",
    "plot_return_cvar_facets",
    "plot_graph_recovery_diagnostics",
    "plot_metric_distributions",
    "plot_grouped_scenario_boxplots",
    "plot_wealth_facets",
    "plot_drawdown_facets",
    "plot_graph_recovery_intervals",
    "plot_accuracy_precision_intervals",
)


METHOD_ORDER = (
    "X-GAT-DRL",
    "1/N",
    "GMV",
    "HMM-GMV",
    "GLASSO-GAT",
    "TC-MAC",
    "JM-MPC",
)


EXPERT_ORDER = (
    "1/N",
    "GMV",
    "HMM-GMV",
    "GLASSO-GAT",
    "TC-MAC",
    "JM-MPC",
)



def _load_core_module(explicit_path: str | None = None) -> ModuleType:
    candidate: Path | None = None
    requested = explicit_path or os.getenv("XGAT_CORE_PATH")
    if requested:
        candidate = Path(requested).expanduser().resolve()
    else:
        adjacent = Path(__file__).resolve().with_name("xgat_drl_code.py")
        if adjacent.is_file():
            candidate = adjacent

    if candidate is None:
        try:
            return importlib.import_module("xgat_drl_code")
        except ImportError as exc:
            raise ImportError(
                "Place xgat_drl_code.py beside simulations.py or pass --core-path."
            ) from exc

    if not candidate.is_file():
        raise FileNotFoundError(f"Core file not found: {candidate}")
    specification = importlib.util.spec_from_file_location("xgat_drl_code", candidate)
    if specification is None or specification.loader is None:
        raise ImportError(f"Unable to load the core module from {candidate}")
    module = importlib.util.module_from_spec(specification)
    sys.modules["xgat_drl_code"] = module
    specification.loader.exec_module(module)
    return module


def _validate_core_api(module: ModuleType) -> dict[str, Any]:
    missing = [name for name in REQUIRED_CORE_API if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "The loaded xgat_drl_code.py is incompatible with simulations.py. "
            f"Missing symbols: {', '.join(missing)}"
        )
    module_file = Path(getattr(module, "__file__", "")).resolve()
    digest = hashlib.sha256(module_file.read_bytes()).hexdigest() if module_file.is_file() else None
    return {
        "module": module.__name__,
        "path": str(module_file),
        "sha256": digest,
        "required_symbols": list(REQUIRED_CORE_API),
        "missing_symbols": [],
    }


def _configure_torch_threads() -> None:
    count = max(1, int(os.getenv("XGAT_TORCH_THREADS", "1")))
    torch.set_num_threads(count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _configure_torch_native_bmm(device: torch.device) -> dict[str, Any]:
    """Avoid an unusable experimental CUDA ``bmm`` override.

    Recent PyTorch builds can route an ordinary batched matrix multiplication
    through a Triton native override.  Triton compiles a small Python extension
    on first use, which requires ``Python.h``.  When the interpreter development
    headers are absent, disable only that override and retain the regular ATen
    CUDA implementation.  Users can force either behaviour with
    ``XGAT_DISABLE_NATIVE_BMM=1`` or ``XGAT_DISABLE_NATIVE_BMM=0``.
    """
    mode = os.getenv("XGAT_DISABLE_NATIVE_BMM", "auto").strip().lower()
    truthy = {"1", "true", "yes", "on"}
    falsy = {"0", "false", "no", "off"}
    if mode not in {"auto", *truthy, *falsy}:
        raise ValueError(
            "XGAT_DISABLE_NATIVE_BMM must be 'auto', 1/0, true/false, yes/no, or on/off."
        )

    include_directory = sysconfig.get_path("include")
    python_header = (
        Path(include_directory) / "Python.h" if include_directory else None
    )
    header_available = bool(python_header is not None and python_header.is_file())
    forced = mode in truthy
    disabled_by_user = mode in falsy
    should_disable = bool(
        device.type == "cuda"
        and not disabled_by_user
        and (forced or (mode == "auto" and not header_available))
    )
    status: dict[str, Any] = {
        "device": device.type,
        "mode": mode,
        "python_header": str(python_header) if python_header is not None else None,
        "python_header_available": header_available,
        "native_bmm_disabled": False,
        "native_override_api_available": False,
    }
    if not should_disable:
        return status

    try:
        from torch._native.registry import deregister_op_overrides
    except (ImportError, AttributeError):
        # PyTorch releases without the private native-override registry use the
        # regular ATen path already and need no compatibility action.
        return status

    status["native_override_api_available"] = True
    try:
        deregister_op_overrides(disable_op_symbols="bmm")
    except (RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "Unable to disable PyTorch's native CUDA bmm override. Install the "
            "Python development headers, use --device cpu, or set "
            "XGAT_DISABLE_NATIVE_BMM=0 only after Python.h is available."
        ) from exc
    status["native_bmm_disabled"] = True
    return status


@dataclass(frozen=True)
class SimulationConfig:
    """Locked main-study simulation, training, and inference specification.

    Market seeds are the independent Monte Carlo units.  Policy seeds quantify
    optimisation variability within a market and are averaged before inference.
    """

    # Data-generating process and independent Monte Carlo design.
    n_samples: int = 10000
    n_risk_assets: int = 5
    train_fraction: float = 0.70
    validation_fraction: float = 0.15

    scenario: str = "graph_predictive"
    scenarios: tuple[str, ...] = (
        "covariance_only",
        "graph_predictive",
        "tail_stress",
    )
    # Fifty independent markets per scenario.
    market_seeds: tuple[int, ...] = (
        11, 23, 37, 53, 71, 89, 107, 131, 151, 173,
        197, 223, 251, 281, 313, 347, 373, 401, 433, 463,
        491, 523, 563, 593, 619, 653, 683, 719, 751, 787,
        809, 839, 863, 887, 919, 947, 977, 1009, 1039, 1069,
        1097, 1129, 1151, 1181, 1213, 1237, 1277, 1301, 1321, 1367,
    )
    # These runs are averaged within each market; they are not independent tests.
    policy_seeds: tuple[int, ...] = (101, 211, 307)

    # Regime identification.
    lookback: int = 60
    volatility_window: int = 20
    volatility_method: str = "ewma"
    hmm_ewma_halflife: float = 5.0
    hmm_tolerance: float = 1e-3
    hmm_minimum_covariance: float = 1e-4
    hmm_transition_prior: float = 1.50
    n_regimes: int = 2

    # Dynamic undirected risk graph.
    graph_threshold: float = 0.03
    graph_update_interval: int = 42
    graph_alpha_refresh_interval: int = 126
    graph_window: int = 252
    fast_graph_window: int = 63
    graph_min_history: int = 80
    fast_graph_min_history: int = 45
    graph_change_trigger: float = 0.20
    minimum_effective_samples: float = 25.0
    ebic_gamma: float = 0.50
    graph_bootstrap_replicates: int = 16
    rolling_graph_bootstrap_replicates: int = 8
    graph_bootstrap_block: int = 20
    graph_selection_probability: float = 0.60
    graph_maximum_degree: int = 4
    graph_maximum_density: float = 0.60
    graph_smoothing: float = 0.35
    fast_graph_weight_normal: float = 0.20
    fast_graph_weight_crisis: float = 0.80
    use_multiscale_graphs: bool = True

    # Dynamic directed predictive graph.
    predictive_lags: int = 3
    predictive_threshold: float = 0.01
    predictive_l1_ratio: float = 0.80
    predictive_selection_probability: float = 0.75
    predictive_minimum_stability: float = 0.70
    predictive_maximum_in_degree: int = 2
    predictive_maximum_density: float = 0.30
    predictive_bootstrap_replicates: int = 24
    predictive_report_bootstrap_replicates: int = 100
    predictive_smoothing: float = 0.35
    predictive_regime_shrinkage: float = 40.0
    predictive_null_replicates: int = 50
    predictive_report_null_replicates: int = 200
    predictive_fdr_level: float = 0.05
    predictive_null_quantile: float = 0.975

    # Trading frictions and portfolio feasibility.  The cash cap is shared by
    # all optimised comparators that can choose a cash allocation.
    transaction_cost: float = 0.0010
    slippage_coefficient: float = 0.00005
    impact_coefficient: float = 0.00020
    maximum_cash: float = 0.50
    normal_cash_maximum: float = 0.30
    crisis_cash_minimum: float = 0.20
    benchmark_cash_weight: float = 0.50
    normal_cash_target: float = 0.03
    crisis_cash_target: float = 0.25

    # CMDP constraints and reward specification.
    reward_scale: float = 1.0
    cvar_limit: float = 0.025
    cvar_alpha: float = 0.05
    drawdown_limit: float = 0.25
    turnover_limit: float = 0.04
    maximum_hhi: float = 0.50
    multiplier_learning_rate: float = 0.010
    active_cmdp_constraints: tuple[str, ...] = ("cvar", "drawdown")

    # X-GAT-DRL optimisation.
    encoder_learning_rate: float = 1e-4
    actor_learning_rate: float = 1e-4
    critic_learning_rate: float = 3e-4
    critic_updates_per_actor: int = 3
    encoder_freeze_episodes: int = 8
    ppo_episodes: int = 1000
    ppo_update_epochs: int = 6
    episode_length: int = 128
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.20  
    value_coefficient: float = 0.50
    cost_value_coefficient: float = 0.25
    entropy_coefficient: float = 0.002
    value_clip: float | None = 0.20
    quantile_coefficient: float = 0.05
    quantile_horizon: int = 20
    direct_utility_coefficient: float = 0.0  
    direct_downside_penalty: float = 0.10
    direct_concentration_penalty: float = 0.01
    expert_auxiliary_coefficient: float = 0.01  
    gate_auxiliary_coefficient: float = 0.01   
    max_gradient_norm: float = 0.50
    target_kl: float | None = 0.005
    initial_kl_coefficient: float = 0.02
    validation_interval: int = 5
    validation_patience: int = 8
    behaviour_epochs: int = 2
    behaviour_batches_per_epoch: int = 8
    stress_sampling_final: float = 0.50
    representation_pretrain_epochs: int = 8
    representation_batches_per_epoch: int = 12
    future_risk_horizon: int = 20
    use_active_reward: bool = True
    active_reward_weight: float = 1.0
    absolute_reward_weight: float = 0.25

    # Expert prior and controlled neural benchmarks.
    teacher_window: int = 60
    teacher_temperature: float = 0.15
    policy_base_method: str = "HMM-GMV"

    benchmark_epochs: int = 60
    benchmark_validation_patience: int = 8
    benchmark_minimum_epochs: int = 12
    benchmark_variance_penalty: float = 0.10
    benchmark_concentration_penalty: float = 0.02
    tcmac_information_coefficient: float = 0.05
    tcmac_value_coefficient: float = 0.25

    # Network capacity and model components.
    hidden_gru: int = 32
    hidden_gat: int = 32
    hidden_tcmac: int = 32
    dropout: float = 0.05

    use_benchmark_anchor: bool = False
    use_behaviour_cloning: bool = False
    use_cmdp: bool = True
    use_dynamic_cash: bool = True   
    use_risk_graph: bool = True
    use_predictive_graph: bool = True
    use_quantile_head: bool = True
    use_differentiable_optimizer: bool = True

    # Final inference.  
    periods_per_year: int = 252
    # Number of effectively independent model/configuration trials for DSR.
    dsr_effective_trials: float | None = None
    mcs_alpha: float = 0.10
    mcs_block_length: int = 20
    mcs_bootstraps: int = 5_000
    hierarchical_bootstraps: int = 20_000
    sign_flip_repetitions: int = 50_000
    confidence_level: float = 0.95

    output_dir: Path = Path("simulation_results_final")
    continue_on_error: bool = False

    @property
    def n_assets(self) -> int:
        return self.n_risk_assets + 1

    @property
    def train_end(self) -> int:
        return int(math.floor(self.n_samples * self.train_fraction))

    @property
    def fit_end(self) -> int:
        validation_size = max(20, int(math.floor(self.train_end * self.validation_fraction)))
        return self.train_end - validation_size

    @property
    def external_state_dim(self) -> int:
        return self.n_assets + 2 + self.n_regimes + 2

    def validate(self) -> None:
        if self.n_samples < 360:
            raise ValueError("n_samples must be at least 360.")
        if self.n_risk_assets < 2:
            raise ValueError("n_risk_assets must be at least 2.")
        if not 0.50 < self.train_fraction < 0.90:
            raise ValueError("train_fraction must lie in (0.50, 0.90).")
        if not 0.05 <= self.validation_fraction < 0.30:
            raise ValueError("validation_fraction must lie in [0.05, 0.30).")
        minimum_history = max(self.lookback, self.graph_min_history) + self.episode_length + 20
        if self.fit_end <= minimum_history:
            raise ValueError("The fitting sample is too short for the requested windows.")
        if not 10 <= self.fast_graph_window <= self.graph_window:
            raise ValueError("fast_graph_window must lie between 10 and graph_window.")
        if self.fast_graph_min_history > self.fast_graph_window:
            raise ValueError("fast_graph_min_history cannot exceed fast_graph_window.")
        if self.predictive_lags < 1 or self.predictive_lags >= self.graph_min_history // 2:
            raise ValueError("predictive_lags is incompatible with graph_min_history.")
        for value in (
            self.graph_smoothing,
            self.predictive_smoothing,
            self.fast_graph_weight_normal,
            self.fast_graph_weight_crisis,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Graph mixing coefficients must lie in [0, 1].")
        if self.fast_graph_weight_normal > self.fast_graph_weight_crisis:
            raise ValueError("Fast-graph weight must not decrease in crisis.")
        if not 0.0 < self.maximum_cash <= 1.0:
            raise ValueError("maximum_cash must lie in (0, 1].")
        if not 0.0 <= self.normal_cash_maximum <= self.maximum_cash:
            raise ValueError("normal_cash_maximum must lie in [0, maximum_cash].")
        if not 0.0 <= self.crisis_cash_minimum <= self.maximum_cash:
            raise ValueError("crisis_cash_minimum must lie in [0, maximum_cash].")
        if not 0.0 <= self.benchmark_cash_weight <= self.maximum_cash:
            raise ValueError("benchmark_cash_weight must lie in [0, maximum_cash].")
        if not 0.0 < self.cvar_alpha < 0.5:
            raise ValueError("cvar_alpha must lie in (0, 0.5).")
        if self.dsr_effective_trials is not None and (
            not np.isfinite(self.dsr_effective_trials) or self.dsr_effective_trials < 1.0
        ):
            raise ValueError("dsr_effective_trials must be finite and at least 1.")
        if min(
            self.encoder_learning_rate, self.actor_learning_rate,
            self.critic_learning_rate, self.multiplier_learning_rate,
        ) <= 0.0:
            raise ValueError("Learning rates must be positive.")
        if self.reward_scale <= 0.0:
            raise ValueError("reward_scale must be positive.")
        if not 0.0 < self.clip_epsilon < 1.0:
            raise ValueError("clip_epsilon must lie in (0, 1).")
        if self.target_kl is not None and self.target_kl <= 0.0:
            raise ValueError("target_kl must be positive when supplied.")
        if min(
            self.ppo_episodes,
            self.ppo_update_epochs,
            self.benchmark_epochs,
            self.batch_size,
            self.graph_update_interval,
            self.graph_alpha_refresh_interval,
            self.future_risk_horizon,
            self.critic_updates_per_actor,
            self.benchmark_validation_patience,
            self.benchmark_minimum_epochs,
        ) < 1:
            raise ValueError("Training counts and graph scheduling intervals must be positive.")
        if min(self.representation_pretrain_epochs, self.representation_batches_per_epoch) < 0:
            raise ValueError("Pretraining counts cannot be negative.")
        if (self.representation_pretrain_epochs == 0) != (self.representation_batches_per_epoch == 0):
            raise ValueError("Pretraining epochs and batches must both be zero or both be positive.")
        if min(
            self.graph_bootstrap_replicates,
            self.rolling_graph_bootstrap_replicates,
            self.predictive_bootstrap_replicates,
            self.predictive_report_bootstrap_replicates,
            self.predictive_null_replicates,
            self.predictive_report_null_replicates,
        ) < 0:
            raise ValueError("Bootstrap replicate counts cannot be negative.")
        if not 0.5 < self.predictive_null_quantile < 1.0:
            raise ValueError("predictive_null_quantile must lie in (0.5, 1).")
        for probability in (
            self.graph_selection_probability,
            self.predictive_selection_probability,
            self.predictive_minimum_stability,
            self.predictive_fdr_level,
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError("Graph probabilities must lie in [0, 1].")
        for density in (self.graph_maximum_density, self.predictive_maximum_density):
            if not 0.0 < density <= 1.0:
                raise ValueError("Graph density caps must lie in (0, 1].")
        valid_constraints = {"cvar", "drawdown", "turnover", "hhi"}
        if not self.active_cmdp_constraints or set(self.active_cmdp_constraints) - valid_constraints:
            raise ValueError("active_cmdp_constraints contains an invalid value.")
        if self.policy_base_method not in {"1/N", "GMV", "HMM-GMV"}:
            raise ValueError("policy_base_method must be 1/N, GMV, or HMM-GMV.")
        if not self.scenarios or not self.market_seeds or not self.policy_seeds:
            raise ValueError("Scenarios and seed collections cannot be empty.")
        if len(set(self.market_seeds)) != len(self.market_seeds):
            raise ValueError("market_seeds must be unique.")
        if len(set(self.policy_seeds)) != len(self.policy_seeds):
            raise ValueError("policy_seeds must be unique.")
        unknown = set(self.scenarios) - {"covariance_only", "graph_predictive", "tail_stress"}
        if unknown:
            raise ValueError(f"Unknown scenarios: {sorted(unknown)}")

@dataclass(frozen=True)
class MarketData:
    log_returns: np.ndarray
    true_regimes: np.ndarray
    parameters: dict[str, Any]


@dataclass(frozen=True)
class PreparedData:
    features: torch.Tensor
    feature_names: tuple[str, ...]
    hmm_observations: np.ndarray
    hmm_probabilities: np.ndarray
    hmm_model: Any
    hmm_diagnostics: dict[str, Any]
    state_order: np.ndarray
    hmm_start: int
    warmup: int
    fit_end: int
    reward_scale_factor: float
    dynamic_risk_adjacencies: torch.Tensor
    dynamic_predictive_adjacencies: torch.Tensor
    static_risk_adjacency: torch.Tensor
    static_predictive_adjacency: torch.Tensor
    regime_covariances: tuple[np.ndarray, ...]
    regime_effective_samples: np.ndarray
    report_risk_graphs: tuple[Any, ...]
    report_predictive_graphs: tuple[Any, ...]
    report_glasso: tuple[Any, ...]

@dataclass
class PortfolioState:
    weights: np.ndarray
    wealth: float = 1.0
    peak: float = 1.0
    drawdown: float = 0.0

    @property
    def peak_ratio(self) -> float:
        return float(self.wealth / max(self.peak, 1e-12))


@dataclass
class EvaluationResult:
    returns: dict[str, np.ndarray]
    turnover: dict[str, np.ndarray]
    costs: dict[str, np.ndarray]
    weights: dict[str, np.ndarray]
    hhi: dict[str, np.ndarray]
    cash_weights: dict[str, np.ndarray]


@dataclass
class ModelBundle:
    glasso_gat: Any
    tcmac: Any
    training_diagnostics: pd.DataFrame
    training_times: dict[str, float]


def _correlation_matrix(
    rng: np.random.Generator,
    dimension: int,
    base_correlation: float,
    noise_scale: float,
) -> np.ndarray:
    matrix = np.full((dimension, dimension), base_correlation, dtype=float)
    noise = rng.normal(0.0, noise_scale, size=(dimension, dimension))
    matrix += 0.5 * (noise + noise.T)
    np.fill_diagonal(matrix, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    matrix = (eigenvectors * np.clip(eigenvalues, 1e-4, None)) @ eigenvectors.T
    scale = np.sqrt(np.clip(np.diag(matrix), 1e-12, None))
    matrix /= np.outer(scale, scale)
    matrix = np.clip(matrix, -0.99, 0.99)
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _stable_lead_lag_matrix(correlation: np.ndarray, spectral_radius: float) -> np.ndarray:
    """Create an identifiable sparse directed spillover network.

    A directed ring guarantees a stable recurrent transmission channel. One
    additional high-correlation cross-edge prevents the graph from being a
    trivial chain while keeping density and in-degree low.
    """
    association = np.abs(np.asarray(correlation, dtype=float))
    n_assets = association.shape[0]
    matrix = np.zeros_like(association)
    for source in range(n_assets):
        target = (source + 1) % n_assets
        matrix[target, source] = 0.75 + 0.25 * association[target, source]

    candidates: list[tuple[float, int, int]] = []
    for target in range(n_assets):
        for source in range(n_assets):
            if source == target or target == (source + 1) % n_assets:
                continue
            candidates.append((association[target, source], target, source))
    if candidates:
        _, target, source = max(candidates, key=lambda item: item[0])
        matrix[target, source] = 0.60 + 0.40 * association[target, source]

    radius = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if radius <= 1e-12:
        raise ValueError("The directed spillover template is degenerate.")
    matrix *= float(spectral_radius) / radius
    return matrix


def generate_synthetic_market(config: SimulationConfig, seed: int) -> MarketData:
    """Generate negative-control, graph-predictive, and tail-stress markets."""
    rng = np.random.default_rng(seed)
    n_risk = config.n_risk_assets
    normal_mean = rng.uniform(0.00015, 0.00065, size=n_risk)
    crisis_mean = rng.uniform(-0.0022, -0.0006, size=n_risk)
    normal_volatility = rng.uniform(0.006, 0.013, size=n_risk)
    crisis_volatility = rng.uniform(0.018, 0.036, size=n_risk)
    normal_correlation = _correlation_matrix(rng, n_risk, 0.18, 0.05)
    crisis_correlation = _correlation_matrix(rng, n_risk, 0.68, 0.07)
    normal_covariance = np.outer(normal_volatility, normal_volatility) * normal_correlation
    crisis_covariance = np.outer(crisis_volatility, crisis_volatility) * crisis_correlation
    transition = np.array([[0.985, 0.015], [0.10, 0.90]], dtype=float)

    if config.scenario == "covariance_only":
        lead_lag = (np.zeros((n_risk, n_risk)), np.zeros((n_risk, n_risk)))
        momentum_coefficients = (0.0, 0.0)
        jump_probabilities = (0.003, 0.015)
    else:
        lead_lag = (
            _stable_lead_lag_matrix(normal_correlation, 0.18),
            _stable_lead_lag_matrix(crisis_correlation, 0.32),
        )
        momentum_coefficients = (0.06, -0.04)
        crisis_mean[-1] = rng.uniform(-0.00030, 0.00020)
        jump_probabilities = (
            (0.004, 0.025)
            if config.scenario == "tail_stress"
            else (0.003, 0.018)
        )

    degrees_of_freedom = 5.0 if config.scenario == "tail_stress" else 7.0
    risk_returns = np.zeros((config.n_samples, n_risk), dtype=float)
    regimes = np.empty(config.n_samples, dtype=int)
    regime = 0
    for t in range(config.n_samples):
        regimes[t] = regime
        covariance = (normal_covariance, crisis_covariance)[regime]
        gaussian = rng.multivariate_normal(np.zeros(n_risk), covariance, check_valid="raise")
        scale = math.sqrt(degrees_of_freedom / rng.chisquare(degrees_of_freedom))
        innovation = gaussian * scale
        if t == 0:
            spillover = np.zeros(n_risk, dtype=float)
            momentum = np.zeros(n_risk, dtype=float)
        else:
            spillover = lead_lag[regime] @ risk_returns[t - 1]
            momentum = risk_returns[max(0, t - 5) : t].mean(axis=0)
        if rng.random() < jump_probabilities[regime]:
            jump_location = -0.018 if regime == 1 else -0.006
            jump_scale = 0.012 if regime == 1 else 0.005
            innovation += rng.normal(jump_location, jump_scale, size=n_risk)
        risk_returns[t] = (
            (normal_mean, crisis_mean)[regime]
            + spillover
            + momentum_coefficients[regime] * momentum
            + innovation
        )
        regime = int(rng.choice(2, p=transition[regime]))

    cash_mean = np.where(regimes == 1, 0.00012, 0.00007)
    cash_returns = cash_mean + rng.normal(0.0, 0.000015, size=config.n_samples)
    return MarketData(
        log_returns=np.column_stack([risk_returns, cash_returns]),
        true_regimes=regimes,
        parameters={
            "scenario": config.scenario,
            "normal_mean": normal_mean.tolist(),
            "crisis_mean": crisis_mean.tolist(),
            "normal_volatility": normal_volatility.tolist(),
            "crisis_volatility": crisis_volatility.tolist(),
            "normal_correlation": normal_correlation.tolist(),
            "crisis_correlation": crisis_correlation.tolist(),
            "normal_lead_lag": lead_lag[0].tolist(),
            "crisis_lead_lag": lead_lag[1].tolist(),
            "transition_matrix": transition.tolist(),
            "student_t_degrees_of_freedom": degrees_of_freedom,
        },
    )


def _ordered_hmm_probabilities(
    model: Any,
    probabilities: np.ndarray,
    feature_index: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(model.means_, dtype=float)
    if means.ndim != 2 or not 0 <= feature_index < means.shape[1]:
        raise ValueError("Invalid HMM means or ordering feature.")
    order = np.argsort(means[:, feature_index], kind="stable")
    ordered = np.asarray(probabilities, dtype=float)[:, order]
    ordered = np.clip(ordered, 0.0, None)
    ordered /= ordered.sum(axis=1, keepdims=True)
    return ordered, order



def _signed_adjacency(graph: Any) -> np.ndarray:
    """Return confidence-weighted signed edges without a second hard mask."""
    signed = np.asarray(graph.signed_partial_correlation, dtype=float).copy()
    stability = np.asarray(getattr(graph, "stability", np.ones_like(signed)), dtype=float)
    signed = np.where(stability > 0.0, signed, 0.0)
    np.fill_diagonal(signed, 1.0)
    return signed

def _fit_regime_graphs(
    core: ModuleType,
    returns: np.ndarray,
    probabilities: np.ndarray,
    config: SimulationConfig,
    seed: int,
    *,
    fixed_alphas: Sequence[float] | None = None,
) -> tuple[
    tuple[Any, ...],
    tuple[Any, ...],
    np.ndarray,
    tuple[np.ndarray, ...],
    tuple[float, ...],
]:
    """Fit regime risk graphs, optionally reusing causally selected penalties."""
    if fixed_alphas is not None and len(fixed_alphas) != config.n_regimes:
        raise ValueError("fixed_alphas must match the number of regimes.")

    fallback = core.estimate_covariance(returns, method="ledoit_wolf")
    graphs: list[Any] = []
    glasso_results: list[Any] = []
    effective_samples: list[float] = []
    covariances: list[np.ndarray] = []
    selected_alphas: list[float] = []
    for state in range(config.n_regimes):
        covariance, effective = core.compute_regime_weighted_covariance(
            returns,
            probabilities[:, state],
            minimum_effective_samples=config.minimum_effective_samples,
            fallback_covariance=fallback,
        )
        requested_alpha = None if fixed_alphas is None else float(fixed_alphas[state])
        glasso, graph, effective_selected = core.stability_selected_glasso_graph(
            returns,
            probabilities[:, state],
            minimum_effective_samples=config.minimum_effective_samples,
            ebic_gamma=config.ebic_gamma,
            threshold=config.graph_threshold,
            bootstrap_replicates=config.rolling_graph_bootstrap_replicates,
            block_length=config.graph_bootstrap_block,
            selection_probability=config.graph_selection_probability,
            maximum_degree=config.graph_maximum_degree,
            maximum_density=config.graph_maximum_density,
            random_state=seed + 1009 * state,
            alpha=requested_alpha,
        )
        covariances.append(covariance)
        effective_samples.append(min(effective, effective_selected))
        glasso_results.append(glasso)
        graphs.append(graph)
        selected_alphas.append(float(glasso.alpha))
    return (
        tuple(graphs),
        tuple(glasso_results),
        np.asarray(effective_samples),
        tuple(covariances),
        tuple(selected_alphas),
    )



def _fit_regime_predictive_graphs(
    core: ModuleType,
    returns: np.ndarray,
    probabilities: np.ndarray,
    config: SimulationConfig,
    seed: int,
    *,
    report: bool = False,
) -> tuple[Any, ...]:
    """Estimate lag-specific predictive graphs with stability and pooled shrinkage."""
    if probabilities.shape != (returns.shape[0], config.n_regimes):
        raise ValueError("returns and probabilities have incompatible shapes.")
    replicates = config.predictive_report_bootstrap_replicates if report else config.predictive_bootstrap_replicates
    null_replicates = (
        config.predictive_report_null_replicates if report
        else config.predictive_null_replicates
    )
    common_kwargs = dict(
        maximum_lag=config.predictive_lags,
        l1_ratio=config.predictive_l1_ratio,
        ebic_gamma=config.ebic_gamma,
        coefficient_threshold=config.predictive_threshold,
        bootstrap_replicates=replicates,
        block_length=config.graph_bootstrap_block,
        selection_probability=config.predictive_selection_probability,
        maximum_in_degree=config.predictive_maximum_in_degree,
        maximum_density=config.predictive_maximum_density,
        soft_selection=False,
        minimum_stability_weight=config.predictive_minimum_stability,
        null_replicates=null_replicates,
        false_discovery_rate=config.predictive_fdr_level,
        null_quantile=config.predictive_null_quantile,
    )
    pooled = core.stability_selected_predictive_graph(
        returns, None, random_state=seed + 7919, **common_kwargs
    )
    graphs: list[Any] = []
    for state in range(config.n_regimes):
        estimated = core.stability_selected_predictive_graph(
            returns, probabilities[:, state], random_state=seed + 1009 * state, **common_kwargs
        )
        effective = max(float(estimated.effective_samples), 0.0)
        state_weight = effective / (effective + max(config.predictive_regime_shrinkage, 1e-8))
        state_lagged = np.asarray(estimated.lagged_signed_coefficients, dtype=float)
        pooled_lagged = np.asarray(pooled.lagged_signed_coefficients, dtype=float)
        lagged_signed = state_weight * state_lagged + (1.0 - state_weight) * pooled_lagged
        state_stability = np.asarray(estimated.lagged_stability, dtype=float)
        pooled_stability = np.asarray(pooled.lagged_stability, dtype=float)
        lagged_stability = state_weight * state_stability + (1.0 - state_weight) * pooled_stability
        state_selected = np.asarray(estimated.lagged_adjacency, dtype=float)
        pooled_selected = np.asarray(pooled.lagged_adjacency, dtype=float)
        selected_blend = state_weight * state_selected + (1.0 - state_weight) * pooled_selected
        lagged_adjacency = []
        for lag in range(config.predictive_lags):
            lagged_adjacency.append(
                core.sparsify_signed_graph(
                    selected_blend[lag],
                    threshold=1e-12,
                    maximum_degree=config.predictive_maximum_in_degree,
                    maximum_density=config.predictive_maximum_density,
                    directed=True,
                    include_self_loops=True,
                )
            )
        lagged_adjacency = np.stack(lagged_adjacency)
        discount = 1.0 / np.arange(1, config.predictive_lags + 1, dtype=float)
        signed = np.einsum("ltk,l->tk", lagged_signed, discount)
        aggregate = np.einsum("ltk,l->tk", lagged_adjacency, discount)
        adjacency = core.sparsify_signed_graph(
            aggregate,
            threshold=1e-12,
            maximum_degree=config.predictive_maximum_in_degree,
            maximum_density=config.predictive_maximum_density,
            directed=True,
            include_self_loops=True,
        )
        graphs.append(
            type(pooled)(
                signed_coefficients=signed,
                adjacency=adjacency,
                edge_mask=np.abs(adjacency) > 1e-12,
                stability=np.max(lagged_stability, axis=0),
                selected_alphas=np.asarray(estimated.selected_alphas, dtype=float),
                effective_samples=effective,
                detection_threshold=float(estimated.detection_threshold),
                lagged_signed_coefficients=lagged_signed,
                lagged_adjacency=lagged_adjacency,
                lagged_stability=lagged_stability,
                lagged_p_values=(
                    state_weight * np.asarray(estimated.lagged_p_values, dtype=float)
                    + (1.0 - state_weight) * np.asarray(pooled.lagged_p_values, dtype=float)
                ),
                lagged_null_thresholds=(
                    state_weight * np.asarray(estimated.lagged_null_thresholds, dtype=float)
                    + (1.0 - state_weight) * np.asarray(pooled.lagged_null_thresholds, dtype=float)
                ),
            )
        )
    return tuple(graphs)


def prepare_data(
    core: ModuleType,
    market: MarketData,
    config: SimulationConfig,
    seed: int,
    device: torch.device,
) -> PreparedData:
    """Build causal features, filtered regimes, and confidence-weighted graph paths."""
    raw_features, feature_names = core.build_causal_feature_tensor(
        market.log_returns,
        window=config.lookback,
        include_volatility=True,
        include_momentum=True,
    )
    observations, hmm_start = core.build_hmm_observations(
        market.log_returns[:, : config.n_risk_assets],
        volatility_window=config.volatility_window,
        use_log_volatility=True,
        volatility_method=config.volatility_method,
        ewma_halflife=config.hmm_ewma_halflife,
    )
    warmup = max(config.lookback, hmm_start + 1)
    if config.fit_end <= warmup + config.graph_min_history:
        raise ValueError("Insufficient fitting history after causal warm-up.")

    scaled = core.robust_scale_tensor(raw_features, raw_features[warmup : config.fit_end])
    scaled[:warmup] = 0.0
    features = torch.as_tensor(scaled, dtype=torch.float32, device=device)

    hmm_model = core.fit_hmm_with_restarts(
        observations[hmm_start : config.fit_end],
        n_components=config.n_regimes,
        seeds=(seed, seed + 101, seed + 307, seed + 701, seed + 1009),
        covariance_type="full",
        tolerance=config.hmm_tolerance,
        minimum_covariance=config.hmm_minimum_covariance,
        transition_prior=config.hmm_transition_prior,
    )
    hmm_diagnostics = dict(getattr(hmm_model, "_xgat_hmm_diagnostics", {}))
    if not bool(hmm_diagnostics.get("converged", False)):
        raise RuntimeError("The selected HMM did not meet the convergence check.")
    probabilities = core.filtered_hmm_probabilities(
        hmm_model,
        observations,
        start_index=hmm_start,
        initial_fill="start",
    )
    probabilities, state_order = _ordered_hmm_probabilities(hmm_model, probabilities)

    truth = np.asarray(market.true_regimes, dtype=int)
    predicted = np.argmax(probabilities, axis=1)
    valid = np.arange(max(hmm_start, warmup), config.n_samples)
    accuracy = float(np.mean(predicted[valid] == truth[valid]))
    recalls = []
    confusion = np.zeros((config.n_regimes, config.n_regimes), dtype=int)
    for actual in range(config.n_regimes):
        mask = truth[valid] == actual
        recalls.append(float(np.mean(predicted[valid][mask] == actual)) if mask.any() else np.nan)
        for estimated in range(config.n_regimes):
            confusion[actual, estimated] = int(np.sum(mask & (predicted[valid] == estimated)))
    one_hot = np.eye(config.n_regimes)[truth[valid]]
    hmm_diagnostics.update(
        {
            "state_accuracy": accuracy,
            "balanced_accuracy": float(np.nanmean(recalls)),
            "brier_score": float(np.mean(np.sum((probabilities[valid] - one_hot) ** 2, axis=1))),
            "confusion_matrix": confusion.tolist(),
        }
    )

    equal_risk = np.full(config.n_risk_assets, 1.0 / config.n_risk_assets)
    training_risk = market.log_returns[warmup : config.fit_end, : config.n_risk_assets]
    equal_growth = np.exp(training_risk) @ equal_risk
    equal_returns = np.log(np.clip(equal_growth, 1e-12, None))
    training_scale = float(np.std(equal_returns, ddof=1))
    reward_scale_factor = 1.0 / max(training_scale, 1e-4)

    identity = np.eye(config.n_risk_assets, dtype=np.float32)
    risk_path = np.repeat(identity[None, :, :], config.n_samples, axis=0)
    predictive_path = np.repeat(
        identity[None, None, :, :], config.n_samples * config.predictive_lags, axis=0
    ).reshape(config.n_samples, config.predictive_lags, config.n_risk_assets, config.n_risk_assets)
    latest_risk: list[np.ndarray] | None = None
    latest_predictive: list[np.ndarray] | None = None
    latest_risk_alphas: tuple[float, ...] | None = None
    last_alpha_refresh: int | None = None
    last_update_probability: float | None = None
    report_risk_graphs: tuple[Any, ...] | None = None
    report_predictive_graphs: tuple[Any, ...] | None = None
    report_glasso: tuple[Any, ...] | None = None
    report_effective: np.ndarray | None = None
    report_covariances: tuple[np.ndarray, ...] | None = None

    first_update = warmup + config.graph_min_history
    fast_weights = np.linspace(
        config.fast_graph_weight_normal,
        config.fast_graph_weight_crisis,
        config.n_regimes,
    )
    for decision_time in range(first_update, config.n_samples):
        current_crisis_probability = float(probabilities[max(hmm_start, decision_time - 1), -1])
        periodic = (decision_time - first_update) % config.graph_update_interval == 0
        changed = (
            last_update_probability is not None
            and abs(current_crisis_probability - last_update_probability) >= config.graph_change_trigger
        )
        update = latest_risk is None or periodic or changed
        if update:
            slow_start = max(warmup, decision_time - config.graph_window)
            slow_returns = market.log_returns[slow_start:decision_time, : config.n_risk_assets]
            slow_probabilities = probabilities[slow_start:decision_time]
            refresh_penalties = (
                latest_risk_alphas is None
                or last_alpha_refresh is None
                or decision_time - last_alpha_refresh >= config.graph_alpha_refresh_interval
            )
            fixed_alphas = None if refresh_penalties else latest_risk_alphas
            slow_risk, slow_glasso, slow_effective, slow_covariances, selected_alphas = _fit_regime_graphs(
                core,
                slow_returns,
                slow_probabilities,
                config,
                seed + decision_time,
                fixed_alphas=fixed_alphas,
            )
            latest_risk_alphas = selected_alphas
            if refresh_penalties:
                last_alpha_refresh = decision_time
            slow_predictive = _fit_regime_predictive_graphs(
                core,
                slow_returns,
                slow_probabilities,
                config,
                seed + decision_time,
                report=False,
            )

            if config.use_multiscale_graphs:
                fast_start = max(warmup, decision_time - config.fast_graph_window)
                if decision_time - fast_start >= config.fast_graph_min_history:
                    fast_returns = market.log_returns[fast_start:decision_time, : config.n_risk_assets]
                    fast_probabilities = probabilities[fast_start:decision_time]
                    fast_risk, _, _, _, _ = _fit_regime_graphs(
                        core,
                        fast_returns,
                        fast_probabilities,
                        config,
                        seed + decision_time + 1543,
                        fixed_alphas=selected_alphas,
                    )
                    fast_predictive = _fit_regime_predictive_graphs(
                        core,
                        fast_returns,
                        fast_probabilities,
                        config,
                        seed + decision_time + 3253,
                        report=False,
                    )
                else:
                    fast_risk, fast_predictive = slow_risk, slow_predictive
            else:
                fast_risk, fast_predictive = slow_risk, slow_predictive

            new_risk = []
            new_predictive = []
            for state in range(config.n_regimes):
                fast_weight = float(fast_weights[state]) if config.use_multiscale_graphs else 0.0
                risk_value = (1.0 - fast_weight) * _signed_adjacency(slow_risk[state])
                risk_value += fast_weight * _signed_adjacency(fast_risk[state])
                predictive_value = (1.0 - fast_weight) * np.asarray(
                    slow_predictive[state].lagged_adjacency, dtype=float
                )
                predictive_value += fast_weight * np.asarray(
                    fast_predictive[state].lagged_adjacency, dtype=float
                )
                new_risk.append(risk_value)
                new_predictive.append(predictive_value)
            if latest_risk is not None:
                new_risk = [
                    config.graph_smoothing * old + (1.0 - config.graph_smoothing) * new
                    for old, new in zip(latest_risk, new_risk)
                ]
                new_predictive = [
                    config.predictive_smoothing * old + (1.0 - config.predictive_smoothing) * new
                    for old, new in zip(latest_predictive or new_predictive, new_predictive)
                ]
            latest_risk = [
                core.sparsify_signed_graph(
                    graph,
                    threshold=config.graph_threshold,
                    maximum_degree=config.graph_maximum_degree,
                    maximum_density=config.graph_maximum_density,
                    directed=False,
                    include_self_loops=True,
                )
                for graph in new_risk
            ]
            latest_predictive = [
                np.stack(
                    [
                        core.sparsify_signed_graph(
                            graph[lag],
                            threshold=config.predictive_threshold,
                            maximum_degree=config.predictive_maximum_in_degree,
                            maximum_density=config.predictive_maximum_density,
                            directed=True,
                            include_self_loops=True,
                        )
                        for lag in range(config.predictive_lags)
                    ],
                    axis=0,
                )
                for graph in new_predictive
            ]
            last_update_probability = current_crisis_probability
            if decision_time <= config.fit_end:
                report_risk_graphs = slow_risk
                report_predictive_graphs = slow_predictive
                report_glasso = slow_glasso
                report_effective = slow_effective
                report_covariances = slow_covariances

        if latest_risk is not None and latest_predictive is not None:
            decision_probability = probabilities[max(hmm_start, decision_time - 1)]
            mixed_risk = np.tensordot(decision_probability, np.stack(latest_risk), axes=(0, 0))
            mixed_predictive = np.tensordot(decision_probability, np.stack(latest_predictive), axes=(0, 0))
            risk_path[decision_time] = core.sparsify_signed_graph(
                mixed_risk,
                threshold=config.graph_threshold,
                maximum_degree=config.graph_maximum_degree,
                maximum_density=config.graph_maximum_density,
                directed=False,
                include_self_loops=True,
            ).astype(np.float32)
            predictive_path[decision_time] = np.stack(
                [
                    core.sparsify_signed_graph(
                        mixed_predictive[lag],
                        threshold=config.predictive_threshold,
                        maximum_degree=config.predictive_maximum_in_degree,
                        maximum_density=config.predictive_maximum_density,
                        directed=True,
                        include_self_loops=True,
                    )
                    for lag in range(config.predictive_lags)
                ],
                axis=0,
            ).astype(np.float32)

    training_returns = market.log_returns[warmup : config.fit_end, : config.n_risk_assets]
    training_probabilities = probabilities[warmup : config.fit_end]
    robust_predictive_report = _fit_regime_predictive_graphs(
        core,
        training_returns,
        training_probabilities,
        config,
        seed + 7919,
        report=True,
    )
    if (
        report_risk_graphs is None
        or report_predictive_graphs is None
        or report_glasso is None
        or report_effective is None
        or report_covariances is None
    ):
        report_risk_graphs, report_glasso, report_effective, report_covariances, _ = _fit_regime_graphs(
            core, training_returns, training_probabilities, config, seed
        )
    report_predictive_graphs = robust_predictive_report

    _, static_risk_graph, _ = core.stability_selected_glasso_graph(
        training_returns,
        None,
        ebic_gamma=config.ebic_gamma,
        threshold=config.graph_threshold,
        bootstrap_replicates=config.graph_bootstrap_replicates,
        block_length=config.graph_bootstrap_block,
        selection_probability=config.graph_selection_probability,
        maximum_degree=config.graph_maximum_degree,
        maximum_density=config.graph_maximum_density,
        random_state=seed + 17,
    )
    static_predictive_graph = core.stability_selected_predictive_graph(
        training_returns,
        None,
        maximum_lag=config.predictive_lags,
        l1_ratio=config.predictive_l1_ratio,
        ebic_gamma=config.ebic_gamma,
        coefficient_threshold=config.predictive_threshold,
        bootstrap_replicates=config.predictive_report_bootstrap_replicates,
        block_length=config.graph_bootstrap_block,
        selection_probability=config.predictive_selection_probability,
        maximum_in_degree=config.predictive_maximum_in_degree,
        maximum_density=config.predictive_maximum_density,
        random_state=seed + 19,
        soft_selection=False,
        minimum_stability_weight=config.predictive_minimum_stability,
        null_replicates=config.predictive_report_null_replicates,
        false_discovery_rate=config.predictive_fdr_level,
        null_quantile=config.predictive_null_quantile,
    )
    return PreparedData(
        features=features,
        feature_names=tuple(feature_names),
        hmm_observations=observations,
        hmm_probabilities=probabilities,
        hmm_model=hmm_model,
        hmm_diagnostics=hmm_diagnostics,
        state_order=state_order,
        hmm_start=hmm_start,
        warmup=warmup,
        fit_end=config.fit_end,
        reward_scale_factor=reward_scale_factor,
        dynamic_risk_adjacencies=torch.as_tensor(risk_path, dtype=torch.float32, device=device),
        dynamic_predictive_adjacencies=torch.as_tensor(predictive_path, dtype=torch.float32, device=device),
        static_risk_adjacency=torch.as_tensor(_signed_adjacency(static_risk_graph), dtype=torch.float32, device=device).unsqueeze(0),
        static_predictive_adjacency=torch.as_tensor(
            static_predictive_graph.lagged_adjacency,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0),
        regime_covariances=report_covariances,
        regime_effective_samples=report_effective,
        report_risk_graphs=report_risk_graphs,
        report_predictive_graphs=report_predictive_graphs,
        report_glasso=report_glasso,
    )

def _constraints(core: ModuleType, *, max_cash: float) -> Any:
    return core.PortfolioConstraints(
        max_cash=float(max_cash),
        minimum_weight=0.0,
        maximum_risk_weight=1.0,
        tolerance=1e-6,
    )


def _initial_weights(config: SimulationConfig, cash_weight: float = 0.05) -> np.ndarray:
    cash = float(np.clip(cash_weight, 0.0, config.maximum_cash))
    risk = np.full(config.n_risk_assets, (1.0 - cash) / config.n_risk_assets)
    return np.concatenate([risk, [cash]])


def _external_state(
    state: PortfolioState,
    regime_probability: np.ndarray,
    previous_probability: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    probability = np.asarray(regime_probability, dtype=float)
    previous = np.asarray(previous_probability, dtype=float)
    entropy_denominator = math.log(max(2, probability.size))
    entropy = float(
        -np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)))
        / entropy_denominator
    )
    vector = np.concatenate(
        [
            state.weights,
            [state.drawdown, state.peak_ratio],
            probability,
            [float(probability[-1] - previous[-1]), entropy],
        ]
    )
    return torch.as_tensor(vector, dtype=torch.float32, device=device).unsqueeze(0)


def _apply_action(
    state: PortfolioState,
    target_weights: np.ndarray,
    asset_log_returns: np.ndarray,
    config: SimulationConfig,
) -> tuple[PortfolioState, dict[str, float]]:
    target = np.asarray(target_weights, dtype=float).reshape(-1)
    if target.size != config.n_assets:
        raise ValueError(
            f"Expected {config.n_assets} target weights; received {target.size}."
        )
    if not np.all(np.isfinite(target)):
        raise ValueError("Target weights contain non-finite values.")
    if np.any(target < -1e-8):
        raise ValueError("Target weights contain a materially negative allocation.")
    target = np.clip(target, 0.0, None)
    target_total = float(target.sum())
    if target_total <= 1e-12:
        raise ValueError("Target weights do not have positive mass.")
    target /= target_total
    if target[-1] > config.maximum_cash + 1e-6:
        raise ValueError(
            "Target cash allocation exceeds maximum_cash after normalisation: "
            f"{target[-1]:.8f} > {config.maximum_cash:.8f}."
        )

    current = np.asarray(state.weights, dtype=float).reshape(-1)
    if current.size != config.n_assets or not np.all(np.isfinite(current)):
        raise ValueError("PortfolioState contains invalid current weights.")
    if np.any(current < -1e-8):
        raise ValueError("PortfolioState contains a materially negative weight.")
    current = np.clip(current, 0.0, None)
    current_total = float(current.sum())
    if current_total <= 1e-12:
        raise ValueError("PortfolioState current weights do not have positive mass.")
    current /= current_total

    asset_returns = np.asarray(asset_log_returns, dtype=float).reshape(-1)
    if asset_returns.size != config.n_assets:
        raise ValueError(
            f"Expected {config.n_assets} asset returns; received {asset_returns.size}."
        )
    if not np.all(np.isfinite(asset_returns)):
        raise ValueError("Asset log returns contain non-finite values.")

    turnover = 0.5 * float(np.abs(target - current).sum())
    execution_cost = (
        config.transaction_cost * turnover
        + config.slippage_coefficient * math.sqrt(max(turnover, 0.0))
        + config.impact_coefficient * turnover * turnover
    )
    asset_growth = np.exp(asset_returns)
    if not np.all(np.isfinite(asset_growth)):
        raise FloatingPointError("Asset growth overflowed or became non-finite.")
    gross_growth = max(float(target @ asset_growth), 1e-12)
    gross_log_return = float(math.log(gross_growth))
    net_log_return = gross_log_return - execution_cost
    wealth = state.wealth * math.exp(net_log_return)
    peak = max(state.peak, wealth)
    drawdown = wealth / peak - 1.0
    hhi = float(np.sum(target**2))
    drifted_weights = np.clip(target * asset_growth / gross_growth, 0.0, None)
    drifted_total = float(drifted_weights.sum())
    if not np.isfinite(drifted_total) or drifted_total <= 1e-12:
        raise FloatingPointError("Post-return portfolio weights are invalid.")
    drifted_weights /= drifted_total
    return (
        PortfolioState(drifted_weights, wealth=wealth, peak=peak, drawdown=drawdown),
        {
            "gross_log_return": gross_log_return,
            "log_return": net_log_return,
            "turnover": turnover,
            "transaction_cost": execution_cost,
            "hhi": hhi,
            "cash_weight": float(target[-1]),
            "drawdown": drawdown,
        },
    )


def _safe_weights(
    core: ModuleType,
    action: np.ndarray,
    config: SimulationConfig,
    *,
    max_cash: float | None = None,
) -> np.ndarray:
    maximum = config.maximum_cash if max_cash is None else float(max_cash)
    constraints = _constraints(core, max_cash=maximum)
    vector = np.asarray(action, dtype=float).reshape(-1)
    if vector.size != config.n_assets:
        raise ValueError(
            f"Expected {config.n_assets} action weights; received {vector.size}."
        )
    if not np.all(np.isfinite(vector)):
        raise ValueError("Action weights contain non-finite values.")
    try:
        return core.validate_weights(vector, constraints=constraints, expected_assets=config.n_assets)
    except ValueError:
        projected = core.project_long_only_weights(vector, constraints=constraints)
        return core.validate_weights(
            projected,
            constraints=constraints,
            expected_assets=config.n_assets,
        )


def _enforce_cash_interval(
    weights: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    vector = np.asarray(weights, dtype=float).copy()
    target_cash = float(np.clip(vector[-1], lower, upper))
    risk = np.clip(vector[:-1], 0.0, None)
    if risk.sum() <= 1e-12:
        risk = np.full(risk.size, 1.0 / risk.size)
    else:
        risk /= risk.sum()
    return np.concatenate([risk * (1.0 - target_cash), [target_cash]])


def _empirical_cvar(values: Sequence[float], level: float = 0.05) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    threshold = float(np.quantile(array, level))
    tail = array[array <= threshold]
    return float(np.mean(tail)) if tail.size else threshold


def _hac_standard_error(values: np.ndarray) -> float:
    """Newey-West standard error of the mean for a univariate return path."""
    observations = np.asarray(values, dtype=float)
    n_observations = observations.size
    if n_observations < 2:
        return float("inf")
    centred = observations - observations.mean()
    maximum_lag = min(
        n_observations - 1,
        max(1, int(math.floor(4.0 * (n_observations / 100.0) ** (2.0 / 9.0)))),
    )
    long_run_variance = float(np.dot(centred, centred) / n_observations)
    for lag in range(1, maximum_lag + 1):
        covariance = float(np.dot(centred[lag:], centred[:-lag]) / n_observations)
        long_run_variance += 2.0 * (1.0 - lag / (maximum_lag + 1.0)) * covariance
    return math.sqrt(max(long_run_variance, 0.0) / n_observations)


def _portfolio_metrics(
    log_returns: Sequence[float],
    turnover: Sequence[float],
    hhi: Sequence[float],
    periods: int,
) -> dict[str, float]:
    values = np.asarray(log_returns, dtype=float)
    turns = np.asarray(turnover, dtype=float)
    concentrations = np.asarray(hhi, dtype=float)
    if values.size < 2:
        return {
            "return": -np.inf,
            "return_lcb": -np.inf,
            "sharpe": -np.inf,
            "cvar": -np.inf,
            "drawdown": -1.0,
            "turnover": float(np.mean(turns)) if turns.size else 0.0,
            "hhi": float(np.mean(concentrations)) if concentrations.size else 1.0,
        }
    mean = float(values.mean())
    volatility = float(values.std(ddof=1))
    wealth = np.concatenate([[1.0], np.exp(np.cumsum(values))])
    peaks = np.maximum.accumulate(wealth)
    drawdown = float(np.min(wealth / peaks - 1.0))
    mean_standard_error = _hac_standard_error(values)
    conservative_mean = mean - 1.645 * mean_standard_error
    return {
        "return": math.exp(mean * periods) - 1.0,
        "return_lcb": math.exp(conservative_mean * periods) - 1.0,
        "sharpe": mean / volatility * math.sqrt(periods) if volatility > 1e-12 else 0.0,
        "cvar": _empirical_cvar(values),
        "drawdown": drawdown,
        "turnover": float(np.mean(turns)) if turns.size else 0.0,
        "hhi": float(np.mean(concentrations)) if concentrations.size else 1.0,
    }


def _validation_rank(
    metrics: Mapping[str, float],
    config: SimulationConfig,
) -> tuple[int, float, float, float]:
    """Lexicographic checkpoint rule using a HAC return lower bound."""
    violations = np.asarray(
        [
            max(abs(min(float(metrics["cvar"]), 0.0)) - config.cvar_limit, 0.0)
            / max(config.cvar_limit, 1e-8),
            max(abs(min(float(metrics["drawdown"]), 0.0)) - config.drawdown_limit, 0.0)
            / max(config.drawdown_limit, 1e-8),
            max(float(metrics["turnover"]) - config.turnover_limit, 0.0)
            / max(config.turnover_limit, 1e-8),
            max(float(metrics["hhi"]) - config.maximum_hhi, 0.0)
            / max(config.maximum_hhi, 1e-8),
        ],
        dtype=float,
    )
    total_violation = float(violations.sum())
    feasible = int(total_violation <= 1e-12)
    return (
        feasible,
        -total_violation,
        float(metrics["return_lcb"]),
        float(metrics["sharpe"]),
    )


def build_classical_benchmarks(
    core: ModuleType,
    market: MarketData,
    prepared: PreparedData,
    config: SimulationConfig,
    seed: int,
) -> dict[str, Any]:
    equal_weight = np.concatenate(
        [np.full(config.n_risk_assets, 1.0 / config.n_risk_assets), [0.0]]
    )
    training_returns = market.log_returns[prepared.warmup : prepared.fit_end]
    static_covariance = core.estimate_covariance(training_returns, method="ledoit_wolf")
    gmv_constraints = _constraints(core, max_cash=config.benchmark_cash_weight)
    gmv_optimiser = core.ModelPredictiveControlOptimiser(
        config.n_assets,
        constraints=gmv_constraints,
        risk_aversion=1.0,
        turnover_penalty=0.0,
    )
    gmv_initial = core.project_long_only_weights(
        _initial_weights(config, config.benchmark_cash_weight),
        constraints=gmv_constraints,
    )
    gmv_weights = gmv_optimiser.allocate(
        np.zeros(config.n_assets), static_covariance, gmv_initial
    )

    regime_allocations: list[np.ndarray] = []
    cash_targets = np.linspace(
        config.normal_cash_target,
        config.crisis_cash_target,
        config.n_regimes,
    )
    training_probabilities = prepared.hmm_probabilities[
        prepared.warmup : prepared.fit_end
    ]
    for state, cash_target in enumerate(cash_targets):
        covariance, _ = core.compute_regime_weighted_covariance(
            training_returns,
            training_probabilities[:, state],
            minimum_effective_samples=config.minimum_effective_samples,
            fallback_covariance=static_covariance,
        )
        state_constraints = _constraints(core, max_cash=float(cash_target))
        optimiser = core.ModelPredictiveControlOptimiser(
            config.n_assets,
            constraints=state_constraints,
            risk_aversion=1.0,
            turnover_penalty=0.0,
        )
        current = core.project_long_only_weights(
            _initial_weights(config, float(cash_target)),
            constraints=state_constraints,
        )
        regime_allocations.append(
            optimiser.allocate(np.zeros(config.n_assets), covariance, current)
        )

    jump_mpc = core.JumpModelMPCBenchmark(
        config.n_assets,
        n_components=config.n_regimes,
        constraints=_constraints(core, max_cash=config.maximum_cash),
        random_state=seed,
    )
    jump_mpc.fit(
        prepared.hmm_observations[prepared.hmm_start : prepared.fit_end],
        market.log_returns[prepared.hmm_start : prepared.fit_end],
    )
    return {
        "1/N": equal_weight,
        "GMV": gmv_weights,
        "HMM-GMV": np.asarray(regime_allocations),
        "JM-MPC": jump_mpc,
    }



def build_policy_base_path(
    classical: Mapping[str, Any],
    prepared: PreparedData,
    config: SimulationConfig,
) -> np.ndarray:
    """Construct a causal financial prior for residual policy learning."""
    path = np.repeat(_initial_weights(config)[None, :], config.n_samples, axis=0)
    method = config.policy_base_method
    if method in {"1/N", "GMV"}:
        path[:] = np.asarray(classical[method], dtype=float)
        return path
    allocations = np.asarray(classical["HMM-GMV"], dtype=float)
    for t in range(config.n_samples):
        probability_index = max(prepared.hmm_start, t - 1)
        path[t] = prepared.hmm_probabilities[probability_index] @ allocations
    return path

def _benchmark_loss(
    weights: torch.Tensor,
    realised_returns: torch.Tensor,
    config: SimulationConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    gross_growth = torch.sum(weights * torch.exp(realised_returns), dim=-1)
    portfolio_returns = torch.log(torch.clamp(gross_growth, min=1e-12))
    concentration = torch.sum(weights**2, dim=-1)
    loss = (
        -portfolio_returns.mean()
        + config.benchmark_variance_penalty * torch.mean(portfolio_returns**2)
        + config.benchmark_concentration_penalty * concentration.mean()
    )
    return loss, portfolio_returns


def _validation_supervised_loss(
    model: Any,
    kind: str,
    market: MarketData,
    prepared: PreparedData,
    config: SimulationConfig,
    device: torch.device,
) -> float:
    start = prepared.fit_end
    end = config.train_end
    indices = np.arange(start, end, dtype=int)
    if indices.size == 0:
        return float("inf")
    with torch.no_grad():
        sequence = prepared.features[indices]
        realised = torch.as_tensor(market.log_returns[indices], dtype=torch.float32, device=device)
        if kind == "GLASSO-GAT":
            adjacency = prepared.static_risk_adjacency.expand(indices.size, -1, -1)
            weights = model(sequence, adjacency)
            loss, _ = _benchmark_loss(weights, realised, config)
        else:
            weights, values, information = model(sequence)
            policy_loss, portfolio_returns = _benchmark_loss(weights, realised, config)
            loss = (
                policy_loss
                + config.tcmac_value_coefficient * F.smooth_l1_loss(values, portfolio_returns)
                - config.tcmac_information_coefficient * F.logsigmoid(information).mean()
            )
    return float(loss.cpu())


def train_standard_neural_benchmarks(
    core: ModuleType,
    market: MarketData,
    prepared: PreparedData,
    config: SimulationConfig,
    device: torch.device,
    seed: int,
) -> tuple[Any, Any, pd.DataFrame, dict[str, float]]:
    core.seed_everything(seed + 17, deterministic=True)
    glasso_model = core.GLASSOGATBenchmark(
        config.n_risk_assets,
        len(prepared.feature_names),
        hidden_gru=config.hidden_gru,
        hidden_gat=config.hidden_gat,
        max_cash=config.maximum_cash,
        dropout=config.dropout,
    ).to(device)
    tcmac_model = core.TCMACBenchmark(
        config.n_risk_assets,
        len(prepared.feature_names),
        hidden_dim=config.hidden_tcmac,
        max_cash=config.maximum_cash,
    ).to(device)
    glasso_optimiser = torch.optim.Adam(glasso_model.parameters(), lr=config.actor_learning_rate)
    tcmac_optimiser = torch.optim.Adam(tcmac_model.parameters(), lr=config.actor_learning_rate)

    indices = np.arange(
        max(prepared.warmup + 1, prepared.hmm_start + 1),
        prepared.fit_end,
        dtype=int,
    )
    rng = np.random.default_rng(seed + 17)
    rows: list[dict[str, Any]] = []
    elapsed = {"GLASSO-GAT": 0.0, "TC-MAC": 0.0}
    # Keep a valid fallback even if every validation loss is non-finite.  This
    # prevents a late KeyError from obscuring the actual numerical diagnostic.
    best_states: dict[str, Any] = {
        "GLASSO-GAT": copy.deepcopy(glasso_model.state_dict()),
        "TC-MAC": copy.deepcopy(tcmac_model.state_dict()),
    }
    best_losses = {"GLASSO-GAT": float("inf"), "TC-MAC": float("inf")}
    no_improvement = {"GLASSO-GAT": 0, "TC-MAC": 0}

    for epoch in range(config.benchmark_epochs):
        shuffled = rng.permutation(indices)
        for start in range(0, shuffled.size, config.batch_size):
            batch_numpy = shuffled[start : start + config.batch_size]
            if batch_numpy.size < 2:
                continue
            batch = torch.as_tensor(batch_numpy, dtype=torch.long, device=device)
            sequence = prepared.features[batch]
            realised = torch.as_tensor(
                market.log_returns[batch_numpy], dtype=torch.float32, device=device
            )
            adjacency = prepared.static_risk_adjacency.expand(batch_numpy.size, -1, -1)

            clock = time.perf_counter()
            glasso_weights = glasso_model(sequence, adjacency)
            glasso_loss, _ = _benchmark_loss(glasso_weights, realised, config)
            glasso_optimiser.zero_grad(set_to_none=True)
            glasso_loss.backward()
            torch.nn.utils.clip_grad_norm_(glasso_model.parameters(), config.max_gradient_norm)
            glasso_optimiser.step()
            elapsed["GLASSO-GAT"] += time.perf_counter() - clock

            clock = time.perf_counter()
            tcmac_weights, tcmac_values, information = tcmac_model(sequence)
            policy_loss, portfolio_returns = _benchmark_loss(tcmac_weights, realised, config)
            value_loss = F.smooth_l1_loss(tcmac_values, portfolio_returns.detach())
            information_loss = -F.logsigmoid(information).mean()
            tcmac_loss = (
                policy_loss
                + config.tcmac_value_coefficient * value_loss
                + config.tcmac_information_coefficient * information_loss
            )
            tcmac_optimiser.zero_grad(set_to_none=True)
            tcmac_loss.backward()
            torch.nn.utils.clip_grad_norm_(tcmac_model.parameters(), config.max_gradient_norm)
            tcmac_optimiser.step()
            elapsed["TC-MAC"] += time.perf_counter() - clock

        glasso_model.eval()
        tcmac_model.eval()
        glasso_validation = _validation_supervised_loss(
            glasso_model, "GLASSO-GAT", market, prepared, config, device
        )
        tcmac_validation = _validation_supervised_loss(
            tcmac_model, "TC-MAC", market, prepared, config, device
        )
        if glasso_validation < best_losses["GLASSO-GAT"] - 1e-8:
            best_losses["GLASSO-GAT"] = glasso_validation
            best_states["GLASSO-GAT"] = copy.deepcopy(glasso_model.state_dict())
            no_improvement["GLASSO-GAT"] = 0
        else:
            no_improvement["GLASSO-GAT"] += 1
        if tcmac_validation < best_losses["TC-MAC"] - 1e-8:
            best_losses["TC-MAC"] = tcmac_validation
            best_states["TC-MAC"] = copy.deepcopy(tcmac_model.state_dict())
            no_improvement["TC-MAC"] = 0
        else:
            no_improvement["TC-MAC"] += 1
        rows.append(
            {
                "Model": "GLASSO-GAT",
                "Epoch": epoch,
                "Validation loss": glasso_validation,
            }
        )
        rows.append(
            {
                "Model": "TC-MAC",
                "Epoch": epoch,
                "Validation loss": tcmac_validation,
            }
        )
        glasso_model.train()
        tcmac_model.train()
        if (
            epoch + 1 >= config.benchmark_minimum_epochs
            and all(value >= config.benchmark_validation_patience for value in no_improvement.values())
        ):
            break

    glasso_model.load_state_dict(best_states["GLASSO-GAT"])
    tcmac_model.load_state_dict(best_states["TC-MAC"])
    glasso_model.eval()
    tcmac_model.eval()
    return glasso_model, tcmac_model, pd.DataFrame(rows), elapsed




def train_benchmark_models(
    core: ModuleType,
    market: MarketData,
    prepared: PreparedData,
    config: SimulationConfig,
    device: torch.device,
    seed: int,
) -> ModelBundle:
    glasso, tcmac, diagnostics, training_times = train_standard_neural_benchmarks(
        core,
        market,
        prepared,
        config,
        device,
        seed,
    )
    return ModelBundle(
        glasso_gat=glasso,
        tcmac=tcmac,
        training_diagnostics=diagnostics,
        training_times=training_times,
    )


def _expert_actions(
    core: ModuleType,
    bundle: ModelBundle,
    classical: Mapping[str, Any],
    prepared: PreparedData,
    states: Mapping[str, PortfolioState],
    t: int,
    config: SimulationConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    probability_index = max(prepared.hmm_start, t - 1)
    probability = prepared.hmm_probabilities[probability_index]
    sequence = prepared.features[t : t + 1]

    with torch.no_grad():
        glasso = bundle.glasso_gat(
            sequence,
            prepared.static_risk_adjacency,
        ).squeeze(0).cpu().numpy()
        tcmac = bundle.tcmac(sequence)[0].squeeze(0).cpu().numpy()

    hmm_gmv = probability @ np.asarray(classical["HMM-GMV"], dtype=float)
    jm_mpc = classical["JM-MPC"].allocate(
        prepared.hmm_observations[probability_index],
        states["JM-MPC"].weights,
    )
    raw = {
        "1/N": classical["1/N"],
        "GMV": classical["GMV"],
        "HMM-GMV": hmm_gmv,
        "GLASSO-GAT": glasso,
        "TC-MAC": tcmac,
        "JM-MPC": jm_mpc,
    }
    return {
        method: _safe_weights(core, action, config)
        for method, action in raw.items()
    }


def build_causal_teacher_path(
    core: ModuleType,
    bundle: ModelBundle,
    classical: Mapping[str, Any],
    market: MarketData,
    prepared: PreparedData,
    config: SimulationConfig,
    device: torch.device,
) -> tuple[np.ndarray, pd.DataFrame]:
    start = max(prepared.warmup + config.graph_min_history, prepared.hmm_start + 1)
    bases = np.repeat(_initial_weights(config)[None, :], config.n_samples, axis=0)
    states = {method: PortfolioState(_initial_weights(config)) for method in EXPERT_ORDER}
    return_histories: dict[str, list[float]] = {method: [] for method in EXPERT_ORDER}
    turnover_histories: dict[str, list[float]] = {method: [] for method in EXPERT_ORDER}
    rows: list[dict[str, Any]] = []

    for t in range(start, config.n_samples):
        actions = _expert_actions(
            core, bundle, classical, prepared, states, t, config, device
        )
        scores = []
        for method in EXPERT_ORDER:
            past_returns = np.asarray(return_histories[method][-config.teacher_window :], dtype=float)
            past_turnover = np.asarray(turnover_histories[method][-config.teacher_window :], dtype=float)
            if past_returns.size < 5:
                score = 0.0
            else:
                wealth = np.concatenate([[1.0], np.exp(np.cumsum(past_returns))])
                peaks = np.maximum.accumulate(wealth)
                trailing_drawdown = abs(float(np.min(wealth / peaks - 1.0)))
                trailing_cvar = abs(min(_empirical_cvar(past_returns), 0.0))
                score = (
                    float(past_returns.mean())
                    - 0.25 * float(past_returns.std(ddof=1))
                    - 0.50 * trailing_cvar
                    - 0.10 * trailing_drawdown
                    - 0.05 * float(past_turnover.mean())
                )
            scores.append(score)
        scaled = np.asarray(scores, dtype=float) / max(config.teacher_temperature, 1e-6)
        scaled -= scaled.max()
        mixture = np.exp(np.clip(scaled, -50.0, 50.0))
        mixture /= mixture.sum()
        base = sum(weight * actions[method] for weight, method in zip(mixture, EXPERT_ORDER))
        bases[t] = _safe_weights(core, base, config)
        rows.append(
            {
                "Time": t,
                **{f"Weight {method}": float(weight) for method, weight in zip(EXPERT_ORDER, mixture)},
            }
        )
        for method in EXPERT_ORDER:
            next_state, info = _apply_action(
                states[method], actions[method], market.log_returns[t], config
            )
            states[method] = next_state
            return_histories[method].append(info["log_return"])
            turnover_histories[method].append(info["turnover"])
    return bases, pd.DataFrame(rows)


def _cash_bounds_for_state(
    core: ModuleType,
    agent: Any,
    sequence: torch.Tensor,
    risk_adjacency: torch.Tensor,
    predictive_adjacency: torch.Tensor,
    external: torch.Tensor,
    portfolio: PortfolioState,
    probability: np.ndarray,
    config: SimulationConfig,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[float, float]]:
    if not config.use_dynamic_cash:
        low, high = 0.0, config.maximum_cash
        return torch.tensor([[low, high]], dtype=torch.float32, device=device), (low, high)
    predicted_cvar = 0.0
    if config.use_quantile_head:
        with torch.no_grad():
            quantiles = agent.predict_quantiles(
                sequence, risk_adjacency, predictive_adjacency, external
            )
        predicted_cvar = float(quantiles[0, 1].cpu()) if quantiles.shape[1] > 1 else 0.0
    entropy = float(
        -np.sum(probability * np.log(np.clip(probability, 1e-12, 1.0)))
        / math.log(max(2, probability.size))
    )
    low, high = core.cash_bounds_from_signals(
        float(probability[-1]),
        portfolio.drawdown,
        predicted_cvar,
        entropy,
        normal_minimum=0.0,
        crisis_minimum=config.crisis_cash_minimum,
        normal_maximum=config.normal_cash_maximum,
        maximum_cash=config.maximum_cash,
    )
    tensor = torch.tensor([[low, high]], dtype=torch.float32, device=device)
    return tensor, (low, high)


def _proposal_validation(
    core: ModuleType,
    agent: Any,
    market: MarketData,
    prepared: PreparedData,
    teacher_path: np.ndarray,
    config: SimulationConfig,
    device: torch.device,
) -> tuple[tuple[int, float, float, float], dict[str, float]]:
    state = PortfolioState(_initial_weights(config))
    returns: list[float] = []
    turns: list[float] = []
    concentrations: list[float] = []
    agent.eval()
    for t in range(prepared.fit_end, config.train_end):
        probability_index = max(prepared.hmm_start, t - 1)
        previous_index = max(prepared.hmm_start, probability_index - 1)
        probability = prepared.hmm_probabilities[probability_index]
        sequence = prepared.features[t : t + 1]
        risk_adjacency = prepared.dynamic_risk_adjacencies[t : t + 1]
        predictive_adjacency = prepared.dynamic_predictive_adjacencies[t : t + 1]
        external = _external_state(
            state,
            probability,
            prepared.hmm_probabilities[previous_index],
            device,
        )
        bounds_tensor, bounds = _cash_bounds_for_state(
            core,
            agent,
            sequence,
            risk_adjacency,
            predictive_adjacency,
            external,
            state,
            probability,
            config,
            device,
        )
        raw_base = teacher_path[t] if config.use_benchmark_anchor else _initial_weights(config)
        base = _enforce_cash_interval(raw_base, *bounds)
        base_tensor = torch.as_tensor(base, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action = agent.deterministic_action(
                sequence,
                risk_adjacency,
                predictive_adjacency,
                external,
                base_tensor,
                bounds_tensor,
            ).squeeze(0).cpu().numpy()
        target = _safe_weights(core, action, config)
        state, info = _apply_action(state, target, market.log_returns[t], config)
        returns.append(info["log_return"])
        turns.append(info["turnover"])
        concentrations.append(info["hhi"])
    metrics = _portfolio_metrics(returns, turns, concentrations, config.periods_per_year)
    return _validation_rank(metrics, config), metrics


def _teacher_state_path(
    market: MarketData,
    teacher_path: np.ndarray,
    config: SimulationConfig,
    start: int,
    end: int,
) -> dict[int, PortfolioState]:
    states: dict[int, PortfolioState] = {}
    state = PortfolioState(_initial_weights(config))
    for t in range(start, end):
        states[t] = PortfolioState(
            state.weights.copy(),
            wealth=state.wealth,
            peak=state.peak,
            drawdown=state.drawdown,
        )
        state, _ = _apply_action(
            state, teacher_path[t], market.log_returns[t], config
        )
    states[end] = state
    return states


def _normalise_advantage(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 2:
        return values
    return (values - values.mean()) / (values.std(unbiased=False) + 1e-8)



def train_policy_agent(
    core: ModuleType,
    market: MarketData,
    prepared: PreparedData,
    teacher_path: np.ndarray,
    config: SimulationConfig,
    device: torch.device,
    seed: int,
    *,
    label: str,
    use_risk_graph: bool,
    use_predictive_graph: bool,
    temporal_mode: str,
) -> tuple[Any, pd.DataFrame, float, dict[str, float]]:
    core.seed_everything(seed, deterministic=True)
    agent = core.XGATDRLAgent(
        config.n_risk_assets,
        len(prepared.feature_names),
        config.external_state_dim,
        hidden_gru=config.hidden_gru,
        hidden_gat=config.hidden_gat,
        max_cash=config.maximum_cash,
        dropout=config.dropout,
        use_risk_graph=use_risk_graph,
        use_predictive_graph=use_predictive_graph,
        use_anchor=config.use_benchmark_anchor,
        temporal_mode=temporal_mode,
        predictive_lags=config.predictive_lags,
        use_optimizer=config.use_differentiable_optimizer,
    ).to(device)
    encoder_optimiser = torch.optim.Adam(
        agent.encoder_parameters(), lr=config.encoder_learning_rate, eps=1e-5
    )
    actor_optimiser = torch.optim.Adam(
        agent.policy_parameters(), lr=config.actor_learning_rate, eps=1e-5
    )
    critic_optimiser = torch.optim.Adam(
        agent.critic_parameters(), lr=config.critic_learning_rate, eps=1e-5
    )
    policy_optimisers = {
        "encoder": encoder_optimiser,
        "actor": actor_optimiser,
        "critic": critic_optimiser,
    }
    ppo_config = core.PPOConfig(
        clip_epsilon=config.clip_epsilon,
        value_coefficient=config.value_coefficient,
        entropy_coefficient=config.entropy_coefficient,
        value_clip=config.value_clip,
        max_gradient_norm=config.max_gradient_norm,
        target_kl=config.target_kl,
    )
    budget_config = core.RiskBudgetConfig(
        cvar_limit=config.cvar_limit,
        drawdown_limit=config.drawdown_limit,
        turnover_limit=config.turnover_limit,
        hhi_limit=config.maximum_hhi,
        multiplier_learning_rate=config.multiplier_learning_rate,
        active_constraints=config.active_cmdp_constraints,
    )
    controller = core.LagrangeController(budget_config)
    earliest = max(prepared.warmup + config.graph_min_history, prepared.hmm_start + 1)
    latest = prepared.fit_end - config.episode_length
    starts = np.arange(earliest, latest + 1, dtype=int)
    if starts.size == 0:
        raise ValueError("No valid PPO episode start exists.")
    state_path = teacher_path if config.use_benchmark_anchor else np.repeat(
        _initial_weights(config)[None, :], config.n_samples, axis=0
    )
    starting_states = _teacher_state_path(market, state_path, config, earliest, prepared.fit_end)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    clock = time.perf_counter()

    # Stage A/B: representation and allocation pretraining with causal one-step
    # returns and multi-step risk targets. This gives every temporal ablation the
    # same direct economic signal before PPO.
    pretrain_indices = np.arange(earliest, prepared.fit_end - config.future_risk_horizon, dtype=int)
    for epoch in range(config.representation_pretrain_epochs):
        diagnostics_epoch: list[dict[str, float]] = []
        for _ in range(config.representation_batches_per_epoch):
            batch_indices = rng.choice(
                pretrain_indices,
                size=min(config.batch_size, pretrain_indices.size),
                replace=False,
            )
            sequences, risk_graphs, predictive_graphs = [], [], []
            externals, bases, bounds_list, previous_list = [], [], [], []
            realised_list, future_risk_list = [], []
            for t_index in batch_indices:
                state = starting_states[int(t_index)]
                probability_index = max(prepared.hmm_start, int(t_index) - 1)
                previous_index = max(prepared.hmm_start, probability_index - 1)
                probability = prepared.hmm_probabilities[probability_index]
                sequence = prepared.features[t_index : t_index + 1]
                risk_adjacency = prepared.dynamic_risk_adjacencies[t_index : t_index + 1]
                predictive_adjacency = prepared.dynamic_predictive_adjacencies[t_index : t_index + 1]
                external = _external_state(
                    state, probability, prepared.hmm_probabilities[previous_index], device
                )
                bounds_tensor, bounds = _cash_bounds_for_state(
                    core, agent, sequence, risk_adjacency, predictive_adjacency,
                    external, state, probability, config, device
                )
                raw_base = teacher_path[t_index] if config.use_benchmark_anchor else state.weights
                base = _enforce_cash_interval(raw_base, *bounds)
                future = market.log_returns[
                    t_index : t_index + config.future_risk_horizon,
                    : config.n_risk_assets,
                ]
                future_risk = np.sqrt(np.mean(future**2, axis=0) + 1e-12)
                sequences.append(sequence)
                risk_graphs.append(risk_adjacency)
                predictive_graphs.append(predictive_adjacency)
                externals.append(external)
                bases.append(torch.as_tensor(base, dtype=torch.float32, device=device).unsqueeze(0))
                bounds_list.append(bounds_tensor)
                previous_list.append(torch.as_tensor(state.weights, dtype=torch.float32, device=device).unsqueeze(0))
                realised_list.append(torch.as_tensor(market.log_returns[t_index], dtype=torch.float32, device=device).unsqueeze(0))
                future_risk_list.append(torch.as_tensor(future_risk, dtype=torch.float32, device=device).unsqueeze(0))
            loss, diagnostics = core.hybrid_pretraining_loss(
                agent,
                (
                    torch.cat(sequences), torch.cat(risk_graphs), torch.cat(predictive_graphs),
                    torch.cat(externals), torch.cat(bases), torch.cat(bounds_list),
                ),
                torch.cat(realised_list),
                torch.cat(future_risk_list),
                torch.cat(previous_list),
                target_scale=prepared.reward_scale_factor,
                transaction_cost=config.transaction_cost,
                slippage_coefficient=config.slippage_coefficient,
                impact_coefficient=config.impact_coefficient,
                downside_penalty=config.direct_downside_penalty,
                concentration_penalty=config.direct_concentration_penalty,
            )
            encoder_optimiser.zero_grad(set_to_none=True)
            actor_optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.encoder_parameters() + agent.policy_parameters(),
                config.max_gradient_norm,
            )
            encoder_optimiser.step()
            actor_optimiser.step()
            diagnostics_epoch.append(diagnostics)
        rows.append({
            "Policy": label,
            "Stage": "pretrain",
            "Episode": epoch,
            "Start": np.nan,
            **{
                key: float(np.mean([item[key] for item in diagnostics_epoch]))
                for key in diagnostics_epoch[0]
            },
        })

    behaviour_epochs = config.behaviour_epochs if config.use_behaviour_cloning else 0
    for epoch in range(behaviour_epochs):
        losses: list[float] = []
        for _ in range(config.behaviour_batches_per_epoch):
            indices = rng.choice(
                np.arange(earliest, prepared.fit_end),
                size=min(config.batch_size, prepared.fit_end - earliest),
                replace=False,
            )
            sequences, risk_graphs, predictive_graphs = [], [], []
            externals, bases, bounds_list, teachers = [], [], [], []
            for t in indices:
                state = starting_states[int(t)]
                probability_index = max(prepared.hmm_start, int(t) - 1)
                previous_index = max(prepared.hmm_start, probability_index - 1)
                probability = prepared.hmm_probabilities[probability_index]
                sequence = prepared.features[t : t + 1]
                risk_adjacency = prepared.dynamic_risk_adjacencies[t : t + 1]
                predictive_adjacency = prepared.dynamic_predictive_adjacencies[t : t + 1]
                external = _external_state(
                    state, probability, prepared.hmm_probabilities[previous_index], device
                )
                bounds_tensor, bounds = _cash_bounds_for_state(
                    core, agent, sequence, risk_adjacency, predictive_adjacency,
                    external, state, probability, config, device
                )
                teacher = _enforce_cash_interval(teacher_path[t], *bounds)
                sequences.append(sequence)
                risk_graphs.append(risk_adjacency)
                predictive_graphs.append(predictive_adjacency)
                externals.append(external)
                bases.append(torch.as_tensor(teacher, dtype=torch.float32, device=device).unsqueeze(0))
                bounds_list.append(bounds_tensor)
                teachers.append(torch.as_tensor(teacher, dtype=torch.float32, device=device).unsqueeze(0))
            loss = core.behaviour_cloning_loss(
                agent,
                (
                    torch.cat(sequences),
                    torch.cat(risk_graphs),
                    torch.cat(predictive_graphs),
                    torch.cat(externals),
                    torch.cat(bases),
                    torch.cat(bounds_list),
                ),
                torch.cat(teachers),
            )
            encoder_optimiser.zero_grad(set_to_none=True)
            actor_optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                agent.encoder_parameters() + agent.policy_parameters(),
                config.max_gradient_norm,
            )
            encoder_optimiser.step()
            actor_optimiser.step()
            losses.append(float(loss.detach().cpu()))
        rows.append({
            "Policy": label,
            "Stage": "behaviour",
            "Episode": epoch,
            "Start": np.nan,
            "Loss": float(np.mean(losses)) if losses else np.nan,
        })

    crisis_scores = prepared.hmm_probabilities[
        np.maximum(prepared.hmm_start, starts - 1), -1
    ]
    crisis_weights = np.exp(crisis_scores - np.max(crisis_scores))
    crisis_weights /= crisis_weights.sum()
    kl_coefficient = float(config.initial_kl_coefficient)
    best_state = copy.deepcopy(agent.state_dict())
    best_controller = controller.state_dict()
    best_rank: tuple[int, float, float, float] | None = None
    no_improvement = 0

    for episode in range(config.ppo_episodes):
        curriculum = config.stress_sampling_final * episode / max(1, config.ppo_episodes - 1)
        if rng.random() < curriculum:
            episode_start = int(rng.choice(starts, p=crisis_weights))
        else:
            episode_start = int(rng.choice(starts))
        source_state = starting_states[episode_start]
        portfolio = PortfolioState(
            source_state.weights.copy(),
            wealth=source_state.wealth,
            peak=source_state.peak,
            drawdown=source_state.drawdown,
        )
        episode_stop = min(episode_start + config.episode_length, prepared.fit_end)
        sequences, risk_graphs, predictive_graphs = [], [], []
        externals, bases, cash_bounds = [], [], []
        actions, old_log_probabilities, values = [], [], []
        previous_weights, realised_asset_returns = [], []
        terminals, rewards, infos = [], [], []
        cost_paths = {name: [] for name in config.active_cmdp_constraints}
        cost_values = {name: [] for name in config.active_cmdp_constraints}
        limit_paths = {name: [] for name in config.active_cmdp_constraints}

        agent.eval()
        for t in range(episode_start, episode_stop):
            probability_index = max(prepared.hmm_start, t - 1)
            previous_index = max(prepared.hmm_start, probability_index - 1)
            probability = prepared.hmm_probabilities[probability_index]
            sequence = prepared.features[t : t + 1]
            risk_adjacency = prepared.dynamic_risk_adjacencies[t : t + 1]
            predictive_adjacency = prepared.dynamic_predictive_adjacencies[t : t + 1]
            external = _external_state(
                portfolio, probability, prepared.hmm_probabilities[previous_index], device
            )
            bounds_tensor, bounds = _cash_bounds_for_state(
                core, agent, sequence, risk_adjacency, predictive_adjacency,
                external, portfolio, probability, config, device
            )
            raw_base = teacher_path[t] if config.use_benchmark_anchor else _initial_weights(config)
            base = _enforce_cash_interval(raw_base, *bounds)
            base_tensor = torch.as_tensor(base, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_probability, _, value, predicted_costs = agent.sample_action_with_costs(
                    sequence,
                    risk_adjacency,
                    predictive_adjacency,
                    external,
                    base_tensor,
                    bounds_tensor,
                )
                policy_base = agent.deterministic_base(
                    sequence,
                    risk_adjacency,
                    predictive_adjacency,
                    external,
                    base_tensor,
                    bounds_tensor,
                )
            previous_weights.append(
                torch.as_tensor(portfolio.weights, dtype=torch.float32, device=device).unsqueeze(0)
            )
            realised_asset_returns.append(
                torch.as_tensor(market.log_returns[t], dtype=torch.float32, device=device).unsqueeze(0)
            )
            action_numpy = _safe_weights(core, action.squeeze(0).cpu().numpy(), config)
            previous_state = PortfolioState(
                portfolio.weights.copy(), portfolio.wealth, portfolio.peak, portfolio.drawdown
            )
            portfolio, info = _apply_action(portfolio, action_numpy, market.log_returns[t], config)
            base_numpy = _safe_weights(core, policy_base.squeeze(0).cpu().numpy(), config)
            _, base_info = _apply_action(previous_state, base_numpy, market.log_returns[t], config)
            limits = core.regime_dependent_limits(float(probability[-1]), budget_config)
            raw_costs = {
                "cvar": 0.0,
                "drawdown": max(abs(min(info["drawdown"], 0.0)) - limits["drawdown"], 0.0) / max(limits["drawdown"], 1e-8),
                "turnover": max(info["turnover"] - limits["turnover"], 0.0) / max(limits["turnover"], 1e-8),
                "hhi": max(info["hhi"] - limits["hhi"], 0.0) / max(limits["hhi"], 1e-8),
            }
            for name in config.active_cmdp_constraints:
                cost_paths[name].append(float(raw_costs[name]))
                cost_values[name].append(predicted_costs[name].detach())
                limit_paths[name].append(float(limits[name]))
            sequences.append(sequence.detach())
            risk_graphs.append(risk_adjacency.detach())
            predictive_graphs.append(predictive_adjacency.detach())
            externals.append(external.detach())
            bases.append(base_tensor.detach())
            cash_bounds.append(bounds_tensor.detach())
            actions.append(action.detach())
            old_log_probabilities.append(log_probability.detach())
            values.append(value.detach())
            active_return = info["log_return"] - base_info["log_return"]
            reward_return = (
                config.active_reward_weight * active_return
                + config.absolute_reward_weight * info["log_return"]
                if config.use_active_reward
                else info["log_return"]
            )
            rewards.append(float(config.reward_scale * prepared.reward_scale_factor * reward_return))
            terminals.append(t == prepared.fit_end - 1)
            infos.append(info)

        if "cvar" in config.active_cmdp_constraints:
            losses = np.maximum(-np.asarray([item["log_return"] for item in infos], dtype=float), 0.0)
            var_level = float(np.quantile(losses, 1.0 - config.cvar_alpha))
            cvar_path = var_level + np.maximum(losses - var_level, 0.0) / max(config.cvar_alpha, 1e-8)
            cvar_limits = np.asarray(limit_paths["cvar"], dtype=float)
            cost_paths["cvar"] = list(
                np.maximum(cvar_path - cvar_limits, 0.0) / np.maximum(cvar_limits, 1e-8)
            )

        next_value = 0.0
        next_cost_values = {name: 0.0 for name in config.active_cmdp_constraints}
        if episode_stop < prepared.fit_end:
            t = episode_stop
            probability_index = max(prepared.hmm_start, t - 1)
            previous_index = max(prepared.hmm_start, probability_index - 1)
            probability = prepared.hmm_probabilities[probability_index]
            sequence = prepared.features[t : t + 1]
            risk_adjacency = prepared.dynamic_risk_adjacencies[t : t + 1]
            predictive_adjacency = prepared.dynamic_predictive_adjacencies[t : t + 1]
            external = _external_state(
                portfolio, probability, prepared.hmm_probabilities[previous_index], device
            )
            with torch.no_grad():
                reward_value, predicted_costs = agent.predict_values(
                    sequence, risk_adjacency, predictive_adjacency, external
                )
            next_value = float(reward_value.squeeze().cpu())
            next_cost_values = {
                name: float(predicted_costs[name].squeeze().cpu())
                for name in config.active_cmdp_constraints
            }

        value_numbers = [float(value.squeeze().cpu()) for value in values]
        returns, reward_advantages = core.compute_gae(
            rewards,
            value_numbers,
            terminals,
            next_value=next_value,
            gamma=config.gamma,
            tau=config.gae_lambda,
            normalise=False,
        )
        policy_advantages = reward_advantages.clone()
        cost_return_tensors: dict[str, torch.Tensor] = {}
        old_cost_tensors: dict[str, torch.Tensor] = {}
        if config.use_cmdp:
            multipliers = controller.state_dict()
            for name in config.active_cmdp_constraints:
                old_values = [float(value.squeeze().cpu()) for value in cost_values[name]]
                cost_returns, cost_advantages = core.compute_gae(
                    cost_paths[name],
                    old_values,
                    terminals,
                    next_value=next_cost_values[name],
                    gamma=config.gamma,
                    tau=config.gae_lambda,
                    normalise=False,
                )
                policy_advantages = policy_advantages - float(multipliers[name]) * cost_advantages
                cost_return_tensors[name] = cost_returns.to(device)
                old_cost_tensors[name] = torch.cat(cost_values[name]).to(device)
        policy_advantages = _normalise_advantage(policy_advantages).to(device)

        tensors = (
            torch.cat(sequences),
            torch.cat(risk_graphs),
            torch.cat(predictive_graphs),
            torch.cat(externals),
            torch.cat(bases),
            torch.cat(cash_bounds),
        )
        action_tensor = torch.cat(actions)
        log_tensor = torch.cat(old_log_probabilities)
        old_value_tensor = torch.cat(values)
        returns = returns.to(device)
        realised_tensor = torch.cat(realised_asset_returns)
        previous_weight_tensor = torch.cat(previous_weights)
        reward_path = np.asarray(rewards, dtype=np.float32)
        quantile_targets = torch.as_tensor(
            np.asarray(
                [
                    reward_path[index : min(reward_path.size, index + config.quantile_horizon)].sum()
                    for index in range(reward_path.size)
                ],
                dtype=np.float32,
            ),
            dtype=torch.float32,
            device=device,
        )

        agent.train()
        agent.set_encoder_trainable(episode >= config.encoder_freeze_episodes)
        update_diagnostics: list[dict[str, float | bool]] = []
        stop_for_kl = False
        for _ in range(config.ppo_update_epochs):
            epoch_diagnostics: list[dict[str, float | bool]] = []
            permutation = torch.randperm(action_tensor.size(0), device=device)
            for start in range(0, action_tensor.size(0), config.batch_size):
                index = permutation[start : start + config.batch_size]
                batch_states = tuple(tensor[index] for tensor in tensors)
                diagnostics = core.constrained_ppo_update_batch(
                    agent,
                    policy_optimisers,
                    batch_states,
                    action_tensor[index],
                    log_tensor[index],
                    old_value_tensor[index],
                    returns[index],
                    policy_advantages[index],
                    config=ppo_config,
                    quantile_targets=quantile_targets[index],
                    quantile_coefficient=(config.quantile_coefficient if config.use_quantile_head else 0.0),
                    kl_coefficient=kl_coefficient,
                    old_cost_values={name: tensor[index] for name, tensor in old_cost_tensors.items()},
                    cost_returns={name: tensor[index] for name, tensor in cost_return_tensors.items()},
                    cost_value_coefficient=config.cost_value_coefficient,
                    realised_asset_returns=realised_tensor[index],
                    previous_weights=previous_weight_tensor[index],
                    direct_utility_coefficient=config.direct_utility_coefficient,
                    expert_auxiliary_coefficient=(config.expert_auxiliary_coefficient if use_risk_graph or use_predictive_graph else 0.0),
                    gate_auxiliary_coefficient=(config.gate_auxiliary_coefficient if use_predictive_graph else 0.0),
                    auxiliary_target_scale=prepared.reward_scale_factor,
                    transaction_cost=config.transaction_cost,
                    slippage_coefficient=config.slippage_coefficient,
                    impact_coefficient=config.impact_coefficient,
                    direct_downside_penalty=config.direct_downside_penalty,
                    direct_concentration_penalty=config.direct_concentration_penalty,
                )
                epoch_diagnostics.append(diagnostics)
                for _ in range(max(0, config.critic_updates_per_actor - 1)):
                    core.critic_regression_update_batch(
                        agent,
                        critic_optimiser,
                        batch_states,
                        returns[index],
                        quantile_targets=quantile_targets[index],
                        quantile_coefficient=(config.quantile_coefficient if config.use_quantile_head else 0.0),
                        cost_returns={name: tensor[index] for name, tensor in cost_return_tensors.items()},
                        cost_value_coefficient=config.cost_value_coefficient,
                        max_gradient_norm=config.max_gradient_norm,
                    )
            update_diagnostics.extend(epoch_diagnostics)
            if config.target_kl is not None and epoch_diagnostics:
                epoch_kl = float(np.mean([float(item["Approximate KL"]) for item in epoch_diagnostics]))
                if epoch_kl > 1.5 * config.target_kl:
                    stop_for_kl = True
                    break

        if update_diagnostics and config.target_kl is not None:
            mean_kl = float(np.mean([float(item["Approximate KL"]) for item in update_diagnostics]))
            if mean_kl > 1.25 * config.target_kl:
                kl_coefficient = min(20.0, 1.5 * kl_coefficient)
            elif mean_kl < 0.50 * config.target_kl:
                kl_coefficient = max(1e-5, kl_coefficient / 1.5)

        episode_returns = [item["log_return"] for item in infos]
        realised = {
            "cvar": abs(min(_empirical_cvar(episode_returns, config.cvar_alpha), 0.0)),
            "drawdown": abs(min(item["drawdown"] for item in infos)),
            "turnover": float(np.mean([item["turnover"] for item in infos])),
            "hhi": float(np.mean([item["hhi"] for item in infos])),
        }
        if config.use_cmdp:
            mean_limits = {
                name: float(np.mean(limit_paths[name]))
                for name in config.active_cmdp_constraints
            }
            controller.update(realised, mean_limits)

        return_numpy = returns.detach().cpu().numpy()
        value_numpy = old_value_tensor.detach().cpu().numpy()
        target_variance = float(np.var(return_numpy))
        explained_variance = (
            1.0 - float(np.var(return_numpy - value_numpy)) / target_variance
            if target_variance > 1e-12
            else np.nan
        )
        validation_rank: tuple[int, float, float, float] | None = None
        validation_metrics: dict[str, float] | None = None
        if (episode + 1) % max(1, config.validation_interval) == 0 or episode == config.ppo_episodes - 1:
            validation_rank, validation_metrics = _proposal_validation(
                core, agent, market, prepared, teacher_path, config, device
            )
            if best_rank is None or validation_rank > best_rank:
                best_rank = validation_rank
                best_state = copy.deepcopy(agent.state_dict())
                best_controller = controller.state_dict()
                no_improvement = 0
            else:
                no_improvement += 1

        row: dict[str, Any] = {
            "Policy": label,
            "Stage": "ppo",
            "Episode": episode,
            "Start": episode_start,
            "Mean reward": float(np.mean(rewards)),
            "Final wealth": portfolio.wealth,
            "Critic explained variance": explained_variance,
            "Validation feasible": validation_rank[0] if validation_rank is not None else np.nan,
            "Validation violation score": validation_rank[1] if validation_rank is not None else np.nan,
            "Validation annualised return": validation_metrics["return"] if validation_metrics is not None else np.nan,
            "Validation return LCB": validation_metrics["return_lcb"] if validation_metrics is not None else np.nan,
            "Validation Sharpe": validation_metrics["sharpe"] if validation_metrics is not None else np.nan,
            "KL coefficient": kl_coefficient,
            "Stop for KL": stop_for_kl,
            **{f"Lambda {name}": value for name, value in controller.state_dict().items()},
        }
        diagnostic_keys = (
            "Loss",
            "Policy loss",
            "Value loss",
            "Quantile loss",
            "Cost value loss",
            "Direct utility loss",
            "Expert auxiliary loss",
            "Gate auxiliary loss",
            "Approximate KL",
            "Clip fraction",
            "Gradient norm",
            "Mean optimiser-base reliance",
            "Mean temporal-return gate",
            "Mean risk-expert usage",
            "Mean predictive-return gate",
        )
        for key in diagnostic_keys:
            row[key] = (
                float(np.mean([float(item[key]) for item in update_diagnostics]))
                if update_diagnostics
                else np.nan
            )
        rows.append(row)
        if no_improvement >= config.validation_patience:
            break

    agent.load_state_dict(best_state)
    controller.load_state_dict(best_controller)
    agent.eval()
    return agent, pd.DataFrame(rows), time.perf_counter() - clock, controller.state_dict()

def train_xgat_agent(
    core: ModuleType,
    market: MarketData,
    prepared: PreparedData,
    teacher_path: np.ndarray,
    config: SimulationConfig,
    device: torch.device,
    seed: int,
) -> tuple[Any, pd.DataFrame, float, dict[str, float]]:
    return train_policy_agent(
        core,
        market,
        prepared,
        teacher_path,
        config,
        device,
        seed,
        label="X-GAT-DRL",
        use_risk_graph=config.use_risk_graph,
        use_predictive_graph=config.use_predictive_graph,
        temporal_mode="lstm",
    )


def evaluate_methods(
    core: ModuleType,
    market: MarketData,
    prepared: PreparedData,
    agents: Mapping[str, Any],
    bundle: ModelBundle,
    classical: Mapping[str, Any],
    teacher_path: np.ndarray,
    config: SimulationConfig,
    device: torch.device,
) -> EvaluationResult:
    states = {method: PortfolioState(_initial_weights(config)) for method in METHOD_ORDER}
    paths: dict[str, dict[str, list[Any]]] = {
        field: {method: [] for method in METHOD_ORDER}
        for field in ("returns", "turnover", "costs", "weights", "hhi", "cash_weights")
    }
    for agent in agents.values():
        agent.eval()
    bundle.glasso_gat.eval()
    bundle.tcmac.eval()

    for t in range(config.train_end, config.n_samples):
        expert_states = {method: states[method] for method in EXPERT_ORDER}
        expert_actions = _expert_actions(
            core, bundle, classical, prepared, expert_states, t, config, device
        )
        probability_index = max(prepared.hmm_start, t - 1)
        previous_index = max(prepared.hmm_start, probability_index - 1)
        probability = prepared.hmm_probabilities[probability_index]
        sequence = prepared.features[t : t + 1]
        risk_adjacency = prepared.dynamic_risk_adjacencies[t : t + 1]
        predictive_adjacency = prepared.dynamic_predictive_adjacencies[t : t + 1]
        policy_actions: dict[str, np.ndarray] = {}
        for label, agent in agents.items():
            external = _external_state(
                states[label],
                probability,
                prepared.hmm_probabilities[previous_index],
                device,
            )
            bounds_tensor, bounds = _cash_bounds_for_state(
                core,
                agent,
                sequence,
                risk_adjacency,
                predictive_adjacency,
                external,
                states[label],
                probability,
                config,
                device,
            )
            raw_base = teacher_path[t] if config.use_benchmark_anchor else _initial_weights(config)
            base = _enforce_cash_interval(raw_base, *bounds)
            base_tensor = torch.as_tensor(base, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action = agent.deterministic_action(
                    sequence,
                    risk_adjacency,
                    predictive_adjacency,
                    external,
                    base_tensor,
                    bounds_tensor,
                ).squeeze(0).cpu().numpy()
            policy_actions[label] = _safe_weights(core, action, config)
        actions = {**policy_actions, **expert_actions}

        for method in METHOD_ORDER:
            next_state, info = _apply_action(
                states[method], actions[method], market.log_returns[t], config
            )
            states[method] = next_state
            paths["returns"][method].append(info["log_return"])
            paths["turnover"][method].append(info["turnover"])
            paths["costs"][method].append(info["transaction_cost"])
            paths["weights"][method].append(actions[method].copy())
            paths["hhi"][method].append(info["hhi"])
            paths["cash_weights"][method].append(info["cash_weight"])

    return EvaluationResult(
        returns={method: np.asarray(paths["returns"][method]) for method in METHOD_ORDER},
        turnover={method: np.asarray(paths["turnover"][method]) for method in METHOD_ORDER},
        costs={method: np.asarray(paths["costs"][method]) for method in METHOD_ORDER},
        weights={method: np.asarray(paths["weights"][method]) for method in METHOD_ORDER},
        hhi={method: np.asarray(paths["hhi"][method]) for method in METHOD_ORDER},
        cash_weights={method: np.asarray(paths["cash_weights"][method]) for method in METHOD_ORDER},
    )


def _maximum_drawdown(log_returns: np.ndarray) -> float:
    wealth = np.concatenate([[1.0], np.exp(np.cumsum(log_returns))])
    peaks = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peaks - 1.0))


def build_metrics_table(
    core: ModuleType,
    evaluation: EvaluationResult,
    config: SimulationConfig,
) -> pd.DataFrame:
    """Compute frequency-consistent PSR and multiplicity-adjusted DSR."""
    statistics: dict[str, dict[str, Any]] = {}
    daily_sharpes: list[float] = []
    for method in METHOD_ORDER:
        returns = np.asarray(evaluation.returns[method], dtype=float)
        if returns.ndim != 1 or returns.size < 3 or not np.all(np.isfinite(returns)):
            raise ValueError(f"Invalid out-of-sample return series for {method}.")
        mean = float(np.mean(returns))
        volatility = float(np.std(returns, ddof=1))
        daily_sharpe = mean / volatility if volatility > 1e-12 else 0.0
        annualised_return = math.exp(mean * config.periods_per_year) - 1.0
        annualised_volatility = volatility * math.sqrt(config.periods_per_year)
        annualised_sharpe = daily_sharpe * math.sqrt(config.periods_per_year)
        daily_sharpes.append(daily_sharpe)
        downside = returns[returns < 0.0]
        downside_deviation = float(np.sqrt(np.mean(downside**2))) if downside.size else 0.0
        maximum_drawdown = _maximum_drawdown(returns)
        series = pd.Series(returns)
        statistics[method] = {
            "returns": returns, "annualised_return": annualised_return,
            "annualised_volatility": annualised_volatility,
            "daily_sharpe": daily_sharpe, "annualised_sharpe": annualised_sharpe,
            "sortino": mean / downside_deviation * math.sqrt(config.periods_per_year) if downside_deviation > 1e-12 else 0.0,
            "maximum_drawdown": maximum_drawdown,
            "calmar": annualised_return / abs(maximum_drawdown) if maximum_drawdown < -1e-12 else 0.0,
            "var": float(np.quantile(returns, 0.05)), "cvar": _empirical_cvar(returns),
            "skewness": float(series.skew()), "excess_kurtosis": float(series.kurt()),
            "final_wealth": float(math.exp(np.sum(returns))),
        }

    trials = np.asarray(daily_sharpes, dtype=float)
    effective_trials = float(len(METHOD_ORDER) if config.dsr_effective_trials is None else config.dsr_effective_trials)
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        item = statistics[method]
        kurtosis = item["excess_kurtosis"] + 3.0
        psr_zero = core.probabilistic_sharpe_ratio(
            item["daily_sharpe"], 0.0, item["returns"].size, item["skewness"], kurtosis
        )
        dsr, dsr_benchmark = core.deflated_sharpe_ratio(
            item["daily_sharpe"], trials, item["returns"].size, item["skewness"], kurtosis,
            effective_trials=effective_trials, return_benchmark=True,
        )
        rows.append(
            {
                "Method": method,
                "Annualised return": item["annualised_return"],
                "Annualised volatility": item["annualised_volatility"],
                "Sharpe ratio": item["annualised_sharpe"],
                "Daily Sharpe ratio": item["daily_sharpe"],
                "Sortino ratio": item["sortino"],
                "Maximum drawdown": item["maximum_drawdown"],
                "Calmar ratio": item["calmar"],
                "Daily VaR 95%": item["var"],
                "Daily CVaR 95%": item["cvar"],
                "Skewness": item["skewness"],
                "Excess kurtosis": item["excess_kurtosis"],
                "Final wealth": item["final_wealth"],
                "Mean turnover": float(np.mean(evaluation.turnover[method])),
                "Total transaction cost": float(np.sum(evaluation.costs[method])),
                "Mean HHI": float(np.mean(evaluation.hhi[method])),
                "Mean cash weight": float(np.mean(evaluation.cash_weights[method])),
                "Probabilistic Sharpe probability vs zero": psr_zero,
                "Deflated Sharpe probability": dsr,
                "Deflated Sharpe benchmark": dsr_benchmark * math.sqrt(config.periods_per_year),
                "DSR effective trials": effective_trials,
            }
        )
    return pd.DataFrame(rows)



def _graph_recovery_metrics(
    market: MarketData,
    prepared: PreparedData,
    config: SimulationConfig,
) -> dict[str, float]:
    true_normal = np.asarray(market.parameters["normal_lead_lag"], dtype=float)
    true_crisis = np.asarray(market.parameters["crisis_lead_lag"], dtype=float)
    graph_time = prepared.fit_end - 1
    # dynamic_predictive_adjacencies[t] is mixed with the filtered regime
    # probability available at t - 1.  Use that exact probability here so the
    # truth, edge mask, signs, and strengths all refer to the same causal graph.
    probability_index = max(prepared.hmm_start, graph_time - 1)
    probability = prepared.hmm_probabilities[probability_index]
    truth = probability[0] * true_normal + probability[-1] * true_crisis
    lagged_estimate = prepared.dynamic_predictive_adjacencies[
        graph_time
    ].cpu().numpy().copy()
    discount = 1.0 / np.arange(1, lagged_estimate.shape[0] + 1, dtype=float)
    estimate = np.einsum("lij,l->ij", lagged_estimate, discount)
    np.fill_diagonal(truth, 0.0)
    np.fill_diagonal(estimate, 0.0)
    off_diagonal = ~np.eye(config.n_risk_assets, dtype=bool)
    true_mask = (np.abs(truth) > 1e-8) & off_diagonal
    estimated_mask = (np.abs(estimate) > 1e-8) & off_diagonal
    true_positive = int(np.sum(true_mask & estimated_mask))
    false_positive = int(np.sum((~true_mask) & estimated_mask & off_diagonal))
    false_negative = int(np.sum(true_mask & (~estimated_mask)))
    true_negative = int(np.sum((~true_mask) & (~estimated_mask) & off_diagonal))
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    accuracy = (true_positive + true_negative) / max(1, int(off_diagonal.sum()))
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    common = true_mask & estimated_mask
    sign_accuracy = (
        float(np.mean(np.sign(truth[common]) == np.sign(estimate[common])))
        if common.any() else float("nan")
    )
    possible_edges = max(1, int(off_diagonal.sum()))
    return {
        "True predictive edges": int(true_mask.sum()),
        "Estimated predictive edges": int(estimated_mask.sum()),
        "Predictive graph true positives": true_positive,
        "Predictive graph true negatives": true_negative,
        "Predictive graph false positives": false_positive,
        "Predictive graph false negatives": false_negative,
        "Predictive graph accuracy": float(accuracy),
        "Predictive graph precision": float(precision),
        "Predictive graph recall": float(recall),
        "Predictive graph F1": float(f1),
        "Predictive graph sign accuracy": sign_accuracy,
        "Predictive graph density": float(estimated_mask.sum() / possible_edges),
        "Predictive graph false-positive rate": float(
            false_positive / max(1, false_positive + true_negative)
        ),
    }


def run_replication(
    core: ModuleType,
    config: SimulationConfig,
    market_seed: int,
    policy_seed: int,
    device: torch.device,
    *,
    market: MarketData | None = None,
    prepared: PreparedData | None = None,
) -> dict[str, Any]:
    """Train stochastic methods for one policy seed on a fixed market path."""
    core.seed_everything(policy_seed, deterministic=True)
    selected_market = market if market is not None else generate_synthetic_market(config, market_seed)
    selected_prepared = prepared if prepared is not None else prepare_data(
        core, selected_market, config, market_seed, device
    )
    market = selected_market
    prepared = selected_prepared
    classical = build_classical_benchmarks(core, market, prepared, config, policy_seed)
    bundle = train_benchmark_models(core, market, prepared, config, device, policy_seed)
    policy_base_path = build_policy_base_path(classical, prepared, config)
    teacher_diagnostics = pd.DataFrame()
    if config.use_behaviour_cloning:
        teacher_path, teacher_diagnostics = build_causal_teacher_path(
            core, bundle, classical, market, prepared, config, device
        )
    else:
        teacher_path = policy_base_path

    xgat, xgat_diagnostics, xgat_time, xgat_multipliers = train_xgat_agent(
        core, market, prepared, teacher_path, config, device, policy_seed
    )
    agents: dict[str, Any] = {"X-GAT-DRL": xgat}
    training_frames = [xgat_diagnostics]
    training_times = {"X-GAT-DRL": xgat_time}
    multipliers: dict[str, Any] = {"X-GAT-DRL": xgat_multipliers}

    evaluation = evaluate_methods(
        core, market, prepared, agents, bundle, classical, teacher_path, config, device
    )
    performance = build_metrics_table(core, evaluation, config)
    losses = np.column_stack([-evaluation.returns[method] for method in METHOD_ORDER])
    block_length = min(config.mcs_block_length, max(2, losses.shape[0] // 2))
    survivors = core.compute_model_confidence_set(
        losses,
        alpha=config.mcs_alpha,
        block_length=block_length,
        bootstraps=config.mcs_bootstraps,
        random_state=market_seed * 10_000 + policy_seed,
    )
    identifiers = {
        "Scenario": config.scenario,
        "Market seed": market_seed,
        "Policy seed": policy_seed,
    }
    performance = performance.assign(**identifiers)
    performance["MCS survivor"] = performance["Method"].isin(
        [METHOD_ORDER[index] for index in survivors]
    )
    return {
        "performance": performance,
        "evaluation": evaluation,
        "market_parameters": market.parameters,
        "hmm_diagnostics": prepared.hmm_diagnostics,
        "graph_recovery": _graph_recovery_metrics(market, prepared, config),
        "policy_training": pd.concat(training_frames, ignore_index=True, sort=False).assign(**identifiers),
        "benchmark_training": bundle.training_diagnostics.assign(**identifiers),
        "teacher_training": teacher_diagnostics.assign(**identifiers),
        "training_times": {**training_times, **bundle.training_times},
        "lagrange_multipliers": multipliers,
        "regime_effective_samples": prepared.regime_effective_samples,
        "regime_glasso": prepared.report_glasso,
        "representative_risk_adjacency": prepared.dynamic_risk_adjacencies[-1].cpu().numpy(),
        "representative_predictive_adjacency": np.einsum(
            "lij,l->ij",
            prepared.dynamic_predictive_adjacencies[-1].cpu().numpy(),
            1.0 / np.arange(1, config.predictive_lags + 1, dtype=float),
        ),
        "identifiers": identifiers,
    }


def _validate_replication_result(
    result: Mapping[str, Any],
    config: SimulationConfig,
) -> None:
    """Reject incomplete or numerically invalid results before checkpointing.

    This deliberately validates the complete serialised payload, including the
    out-of-sample paths used by the final aggregation. A run should fail near
    the offending replication, not after hundreds of later replications.
    """
    required_keys = {
        "performance",
        "evaluation",
        "market_parameters",
        "hmm_diagnostics",
        "graph_recovery",
        "policy_training",
        "benchmark_training",
        "teacher_training",
        "training_times",
        "lagrange_multipliers",
        "regime_effective_samples",
        "regime_glasso",
        "representative_risk_adjacency",
        "representative_predictive_adjacency",
        "identifiers",
    }
    missing = sorted(required_keys - set(result))
    if missing:
        raise RuntimeError(f"Replication result is missing keys: {missing}")

    identifiers = result["identifiers"]
    if not isinstance(identifiers, Mapping):
        raise RuntimeError("Replication identifiers must be a mapping.")
    for key in ("Scenario", "Market seed", "Policy seed"):
        if key not in identifiers:
            raise RuntimeError(f"Replication identifiers are missing {key!r}.")

    performance = result["performance"]
    if not isinstance(performance, pd.DataFrame) or performance.empty:
        raise RuntimeError("Replication performance table is empty or invalid.")
    if "Method" not in performance or set(performance["Method"]) != set(METHOD_ORDER):
        raise RuntimeError("Replication performance table has an invalid method set.")
    if performance["Method"].duplicated().any():
        raise RuntimeError("Replication performance table contains duplicate methods.")

    evaluation = result["evaluation"]
    evaluation_fields = ("returns", "turnover", "costs", "weights", "hhi", "cash_weights")
    if any(not hasattr(evaluation, name) for name in evaluation_fields):
        raise RuntimeError(
            "Replication evaluation payload does not expose the required fields."
        )
    expected_length = config.n_samples - config.train_end
    for field_name in ("returns", "turnover", "costs", "hhi", "cash_weights"):
        field = getattr(evaluation, field_name)
        if set(field) != set(METHOD_ORDER):
            raise RuntimeError(f"Evaluation field {field_name!r} has an invalid method set.")
        for method in METHOD_ORDER:
            values = np.asarray(field[method], dtype=float)
            if values.shape != (expected_length,):
                raise RuntimeError(
                    f"{field_name}[{method!r}] has shape {values.shape}; "
                    f"expected {(expected_length,)}."
                )
            if not np.all(np.isfinite(values)):
                raise RuntimeError(f"{field_name}[{method!r}] contains non-finite values.")

    if set(evaluation.weights) != set(METHOD_ORDER):
        raise RuntimeError("Evaluation weights have an invalid method set.")
    for method in METHOD_ORDER:
        weights = np.asarray(evaluation.weights[method], dtype=float)
        expected_shape = (expected_length, config.n_assets)
        if weights.shape != expected_shape:
            raise RuntimeError(
                f"weights[{method!r}] has shape {weights.shape}; expected {expected_shape}."
            )
        if not np.all(np.isfinite(weights)):
            raise RuntimeError(f"weights[{method!r}] contains non-finite values.")
        if np.any(weights < -1e-7):
            raise RuntimeError(f"weights[{method!r}] contains negative target weights.")
        if not np.allclose(weights.sum(axis=1), 1.0, atol=2e-6, rtol=0.0):
            raise RuntimeError(f"weights[{method!r}] do not sum to one.")
        if np.any(weights[:, -1] > config.maximum_cash + 2e-6):
            raise RuntimeError(f"weights[{method!r}] exceed maximum_cash.")

    for table_name in ("policy_training", "benchmark_training", "teacher_training"):
        table = result[table_name]
        if not isinstance(table, pd.DataFrame):
            raise RuntimeError(f"{table_name} must be a pandas DataFrame.")
    if result["policy_training"].empty:
        raise RuntimeError("policy_training is unexpectedly empty.")
    if result["benchmark_training"].empty:
        raise RuntimeError("benchmark_training is unexpectedly empty.")

    risk_adjacency = np.asarray(result["representative_risk_adjacency"], dtype=float)
    predictive_adjacency = np.asarray(
        result["representative_predictive_adjacency"], dtype=float
    )
    expected_graph_shape = (config.n_risk_assets, config.n_risk_assets)
    for name, graph in (
        ("representative_risk_adjacency", risk_adjacency),
        ("representative_predictive_adjacency", predictive_adjacency),
    ):
        if graph.shape != expected_graph_shape or not np.all(np.isfinite(graph)):
            raise RuntimeError(
                f"{name} must be finite with shape {expected_graph_shape}; "
                f"received {graph.shape}."
            )

def _aggregate_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in metrics.select_dtypes(include=[np.number]).columns
        if column not in {"Market seed", "Policy seed"}
    ]
    rows: list[dict[str, Any]] = []
    for (scenario, method), frame in metrics.groupby(["Scenario", "Method"], sort=False):
        row: dict[str, Any] = {
            "Scenario": scenario,
            "Method": method,
            "Replications": int(frame.shape[0]),
            "MCS survival rate": float(frame["MCS survivor"].mean()),
        }
        for column in numeric_columns:
            values = frame[column].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            standard_error = float(np.std(values, ddof=1) / math.sqrt(values.size)) if values.size > 1 else 0.0
            row[f"{column} mean"] = float(np.mean(values))
            row[f"{column} median"] = float(np.median(values))
            row[f"{column} std"] = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            row[f"{column} CI low"] = float(np.mean(values) - 1.96 * standard_error)
            row[f"{column} CI high"] = float(np.mean(values) + 1.96 * standard_error)
            row[f"{column} q10"] = float(np.quantile(values, 0.10))
            row[f"{column} q90"] = float(np.quantile(values, 0.90))
        rows.append(row)
    return pd.DataFrame(rows)


def _wealth_paths_by_scenario(
    results: Sequence[dict[str, Any]],
) -> dict[str, dict[str, np.ndarray]]:
    output: dict[str, dict[str, np.ndarray]] = {}
    scenarios = list(dict.fromkeys(result["identifiers"]["Scenario"] for result in results))
    for scenario in scenarios:
        scenario_results = [result for result in results if result["identifiers"]["Scenario"] == scenario]
        output[scenario] = {}
        market_seeds = sorted({result["identifiers"]["Market seed"] for result in scenario_results})
        for method in METHOD_ORDER:
            market_paths = []
            for market_seed in market_seeds:
                replications = [
                    result
                    for result in scenario_results
                    if result["identifiers"]["Market seed"] == market_seed
                ]
                wealth = np.stack(
                    [
                        np.exp(np.cumsum(result["evaluation"].returns[method]))
                        for result in replications
                    ]
                )
                market_paths.append(wealth.mean(axis=0))
            output[scenario][method] = np.stack(market_paths)
    return output


def _concise_performance_table(hierarchical: pd.DataFrame) -> pd.DataFrame:
    selected = {
        "Annualised return": "Annualised return",
        "Sharpe ratio": "Sharpe ratio",
        "Daily CVaR 95%": "Daily CVaR 95%",
        "Maximum drawdown": "Maximum drawdown",
        "Mean turnover": "Mean turnover",
    }
    frame = hierarchical.loc[hierarchical["Metric"].isin(selected)].copy()
    table = frame.pivot(index=["Scenario", "Method"], columns="Metric", values="IQM").reset_index()
    scenario_order = list(dict.fromkeys(hierarchical["Scenario"].astype(str)))
    table["Scenario"] = pd.Categorical(table["Scenario"], categories=scenario_order, ordered=True)
    table["Method"] = pd.Categorical(table["Method"], categories=METHOD_ORDER, ordered=True)
    table = table.sort_values(["Scenario", "Method"]).reset_index(drop=True)
    table["Scenario"] = table["Scenario"].astype(str)
    table["Method"] = table["Method"].astype(str)
    return table[["Scenario", "Method", *selected.values()]]


def _formatted_performance_table(table: pd.DataFrame) -> pd.DataFrame:
    output = table.copy()
    for column in ("Annualised return", "Daily CVaR 95%", "Maximum drawdown", "Mean turnover"):
        output[column] = output[column].map(lambda value: f"{100.0 * value:.2f}%")
    output["Sharpe ratio"] = output["Sharpe ratio"].map(lambda value: f"{value:.3f}")
    return output


def _graph_recovery_interval_summary(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    confidence_level: float,
    random_state: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    alpha = 0.5 * (1.0 - confidence_level)
    rows: list[dict[str, Any]] = []
    for scenario, group in frame.groupby("Scenario", sort=False):
        for metric in (
            "Predictive graph accuracy",
            "Predictive graph precision",
            "Predictive graph recall",
            "Predictive graph F1",
            "Predictive graph sign accuracy",
            "Predictive graph false-positive rate",
            "Predictive graph false positives",
            "Estimated predictive edges",
        ):
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            draws = np.empty(max(1, int(n_bootstrap)), dtype=float)
            for index in range(draws.size):
                draws[index] = np.mean(rng.choice(values, size=values.size, replace=True))
            rows.append({
                "Scenario": scenario,
                "Metric": metric,
                "Independent markets": int(values.size),
                "Estimate": float(np.mean(values)),
                "Confidence low": float(np.quantile(draws, alpha)),
                "Confidence high": float(np.quantile(draws, 1.0 - alpha)),
            })
    return pd.DataFrame(rows)


def _hmm_interval_summary(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    confidence_level: float,
    random_state: int,
) -> pd.DataFrame:
    enriched = frame.copy()
    def crisis_precision(value: Any) -> float:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return float("nan")
        matrix = np.asarray(value, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return float("nan")
        true_positive = matrix[-1, -1]
        false_positive = matrix[:-1, -1].sum()
        return float(true_positive / max(1.0, true_positive + false_positive))
    enriched["crisis_precision"] = enriched["confusion_matrix"].map(crisis_precision)
    rng = np.random.default_rng(random_state)
    alpha = 0.5 * (1.0 - confidence_level)
    metrics = {"state_accuracy": "HMM state accuracy", "crisis_precision": "HMM crisis precision"}
    rows: list[dict[str, Any]] = []
    for scenario, group in enriched.groupby("Scenario", sort=False):
        for source, label in metrics.items():
            values = group[source].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            draws = np.asarray([
                np.mean(rng.choice(values, size=values.size, replace=True))
                for _ in range(max(1, int(n_bootstrap)))
            ])
            rows.append({
                "Scenario": scenario, "Metric": label,
                "Independent markets": int(values.size),
                "Estimate": float(np.mean(values)),
                "Confidence low": float(np.quantile(draws, alpha)),
                "Confidence high": float(np.quantile(draws, 1.0 - alpha)),
            })
    return pd.DataFrame(rows)



def _hmm_detection_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a concise scenario-level HMM diagnostic table."""
    enriched = frame.copy()

    def _confusion_rates(value: Any) -> tuple[float, float]:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return float("nan"), float("nan")
        matrix = np.asarray(value, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return float("nan"), float("nan")
        true_positive = float(matrix[-1, -1])
        false_positive = float(matrix[:-1, -1].sum())
        false_negative = float(matrix[-1, :-1].sum())
        precision = true_positive / max(1.0, true_positive + false_positive)
        recall = true_positive / max(1.0, true_positive + false_negative)
        return precision, recall

    rates = enriched["confusion_matrix"].map(_confusion_rates)
    enriched["Crisis precision"] = rates.map(lambda pair: pair[0])
    enriched["Crisis recall"] = rates.map(lambda pair: pair[1])
    summary = (
        enriched.groupby("Scenario", as_index=False)
        .agg(
            {
                "state_accuracy": "mean",
                "balanced_accuracy": "mean",
                "brier_score": "mean",
                "Crisis precision": "mean",
                "Crisis recall": "mean",
                "converged": "mean",
            }
        )
        .rename(
            columns={
                "state_accuracy": "HMM state accuracy",
                "balanced_accuracy": "HMM balanced accuracy",
                "brier_score": "HMM Brier score",
                "converged": "Convergence rate",
            }
        )
    )
    return summary


def _policy_training_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise PPO optimisation diagnostics without pooling pretraining rows."""
    ppo = frame.loc[frame["Stage"].astype(str).str.lower() == "ppo"].copy()
    columns = [
        "Critic explained variance",
        "Approximate KL",
        "Stop for KL",
        "Clip fraction",
        "Gradient norm",
        "Mean optimiser-base reliance",
        "Mean temporal-return gate",
        "Mean predictive-return gate",
    ]
    available = [column for column in columns if column in ppo.columns]
    if ppo.empty or not available:
        return pd.DataFrame(columns=["Scenario", *available])
    return ppo.groupby("Scenario", as_index=False)[available].mean(numeric_only=True)


def _decision_mechanism_summary(
    policy_summary: pd.DataFrame,
    market_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Join route usage with out-of-sample cash and concentration."""
    exposure = (
        market_metrics.loc[market_metrics["Method"] == "X-GAT-DRL"]
        .groupby("Scenario", as_index=False)[["Mean cash weight", "Mean HHI"]]
        .mean(numeric_only=True)
    )
    if policy_summary.empty:
        return exposure
    selected = [
        column
        for column in (
            "Scenario",
            "Mean optimiser-base reliance",
            "Mean temporal-return gate",
            "Mean predictive-return gate",
        )
        if column in policy_summary.columns
    ]
    return policy_summary[selected].merge(exposure, on="Scenario", how="outer")


def _save_market_path_tables(
    returns_by_method: Mapping[str, np.ndarray],
    xgat_weights: np.ndarray,
    *,
    asset_names: Sequence[str],
    scenario_dir: Path,
) -> None:
    """Save the numerical paths underlying median-market diagnostic plots."""
    wealth = {
        method: np.exp(np.cumsum(np.asarray(values, dtype=float)))
        for method, values in returns_by_method.items()
    }
    drawdowns = {
        method: values / np.maximum.accumulate(values) - 1.0
        for method, values in wealth.items()
    }
    pd.DataFrame(wealth).rename_axis("Time step").reset_index().to_csv(
        scenario_dir / "median_market_wealth.csv", index=False
    )
    pd.DataFrame(drawdowns).rename_axis("Time step").reset_index().to_csv(
        scenario_dir / "median_market_drawdowns.csv", index=False
    )
    pd.DataFrame(
        np.asarray(xgat_weights, dtype=float),
        columns=list(asset_names),
    ).rename_axis("Time step").reset_index().to_csv(
        scenario_dir / "median_market_xgat_allocation.csv", index=False
    )


def _write_output_manifests(root: Path, figure_dir: Path) -> None:
    """Write inventories so every reported table and figure is traceable."""
    table_descriptions = {
        "all_replication_performance.csv": "All policy-level performance results.",
        "market_level_performance.csv": "Policy seeds averaged within each market.",
        "performance_summary.csv": "Concise IQM performance table.",
        "performance_summary_formatted.csv": "Formatted concise performance table.",
        "performance_summary_detailed.csv": "Detailed descriptive performance summary.",
        "hierarchical_metric_summary.csv": "Market-bootstrap IQMs and confidence intervals.",
        "hierarchical_paired_comparisons.csv": "Paired market-level comparisons with Holm correction.",
        "graph_recovery_metrics.csv": "Market-level predictive-graph recovery metrics.",
        "graph_recovery_intervals.csv": "Bootstrap intervals for graph-recovery metrics.",
        "graph_recovery_summary.csv": "Scenario-level graph-recovery means.",
        "hmm_diagnostics.csv": "Market-level HMM diagnostics.",
        "hmm_detection_intervals.csv": "Bootstrap intervals for HMM detection metrics.",
        "hmm_detection_summary.csv": "Concise scenario-level HMM summary.",
        "policy_training_diagnostics.csv": "Full policy-training diagnostics.",
        "ppo_training_summary.csv": "Concise PPO diagnostic summary.",
        "xgat_decision_mechanism_summary.csv": "Gate, optimiser, cash, and concentration summary.",
        "benchmark_training_diagnostics.csv": "Benchmark validation and training diagnostics.",
        "representative_market_selection.csv": "Median-market selection rule and chosen market.",
    }
    table_rows = []
    for path in sorted(root.glob("*.csv")):
        table_rows.append(
            {
                "File": path.name,
                "Description": table_descriptions.get(path.name, "Generated analysis table."),
            }
        )
    pd.DataFrame(table_rows).to_csv(root / "table_manifest.csv", index=False)

    figure_rows = []
    for path in sorted(figure_dir.rglob("*.png")):
        relative = path.relative_to(root).as_posix()
        role = "Supplementary diagnostic"
        if path.name in {
            "interval_daily_cvar_95.png",
            "interval_maximum_drawdown.png",
            "return_cvar_tradeoff_iqm_facets.png",
            "drawdown_facets.png",
            "graph_recovery_diagnostics.png",
        }:
            role = "Main-text candidate"
        elif path.name in {
            "median_market_wealth.png",
            "median_market_drawdowns.png",
            "median_market_xgat_allocation.png",
            "xgat_allocation.png",
            "risk_graph.png",
            "predictive_graph.png",
        }:
            role = "Representative-market diagnostic"
        figure_rows.append(
            {
                "File": relative,
                "Role": role,
            }
        )
    pd.DataFrame(figure_rows).to_csv(root / "figure_manifest.csv", index=False)


def aggregate_and_save(
    core: ModuleType,
    results: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    config: SimulationConfig,
    root: Path,
    api_audit: Mapping[str, Any],
    *,
    hard_exit: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    cross_scenario_dir = figure_dir / "cross_scenario"
    cross_scenario_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = pd.concat([result["performance"] for result in results], ignore_index=True)
    market_metrics = (
        all_metrics.groupby(["Scenario", "Method", "Market seed"], as_index=False)
        .mean(numeric_only=True)
    )
    detailed_summary = _aggregate_summary(all_metrics)
    metric_columns = (
        "Annualised return", "Sharpe ratio", "Sortino ratio", "Maximum drawdown",
        "Daily CVaR 95%", "Mean turnover", "Mean HHI", "Final wealth",
    )
    hierarchical = core.hierarchical_metric_summary(
        all_metrics, metric_columns=metric_columns,
        n_bootstrap=config.hierarchical_bootstraps,
        confidence_level=config.confidence_level, random_state=17,
    )
    directions = {
        "Annualised return": True, "Sharpe ratio": True, "Sortino ratio": True,
        "Maximum drawdown": True, "Daily CVaR 95%": True,
        "Mean turnover": False, "Mean HHI": False, "Final wealth": True,
    }
    paired = core.hierarchical_paired_comparisons(
        all_metrics, strategy="X-GAT-DRL", benchmarks=METHOD_ORDER[1:],
        metric_directions=directions, n_bootstrap=config.hierarchical_bootstraps,
        confidence_level=config.confidence_level,
        sign_flip_repetitions=config.sign_flip_repetitions, random_state=29,
    )
    concise = _concise_performance_table(hierarchical)
    formatted = _formatted_performance_table(concise)

    all_metrics.to_csv(root / "all_replication_performance.csv", index=False)
    market_metrics.to_csv(root / "market_level_performance.csv", index=False)
    concise.to_csv(root / "performance_summary.csv", index=False)
    formatted.to_csv(root / "performance_summary_formatted.csv", index=False)
    detailed_summary.to_csv(root / "performance_summary_detailed.csv", index=False)
    hierarchical.to_csv(root / "hierarchical_metric_summary.csv", index=False)
    paired.to_csv(root / "hierarchical_paired_comparisons.csv", index=False)
    for scenario in config.scenarios:
        formatted.loc[formatted["Scenario"] == scenario].to_csv(
            root / f"performance_table_{_safe_filename(scenario)}.csv", index=False
        )

    hmm_frame = pd.DataFrame(
        [{**result["identifiers"], **result["hmm_diagnostics"]} for result in results]
    ).drop_duplicates(["Scenario", "Market seed"]).drop(columns="Policy seed")
    graph_frame = pd.DataFrame(
        [{**result["identifiers"], **result["graph_recovery"]} for result in results]
    ).drop_duplicates(["Scenario", "Market seed"]).drop(columns="Policy seed")
    graph_intervals = _graph_recovery_interval_summary(
        graph_frame, n_bootstrap=config.hierarchical_bootstraps,
        confidence_level=config.confidence_level, random_state=41,
    )
    hmm_intervals = _hmm_interval_summary(
        hmm_frame, n_bootstrap=config.hierarchical_bootstraps,
        confidence_level=config.confidence_level, random_state=43,
    )
    hmm_frame.to_csv(root / "hmm_diagnostics.csv", index=False)
    graph_frame.to_csv(root / "graph_recovery_metrics.csv", index=False)
    graph_intervals.to_csv(root / "graph_recovery_intervals.csv", index=False)
    hmm_intervals.to_csv(root / "hmm_detection_intervals.csv", index=False)

    graph_summary = (
        graph_frame.groupby("Scenario", as_index=False)
        .mean(numeric_only=True)
    )
    hmm_summary = _hmm_detection_summary(hmm_frame)
    policy_training = pd.concat(
        [result["policy_training"] for result in results],
        ignore_index=True,
        sort=False,
    )
    ppo_summary = _policy_training_summary(policy_training)
    decision_summary = _decision_mechanism_summary(ppo_summary, market_metrics)

    graph_summary.to_csv(root / "graph_recovery_summary.csv", index=False)
    hmm_summary.to_csv(root / "hmm_detection_summary.csv", index=False)
    policy_training.to_csv(root / "policy_training_diagnostics.csv", index=False)
    ppo_summary.to_csv(root / "ppo_training_summary.csv", index=False)
    decision_summary.to_csv(
        root / "xgat_decision_mechanism_summary.csv",
        index=False,
    )

    pd.concat([result["benchmark_training"] for result in results], ignore_index=True, sort=False).to_csv(
        root / "benchmark_training_diagnostics.csv", index=False
    )
    teacher_frames = [result["teacher_training"] for result in results if not result["teacher_training"].empty]
    if teacher_frames:
        pd.concat(teacher_frames, ignore_index=True, sort=False).to_csv(
            root / "teacher_mixture_diagnostics.csv", index=False
        )
    failure_path = root / "failed_replications.csv"
    if failures:
        pd.DataFrame(failures).to_csv(failure_path, index=False)
    elif failure_path.exists():
        # Do not let a failed file from an earlier interrupted run survive a
        # later clean run in the same output directory.
        failure_path.unlink()

    wealth_paths = _wealth_paths_by_scenario(results)
    metrics_for_figures = (
        "Annualised return", "Sharpe ratio", "Daily CVaR 95%",
        "Maximum drawdown", "Mean turnover",
    )
    for metric in metrics_for_figures:
        core.plot_grouped_scenario_boxplots(
            market_metrics, metric=metric, method_order=METHOD_ORDER,
            scenario_order=config.scenarios,
            save_path=cross_scenario_dir / f"boxplot_{_safe_filename(metric)}.png",
        )
        core.plot_scenario_metric_intervals(
            hierarchical, metric=metric,
            save_path=cross_scenario_dir / f"interval_{_safe_filename(metric)}.png",
        )
    core.plot_return_cvar_facets(
        concise,
        scenario_order=config.scenarios,
        method_order=METHOD_ORDER,
        save_path=cross_scenario_dir / "return_cvar_tradeoff_iqm_facets.png",
    )
    core.plot_wealth_facets(
        wealth_paths, scenario_order=config.scenarios, method_order=METHOD_ORDER,
        save_path=cross_scenario_dir / "cumulative_wealth_facets.png",
    )
    core.plot_drawdown_facets(
        wealth_paths, scenario_order=config.scenarios, method_order=METHOD_ORDER,
        save_path=cross_scenario_dir / "drawdown_facets.png",
    )
    if not graph_intervals.empty:
        core.plot_graph_recovery_intervals(
            graph_intervals, save_path=cross_scenario_dir / "graph_recovery_intervals.png"
        )
        core.plot_graph_recovery_diagnostics(
            graph_intervals,
            negative_control="covariance_only",
            positive_scenarios=tuple(
                scenario
                for scenario in ("graph_predictive", "tail_stress")
                if scenario in config.scenarios
            ),
            save_path=cross_scenario_dir / "graph_recovery_diagnostics.png",
        )
    if not hmm_intervals.empty:
        core.plot_accuracy_precision_intervals(
            hmm_intervals,
            accuracy_metric="HMM state accuracy",
            precision_metric="HMM crisis precision",
            title="HMM regime detection",
            save_path=cross_scenario_dir / "hmm_detection_intervals.png",
        )

    representative_rows: list[dict[str, Any]] = []
    for scenario in config.scenarios:
        scenario_dir = figure_dir / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_data_dir = root / "scenario_diagnostics" / _safe_filename(scenario)
        scenario_data_dir.mkdir(parents=True, exist_ok=True)

        scenario_metrics = all_metrics.loc[all_metrics["Scenario"] == scenario]
        scenario_market_metrics = market_metrics.loc[
            market_metrics["Scenario"] == scenario
        ]
        scenario_summary = (
            concise.loc[concise["Scenario"] == scenario]
            .set_index("Method")
            .reindex(METHOD_ORDER)
        )
        if (
            scenario not in wealth_paths
            or scenario_metrics.empty
            or scenario_summary.empty
        ):
            continue

        # Use the same IQM estimators as the primary performance table.
        core.plot_pareto_frontier(
            scenario_summary["Annualised return"].to_numpy(dtype=float),
            scenario_summary["Daily CVaR 95%"].to_numpy(dtype=float),
            scenario_summary["Mean turnover"].to_numpy(dtype=float),
            labels=list(METHOD_ORDER),
            save_path=scenario_dir / "return_cvar_tradeoff.png",
        )

        # Select the market whose policy-seed-averaged X-GAT Sharpe is the
        # sample median. All plotted returns and weights are then averaged
        # over the policy seeds for that market; no arbitrary policy seed is
        # used for the representative diagnostic.
        xgat_market = (
            scenario_market_metrics.loc[
                scenario_market_metrics["Method"] == "X-GAT-DRL",
                ["Market seed", "Sharpe ratio"],
            ]
            .set_index("Market seed")["Sharpe ratio"]
            .sort_values()
        )
        if xgat_market.empty:
            continue
        selected_market = int(xgat_market.index[len(xgat_market) // 2])
        selected_results = [
            result
            for result in results
            if result["identifiers"]["Scenario"] == scenario
            and result["identifiers"]["Market seed"] == selected_market
        ]
        if not selected_results:
            continue

        returns_by_method = {
            method: np.mean(
                np.stack(
                    [
                        np.asarray(
                            result["evaluation"].returns[method],
                            dtype=float,
                        )
                        for result in selected_results
                    ],
                    axis=0,
                ),
                axis=0,
            )
            for method in METHOD_ORDER
        }
        xgat_weights = np.mean(
            np.stack(
                [
                    np.asarray(
                        result["evaluation"].weights["X-GAT-DRL"],
                        dtype=float,
                    )
                    for result in selected_results
                ],
                axis=0,
            ),
            axis=0,
        )
        risk_adjacency = np.mean(
            np.stack(
                [
                    np.asarray(
                        result["representative_risk_adjacency"],
                        dtype=float,
                    )
                    for result in selected_results
                ],
                axis=0,
            ),
            axis=0,
        )
        predictive_adjacency = np.mean(
            np.stack(
                [
                    np.asarray(
                        result["representative_predictive_adjacency"],
                        dtype=float,
                    )
                    for result in selected_results
                ],
                axis=0,
            ),
            axis=0,
        )

        core.plot_cumulative_wealth(
            returns_by_method,
            title=(
                f"Median X-GAT market: {scenario}; "
                "policy seeds averaged within market"
            ),
            save_path=scenario_dir / "median_market_wealth.png",
        )
        core.plot_drawdowns(
            returns_by_method,
            title=(
                f"Median X-GAT market drawdowns: {scenario}; "
                "policy seeds averaged within market"
            ),
            save_path=scenario_dir / "median_market_drawdowns.png",
        )

        asset_names = [
            f"Risk Asset {index + 1}"
            for index in range(config.n_risk_assets)
        ] + ["Cash"]
        median_allocation_path = scenario_dir / "median_market_xgat_allocation.png"
        core.plot_allocation_area(
            xgat_weights,
            asset_names,
            save_path=median_allocation_path,
        )

        (scenario_dir / "xgat_allocation.png").write_bytes(
            median_allocation_path.read_bytes()
        )
        _save_market_path_tables(
            returns_by_method,
            xgat_weights,
            asset_names=asset_names,
            scenario_dir=scenario_data_dir,
        )

        graph_asset_names = [
            f"Asset {index + 1}"
            for index in range(config.n_risk_assets)
        ]
        core.plot_network_topology(
            risk_adjacency,
            graph_asset_names,
            title=f"Risk graph: {scenario}",
            save_path=scenario_dir / "risk_graph.png",
        )
        core.plot_network_topology(
            predictive_adjacency,
            graph_asset_names,
            title=f"Directed predictive graph: {scenario}",
            save_path=scenario_dir / "predictive_graph.png",
        )

        representative_rows.append(
            {
                "Scenario": scenario,
                "Selected market seed": selected_market,
                "Selection statistic": "Policy-seed-averaged X-GAT-DRL Sharpe ratio",
                "Selected-market statistic": float(xgat_market.loc[selected_market]),
                "Selection rule": "Sample median across independent market seeds",
                "Policy seeds averaged": int(len(selected_results)),
            }
        )

    pd.DataFrame(representative_rows).to_csv(
        root / "representative_market_selection.csv",
        index=False,
    )
    _write_output_manifests(root, figure_dir)

    provenance = {
        "core": dict(api_audit),
        "configuration": {**asdict(config), "output_dir": str(config.output_dir)},
        "methods": list(METHOD_ORDER),
        "independence_rule": "Policy seeds are averaged inside each market seed before inference.",
    }
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2, default=str), encoding="utf-8")
    summary_lines = [
        "# Simulation summary", "",
        f"- Successful replications: {len(results)}",
        f"- Failed replications: {len(failures)}",
        f"- Independent markets per scenario: {len(config.market_seeds)}",
        f"- Policy seeds per market: {len(config.policy_seeds)}",
        "- Statistical unit: market seed; policy seeds are averaged within market.",
        "", "## Performance table", "", "```text",
        formatted.to_string(index=False), "```", "",
        "## Hierarchical paired comparisons", "",
        "Positive direction-adjusted effects favour X-GAT-DRL.", "", "```text",
        paired[[
            "Scenario", "Metric", "Benchmark", "Independent paired markets",
            "Direction-adjusted effect", "Confidence low", "Confidence high",
            "Probability superior", "Holm-adjusted p-value",
        ]].to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        "```",
    ]
    summary_bytes = ("\n".join(summary_lines) + "\n").encode("utf-8")
    summary_path = root / "simulation_summary.md"
    _atomic_write_bytes(summary_path, summary_bytes)


def _safe_filename(value: str) -> str:
    return "_".join(
        part for part in "".join(character.lower() if character.isalnum() else "_" for character in value).split("_") if part
    )


def _utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically replace ``path`` and durably publish its directory entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(
            os.fspath(path.parent),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_bytes(
        path,
        (json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8"),
    )


def _atomic_write_pickle(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one compressed checkpoint without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as raw_stream:
            with gzip.GzipFile(
                fileobj=raw_stream,
                mode="wb",
                compresslevel=3,
                mtime=0,
            ) as compressed_stream:
                pickle.dump(
                    dict(payload),
                    compressed_stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            raw_stream.flush()
            os.fsync(raw_stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(
            os.fspath(path.parent),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint_signature(
    config: SimulationConfig,
    api_audit: Mapping[str, Any],
) -> dict[str, Any]:
    scientific_configuration = asdict(config)
    scientific_configuration["output_dir"] = str(config.output_dir.expanduser().resolve())
    script_path = Path(__file__).resolve()
    signature_payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "simulation_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "core_sha256": api_audit.get("sha256"),
        "configuration": scientific_configuration,
    }
    canonical = json.dumps(
        signature_payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return {
        **signature_payload,
        "fingerprint": hashlib.sha256(canonical).hexdigest(),
    }


def _replication_key(
    scenario: str,
    market_seed: int,
    policy_seed: int,
) -> tuple[str, int, int]:
    return str(scenario), int(market_seed), int(policy_seed)


def _expected_replication_keys(
    config: SimulationConfig,
) -> tuple[tuple[str, int, int], ...]:
    return tuple(
        _replication_key(scenario, market_seed, policy_seed)
        for scenario in config.scenarios
        for market_seed in config.market_seeds
        for policy_seed in config.policy_seeds
    )


def _checkpoint_path(
    checkpoint_dir: Path,
    key: tuple[str, int, int],
) -> Path:
    scenario, market_seed, policy_seed = key
    return checkpoint_dir / (
        f"{_safe_filename(scenario)}"
        f"__market_{market_seed}__policy_{policy_seed}.pkl.gz"
    )


def _acquire_run_lock(root: Path) -> Any:
    """Prevent concurrent writers from using the same output directory."""
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - the study server is Linux.
        raise RuntimeError("Checkpoint locking requires a POSIX-compatible system.") from exc

    lock_path = root / ".simulation.lock"
    lock_stream = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_stream.seek(0)
        owner = lock_stream.read().strip() or "unknown process"
        lock_stream.close()
        raise RuntimeError(
            "Another simulation is already using this output directory. "
            f"Lock owner: {owner}"
        ) from exc
    lock_stream.seek(0)
    lock_stream.truncate()
    lock_stream.write(
        json.dumps(
            {
                "pid": os.getpid(),
                "host": os.uname().nodename,
                "started_at_utc": _utc_timestamp(),
            },
            sort_keys=True,
        )
    )
    lock_stream.flush()
    os.fsync(lock_stream.fileno())
    return lock_stream


def _archive_checkpoint_directory(root: Path, checkpoint_dir: Path) -> Path | None:
    if not checkpoint_dir.exists():
        return None
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    archive = root / f"checkpoints_archived_{timestamp}"
    if archive.exists():
        archive = root / f"checkpoints_archived_{timestamp}_{os.getpid()}"
    checkpoint_dir.rename(archive)
    return archive


def _write_checkpoint_index(
    checkpoint_dir: Path,
    records: Mapping[tuple[str, int, int], Mapping[str, Any]],
    *,
    total_expected: int,
) -> None:
    ordered_records = [
        dict(records[key])
        for key in sorted(records, key=lambda item: (item[0], item[1], item[2]))
    ]
    manifest = pd.DataFrame(ordered_records)
    if manifest.empty:
        manifest = pd.DataFrame(
            columns=[
                "Scenario",
                "Market seed",
                "Policy seed",
                "Completed at UTC",
                "Checkpoint file",
                "Checkpoint bytes",
            ]
        )
    _atomic_write_bytes(
        checkpoint_dir / "checkpoint_manifest.csv",
        manifest.to_csv(index=False).encode("utf-8"),
    )
    _atomic_write_json(
        checkpoint_dir / "progress.json",
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "completed_replications": len(records),
            "total_expected_replications": total_expected,
            "completion_fraction": (
                len(records) / total_expected if total_expected else 0.0
            ),
            "updated_at_utc": _utc_timestamp(),
        },
    )


def _initialise_checkpoint_store(
    root: Path,
    config: SimulationConfig,
    api_audit: Mapping[str, Any],
    *,
    restart: bool,
    allow_compatible_code_change: bool = False,
) -> tuple[
    Path,
    dict[str, Any],
    dict[tuple[str, int, int], dict[str, Any]],
    dict[tuple[str, int, int], dict[str, Any]],
]:
    checkpoint_dir = root / "checkpoints"
    if restart:
        archived = _archive_checkpoint_directory(root, checkpoint_dir)
        if archived is not None:
            LOGGER.warning("Archived prior checkpoints at %s", archived)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    signature = _checkpoint_signature(config, api_audit)
    signature_path = checkpoint_dir / "run_signature.json"
    accepted_fingerprints = {signature["fingerprint"]}
    if signature_path.exists():
        try:
            stored_signature = json.loads(signature_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Unreadable checkpoint signature: {signature_path}"
            ) from exc
        stored_fingerprint = stored_signature.get("fingerprint")
        accepted_fingerprints.update(
            str(value)
            for value in stored_signature.get("compatible_checkpoint_fingerprints", [])
            if value
        )
        if stored_fingerprint != signature["fingerprint"]:
            same_schema = (
                stored_signature.get("schema_version") == signature["schema_version"]
            )
            stored_configuration = json.dumps(
                stored_signature.get("configuration"),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            current_configuration = json.dumps(
                signature["configuration"],
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            same_configuration = stored_configuration == current_configuration
            if not (
                allow_compatible_code_change
                and same_schema
                and same_configuration
                and stored_fingerprint
            ):
                raise RuntimeError(
                    "Existing checkpoints use a different script/core fingerprint. "
                    "If the scientific configuration is unchanged and this is a "
                    "reviewed bug-fix release, rerun with "
                    "--resume-compatible-checkpoints. Otherwise use a new "
                    "--output-dir or pass --restart."
                )
            accepted_fingerprints.add(str(stored_fingerprint))
            timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            previous_signature_path = checkpoint_dir / f"run_signature_{timestamp}.previous.json"
            _atomic_write_json(previous_signature_path, stored_signature)
            signature = {
                **signature,
                "compatible_checkpoint_fingerprints": sorted(
                    accepted_fingerprints - {signature["fingerprint"]}
                ),
                "adopted_previous_signature": previous_signature_path.name,
            }
            _atomic_write_json(signature_path, signature)
            LOGGER.warning(
                "Adopted compatible checkpoints created by an earlier code "
                "fingerprint; every recovered payload will be fully validated."
            )
        else:
            accepted_fingerprints.add(str(stored_fingerprint))
    else:
        _atomic_write_json(signature_path, signature)

    expected_keys = set(_expected_replication_keys(config))
    recovered: dict[tuple[str, int, int], dict[str, Any]] = {}
    records: dict[tuple[str, int, int], dict[str, Any]] = {}
    for path in sorted(checkpoint_dir.glob("*.pkl.gz")):
        try:
            with gzip.open(path, "rb") as stream:
                payload = pickle.load(stream)
        except Exception as exc:
            raise RuntimeError(
                f"Checkpoint is corrupt or unreadable: {path}. "
                "Do not delete it silently; use --restart to archive the checkpoint set."
            ) from exc
        if not isinstance(payload, dict):
            raise RuntimeError(f"Checkpoint payload is invalid: {path}")
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(f"Checkpoint schema is incompatible: {path}")
        if payload.get("fingerprint") not in accepted_fingerprints:
            raise RuntimeError(f"Checkpoint signature mismatch: {path}")
        identifiers = payload.get("identifiers")
        if not isinstance(identifiers, dict):
            raise RuntimeError(f"Checkpoint identifiers are invalid: {path}")
        key = _replication_key(
            identifiers.get("Scenario"),
            identifiers.get("Market seed"),
            identifiers.get("Policy seed"),
        )
        if key not in expected_keys:
            raise RuntimeError(f"Checkpoint is outside the configured design: {path}")
        result = payload.get("result")
        if not isinstance(result, dict) or result.get("identifiers") != identifiers:
            raise RuntimeError(f"Checkpoint result is invalid: {path}")
        try:
            _validate_replication_result(result, config)
        except Exception as exc:
            raise RuntimeError(
                f"Checkpoint result failed structural validation: {path}"
            ) from exc
        if key in recovered:
            raise RuntimeError(f"Duplicate checkpoint for {key}: {path}")
        recovered[key] = result
        records[key] = {
            **identifiers,
            "Completed at UTC": payload.get("completed_at_utc"),
            "Checkpoint file": path.name,
            "Checkpoint bytes": path.stat().st_size,
        }

    _write_checkpoint_index(
        checkpoint_dir,
        records,
        total_expected=len(expected_keys),
    )
    return checkpoint_dir, signature, recovered, records


def _save_replication_checkpoint(
    checkpoint_dir: Path,
    signature: Mapping[str, Any],
    key: tuple[str, int, int],
    result: dict[str, Any],
    records: dict[tuple[str, int, int], dict[str, Any]],
    *,
    total_expected: int,
    config: SimulationConfig,
) -> None:
    _validate_replication_result(result, config)
    identifiers = dict(result["identifiers"])
    expected_identifiers = {
        "Scenario": key[0],
        "Market seed": key[1],
        "Policy seed": key[2],
    }
    if identifiers != expected_identifiers:
        raise RuntimeError(
            f"Replication identifiers {identifiers} do not match {expected_identifiers}."
        )
    completed_at = _utc_timestamp()
    path = _checkpoint_path(checkpoint_dir, key)
    _atomic_write_pickle(
        path,
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": signature["fingerprint"],
            "completed_at_utc": completed_at,
            "identifiers": identifiers,
            "result": result,
        },
    )
    records[key] = {
        **identifiers,
        "Completed at UTC": completed_at,
        "Checkpoint file": path.name,
        "Checkpoint bytes": path.stat().st_size,
    }
    _write_checkpoint_index(
        checkpoint_dir,
        records,
        total_expected=total_expected,
    )


def _parse_csv_values(value: str | None, cast: Any) -> tuple[Any, ...] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return tuple(cast(item) for item in items)


def _clear_runtime_caches() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_failure_table(root: Path, failures: Sequence[Mapping[str, Any]]) -> None:
    path = root / "failed_replications.csv"
    if failures:
        frame = pd.DataFrame([dict(item) for item in failures])
        _atomic_write_bytes(path, frame.to_csv(index=False).encode("utf-8"))
    elif path.exists():
        path.unlink()


def _preflight_configuration(config: SimulationConfig, output_dir: Path) -> SimulationConfig:
    scenario = config.scenarios[0]
    return replace(
        config,
        scenario=scenario,
        scenarios=(scenario,),
        market_seeds=(config.market_seeds[0],),
        policy_seeds=(config.policy_seeds[0],),
        n_samples=420,
        validation_fraction=0.10,
        graph_window=120,
        fast_graph_window=60,
        graph_min_history=50,
        fast_graph_min_history=36,
        graph_update_interval=80,
        graph_alpha_refresh_interval=160,
        graph_bootstrap_replicates=1,
        rolling_graph_bootstrap_replicates=0,
        predictive_bootstrap_replicates=2,
        predictive_report_bootstrap_replicates=4,
        predictive_null_replicates=8,
        predictive_report_null_replicates=12,
        ppo_episodes=2,
        ppo_update_epochs=1,
        episode_length=48,
        behaviour_epochs=1,
        behaviour_batches_per_epoch=1,
        benchmark_epochs=1,
        benchmark_minimum_epochs=1,
        benchmark_validation_patience=1,
        critic_updates_per_actor=1,
        encoder_freeze_episodes=0,
        representation_pretrain_epochs=1,
        representation_batches_per_epoch=1,
        future_risk_horizon=10,
        batch_size=16,
        validation_interval=1,
        validation_patience=2,
        mcs_bootstraps=20,
        hierarchical_bootstraps=100,
        sign_flip_repetitions=100,
        output_dir=output_dir,
    )


def _run_adversarial_portfolio_preflight(
    core: ModuleType,
    config: SimulationConfig,
) -> None:
    """Exercise the exact drift condition that previously killed long runs."""
    constraints = _constraints(core, max_cash=config.maximum_cash)
    optimiser = core.ModelPredictiveControlOptimiser(
        config.n_assets,
        constraints=constraints,
        risk_aversion=5.0,
        turnover_penalty=0.005,
    )
    drifted = np.full(config.n_assets, 0.0, dtype=float)
    drifted[:-1] = 0.15 / config.n_risk_assets
    drifted[-1] = 0.85
    allocation = optimiser.allocate(
        np.zeros(config.n_assets, dtype=float),
        np.eye(config.n_assets, dtype=float) * 1e-4,
        drifted,
    )
    core.validate_weights(
        allocation,
        constraints=constraints,
        expected_assets=config.n_assets,
    )

    target = _initial_weights(config, cash_weight=config.maximum_cash)
    state, _ = _apply_action(
        PortfolioState(target.copy()),
        target,
        np.concatenate([
            np.full(config.n_risk_assets, -1.0, dtype=float),
            np.array([1.0], dtype=float),
        ]),
        config,
    )
    if state.weights[-1] <= config.maximum_cash:
        raise RuntimeError("Adversarial preflight failed to create cash-weight drift.")
    allocation = optimiser.allocate(
        np.zeros(config.n_assets, dtype=float),
        np.eye(config.n_assets, dtype=float) * 1e-4,
        state.weights,
    )
    core.validate_weights(
        allocation,
        constraints=constraints,
        expected_assets=config.n_assets,
    )


def _run_functional_preflight(
    core: ModuleType,
    config: SimulationConfig,
    device: torch.device,
    api_audit: Mapping[str, Any],
) -> None:
    """Run one reduced end-to-end replication including final aggregation."""
    temporary_root = Path(tempfile.mkdtemp(prefix="xgat_preflight_"))
    try:
        smoke = _preflight_configuration(config, temporary_root)
        smoke.validate()
        market_seed = smoke.market_seeds[0]
        policy_seed = smoke.policy_seeds[0]
        core.seed_everything(market_seed, deterministic=True)
        market = generate_synthetic_market(smoke, market_seed)
        prepared = prepare_data(core, market, smoke, market_seed, device)
        result = run_replication(
            core,
            smoke,
            market_seed,
            policy_seed,
            device,
            market=market,
            prepared=prepared,
        )
        _validate_replication_result(result, smoke)
        aggregate_and_save(
            core,
            [result],
            [],
            smoke,
            temporary_root,
            api_audit,
            hard_exit=False,
        )
        required_outputs = (
            "all_replication_performance.csv",
            "performance_summary_formatted.csv",
            "simulation_summary.md",
            "provenance.json",
        )
        missing = [name for name in required_outputs if not (temporary_root / name).is_file()]
        if missing:
            raise RuntimeError(f"Functional preflight did not produce: {missing}")
    finally:
        _clear_runtime_caches()
        shutil.rmtree(temporary_root, ignore_errors=True)


def _finalize_results(
    core: ModuleType,
    results: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    config: SimulationConfig,
    root: Path,
    checkpoint_dir: Path,
    signature: Mapping[str, Any],
    api_audit: Mapping[str, Any],
    *,
    total_expected: int,
    hard_exit: bool,
) -> None:
    state_path = checkpoint_dir / "finalization_state.json"
    _atomic_write_json(
        state_path,
        {
            "status": "running",
            "started_at_utc": _utc_timestamp(),
            "successful_replications": len(results),
            "failed_replications": len(failures),
            "total_expected_replications": total_expected,
        },
    )
    try:
        aggregate_and_save(
            core,
            results,
            failures,
            config,
            root,
            api_audit,
            hard_exit=hard_exit,
        )
    except Exception as exc:
        _atomic_write_json(
            state_path,
            {
                "status": "failed",
                "failed_at_utc": _utc_timestamp(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "recovery_command": (
                    "python simulations.py --aggregate-only --output-dir "
                    f"{config.output_dir}"
                ),
            },
        )
        raise
    _atomic_write_json(
        checkpoint_dir / "run_complete.json",
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "fingerprint": signature["fingerprint"],
            "successful_replications": len(results),
            "failed_replications": len(failures),
            "total_expected_replications": total_expected,
            "completed_at_utc": _utc_timestamp(),
        },
    )
    _atomic_write_json(
        state_path,
        {
            "status": "complete",
            "completed_at_utc": _utc_timestamp(),
            "successful_replications": len(results),
            "failed_replications": len(failures),
            "total_expected_replications": total_expected,
        },
    )


def main(*, hard_exit: bool = False) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    parser = argparse.ArgumentParser(description="X-GAT-DRL comparative simulations")
    parser.add_argument("--core-path", type=str, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--api-audit-only", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Run adversarial and reduced end-to-end checks, then exit.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip the automatic reduced end-to-end preflight.",
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Regenerate final tables and figures from existing checkpoints only.",
    )
    parser.add_argument("--scenarios", type=str, default=None)
    parser.add_argument("--market-seeds", type=str, default=None)
    parser.add_argument("--policy-seeds", type=str, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "Archive checkpoints already present in --output-dir and start the "
            "configured design from zero. Without this flag, compatible "
            "checkpoints are resumed automatically."
        ),
    )
    parser.add_argument(
        "--resume-compatible-checkpoints",
        action="store_true",
        help=(
            "Adopt checkpoints from a reviewed code-only bug fix when the complete "
            "scientific configuration is unchanged. Recovered payloads are fully "
            "validated before reuse."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately after a terminal market or replication failure.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Number of retries for transient preparation or replication failures.",
    )
    parser.add_argument(
        "--allow-partial-results",
        action="store_true",
        help="Permit final aggregation when some configured replications are missing.",
    )
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--benchmark-epochs", type=int, default=None)
    parser.add_argument("--graph-update-interval", type=int, default=None)
    parser.add_argument("--graph-alpha-refresh-interval", type=int, default=None)
    parser.add_argument("--hierarchical-bootstraps", type=int, default=None)
    parser.add_argument("--mcs-bootstraps", type=int, default=None)
    parser.add_argument("--reward-scale", type=float, default=None)
    parser.add_argument("--encoder-learning-rate", type=float, default=None)
    parser.add_argument("--actor-learning-rate", type=float, default=None)
    parser.add_argument("--critic-learning-rate", type=float, default=None)
    parser.add_argument("--clip-epsilon", type=float, default=None)
    parser.add_argument("--target-kl", type=float, default=None)
    parser.add_argument(
        "--dsr-effective-trials", type=float, default=None,
        help="Effective number of model/configuration trials for the DSR.",
    )
    parser.add_argument(
        "--ablation",
        choices=(
            "none",
            "with_anchor",
            "with_behaviour",
            "no_cmdp",
            "dynamic_cash",
            "no_risk_graph",
            "no_predictive_graph",
            "no_graph",
            "no_quantile",
            "no_optimizer",
            "no_pretrain",
        ),
        default="none",
    )
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative.")
    if args.aggregate_only and args.restart:
        parser.error("--aggregate-only cannot be combined with --restart.")
    if args.aggregate_only and args.preflight_only:
        parser.error("--aggregate-only cannot be combined with --preflight-only.")

    core = _load_core_module(args.core_path)
    api_audit = _validate_core_api(core)
    _configure_torch_threads()
    LOGGER.info("Loaded core module: %s", api_audit["path"])
    LOGGER.info("Core SHA-256: %s", api_audit["sha256"])
    if args.api_audit_only:
        print(json.dumps(api_audit, indent=2))
        return

    config = SimulationConfig()
    scenarios = _parse_csv_values(args.scenarios, str)
    market_seeds = _parse_csv_values(args.market_seeds, int)
    policy_seeds = _parse_csv_values(args.policy_seeds, int)
    if scenarios is not None:
        config = replace(config, scenarios=scenarios)
    if market_seeds is not None:
        config = replace(config, market_seeds=market_seeds)
    if policy_seeds is not None:
        config = replace(config, policy_seeds=policy_seeds)
    if args.output_dir is not None:
        config = replace(config, output_dir=args.output_dir)
    explicit_updates = {
        "n_samples": args.samples,
        "ppo_episodes": args.episodes,
        "benchmark_epochs": args.benchmark_epochs,
        "graph_update_interval": args.graph_update_interval,
        "graph_alpha_refresh_interval": args.graph_alpha_refresh_interval,
        "hierarchical_bootstraps": args.hierarchical_bootstraps,
        "mcs_bootstraps": args.mcs_bootstraps,
        "reward_scale": args.reward_scale,
        "encoder_learning_rate": args.encoder_learning_rate,
        "actor_learning_rate": args.actor_learning_rate,
        "critic_learning_rate": args.critic_learning_rate,
        "clip_epsilon": args.clip_epsilon,
        "target_kl": args.target_kl,
        "dsr_effective_trials": args.dsr_effective_trials,
    }
    explicit_updates = {
        key: value for key, value in explicit_updates.items() if value is not None
    }
    if explicit_updates:
        config = replace(config, **explicit_updates)
    ablation_updates = {
        "with_anchor": {"use_benchmark_anchor": True},
        "with_behaviour": {"use_behaviour_cloning": True},
        "no_cmdp": {"use_cmdp": False},
        "dynamic_cash": {"use_dynamic_cash": True},
        "no_risk_graph": {"use_risk_graph": False},
        "no_predictive_graph": {"use_predictive_graph": False},
        "no_graph": {"use_risk_graph": False, "use_predictive_graph": False},
        "no_quantile": {"use_quantile_head": False},
        "no_optimizer": {"use_differentiable_optimizer": False},
        "no_pretrain": {"representation_pretrain_epochs": 0, "representation_batches_per_epoch": 0},
    }
    if args.ablation in ablation_updates:
        config = replace(config, **ablation_updates[args.ablation])
    if args.quick:
        config = replace(
            config,
            n_samples=420,
            validation_fraction=0.10,
            graph_window=120,
            fast_graph_window=60,
            graph_min_history=50,
            fast_graph_min_history=36,
            graph_update_interval=80,
            graph_alpha_refresh_interval=160,
            graph_bootstrap_replicates=1,
            rolling_graph_bootstrap_replicates=0,
            predictive_bootstrap_replicates=2,
            predictive_report_bootstrap_replicates=4,
            predictive_null_replicates=8,
            predictive_report_null_replicates=12,
            ppo_episodes=2,
            direct_utility_coefficient=0.05,
            expert_auxiliary_coefficient=0.02,
            gate_auxiliary_coefficient=0.01,
            ppo_update_epochs=1,
            episode_length=48,
            behaviour_epochs=1,
            behaviour_batches_per_epoch=1,
            benchmark_epochs=1,
            benchmark_minimum_epochs=1,
            benchmark_validation_patience=1,
            critic_updates_per_actor=1,
            encoder_freeze_episodes=0,
            representation_pretrain_epochs=1,
            representation_batches_per_epoch=1,
            future_risk_horizon=10,
            batch_size=16,
            validation_interval=1,
            validation_patience=2,
            scenarios=(config.scenarios[0],),
            market_seeds=(config.market_seeds[0],),
            policy_seeds=(config.policy_seeds[0],),
            mcs_bootstraps=20,
            hierarchical_bootstraps=100,
            sign_flip_repetitions=100,
            output_dir=config.output_dir if args.output_dir is not None else Path("simulation_smoke_test"),
        )
    config.validate()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    native_bmm_status = _configure_torch_native_bmm(device)
    if native_bmm_status["native_bmm_disabled"]:
        LOGGER.warning(
            "Disabled PyTorch's experimental native CUDA bmm override because "
            "Python.h is unavailable; using the regular ATen CUDA implementation."
        )
    LOGGER.info("Using device: %s", device)

    if not args.aggregate_only:
        _run_adversarial_portfolio_preflight(core, config)
        LOGGER.info("Adversarial portfolio preflight passed.")
        run_functional_preflight = args.preflight_only or (
            not args.skip_preflight and not args.quick
        )
        if run_functional_preflight:
            LOGGER.info("Running reduced end-to-end preflight.")
            _run_functional_preflight(core, config, device, api_audit)
            LOGGER.info("Reduced end-to-end preflight passed.")
        if args.preflight_only:
            return

    root = config.output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Keep the lock stream alive until normal termination or os._exit().  POSIX
    # automatically releases the lock after an abrupt process termination.
    run_lock = _acquire_run_lock(root)
    checkpoint_dir, signature, results_by_key, checkpoint_records = (
        _initialise_checkpoint_store(
            root,
            config,
            api_audit,
            restart=args.restart,
            allow_compatible_code_change=args.resume_compatible_checkpoints,
        )
    )
    expected_keys = _expected_replication_keys(config)
    expected_key_set = set(expected_keys)
    total_expected = len(expected_keys)
    if results_by_key:
        LOGGER.info(
            "Recovered %d/%d completed replications from checkpoints.",
            len(results_by_key),
            total_expected,
        )
    failures: list[dict[str, Any]] = []
    suite_start = time.perf_counter()

    if not args.aggregate_only:
        for scenario in config.scenarios:
            scenario_config = replace(config, scenario=scenario)
            for market_seed in config.market_seeds:
                pending_policy_seeds = tuple(
                    policy_seed
                    for policy_seed in config.policy_seeds
                    if _replication_key(scenario, market_seed, policy_seed)
                    not in results_by_key
                )
                if not pending_policy_seeds:
                    LOGGER.info(
                        "Skipping completed scenario=%s market_seed=%d",
                        scenario,
                        market_seed,
                    )
                    continue
                LOGGER.info(
                    "Preparing scenario=%s market_seed=%d (%d policy seed(s) pending)",
                    scenario,
                    market_seed,
                    len(pending_policy_seeds),
                )

                market: MarketData | None = None
                prepared: PreparedData | None = None
                preparation_failure: dict[str, Any] | None = None
                for attempt in range(args.max_retries + 1):
                    try:
                        core.seed_everything(market_seed, deterministic=True)
                        market = generate_synthetic_market(scenario_config, market_seed)
                        prepared = prepare_data(
                            core,
                            market,
                            scenario_config,
                            market_seed,
                            device,
                        )
                        preparation_failure = None
                        break
                    except Exception as exc:
                        preparation_failure = {
                            "Scenario": scenario,
                            "Market seed": market_seed,
                            "Policy seed": "ALL_PENDING",
                            "Stage": "market_preparation",
                            "Attempt": attempt + 1,
                            "Error type": type(exc).__name__,
                            "Error": str(exc),
                            "Traceback": traceback.format_exc(),
                        }
                        if attempt < args.max_retries:
                            LOGGER.exception(
                                "Market preparation failed for scenario=%s market=%d; "
                                "retrying (%d/%d)",
                                scenario,
                                market_seed,
                                attempt + 1,
                                args.max_retries,
                            )
                            _clear_runtime_caches()
                            continue
                        LOGGER.exception(
                            "Market preparation failed terminally for scenario=%s market=%d",
                            scenario,
                            market_seed,
                        )

                if market is None or prepared is None:
                    assert preparation_failure is not None
                    for policy_seed in pending_policy_seeds:
                        failures.append(
                            {
                                **preparation_failure,
                                "Policy seed": policy_seed,
                            }
                        )
                    _write_failure_table(root, failures)
                    if args.fail_fast:
                        raise RuntimeError(
                            f"Market preparation failed for {scenario}, seed {market_seed}."
                        )
                    continue

                for policy_seed in pending_policy_seeds:
                    key = _replication_key(scenario, market_seed, policy_seed)
                    replication_result: dict[str, Any] | None = None
                    terminal_failure: dict[str, Any] | None = None
                    for attempt in range(args.max_retries + 1):
                        LOGGER.info(
                            "Running scenario=%s market_seed=%d policy_seed=%d "
                            "(attempt %d/%d)",
                            scenario,
                            market_seed,
                            policy_seed,
                            attempt + 1,
                            args.max_retries + 1,
                        )
                        try:
                            replication_result = run_replication(
                                core,
                                scenario_config,
                                market_seed,
                                policy_seed,
                                device,
                                market=market,
                                prepared=prepared,
                            )
                            _save_replication_checkpoint(
                                checkpoint_dir,
                                signature,
                                key,
                                replication_result,
                                checkpoint_records,
                                total_expected=total_expected,
                                config=scenario_config,
                            )
                            results_by_key[key] = replication_result
                            terminal_failure = None
                            LOGGER.info(
                                "Saved checkpoint scenario=%s market_seed=%d "
                                "policy_seed=%d (%d/%d complete)",
                                scenario,
                                market_seed,
                                policy_seed,
                                len(results_by_key),
                                total_expected,
                            )
                            break
                        except Exception as exc:
                            replication_result = None
                            terminal_failure = {
                                "Scenario": scenario,
                                "Market seed": market_seed,
                                "Policy seed": policy_seed,
                                "Stage": "replication",
                                "Attempt": attempt + 1,
                                "Error type": type(exc).__name__,
                                "Error": str(exc),
                                "Traceback": traceback.format_exc(),
                            }
                            if attempt < args.max_retries:
                                LOGGER.exception(
                                    "Replication failed for scenario=%s market=%d "
                                    "policy=%d; retrying (%d/%d)",
                                    scenario,
                                    market_seed,
                                    policy_seed,
                                    attempt + 1,
                                    args.max_retries,
                                )
                                _clear_runtime_caches()
                                continue
                            LOGGER.exception(
                                "Replication failed terminally for scenario=%s "
                                "market=%d policy=%d",
                                scenario,
                                market_seed,
                                policy_seed,
                            )
                    if replication_result is None:
                        assert terminal_failure is not None
                        failures.append(terminal_failure)
                        _write_failure_table(root, failures)
                        if args.fail_fast:
                            raise RuntimeError(
                                "Replication failed terminally for "
                                f"{scenario}/{market_seed}/{policy_seed}."
                            )

    unexpected_keys = set(results_by_key) - expected_key_set
    if unexpected_keys:
        raise RuntimeError(
            f"Recovered results outside the configured design: {sorted(unexpected_keys)}"
        )
    results = [
        results_by_key[key]
        for key in expected_keys
        if key in results_by_key
    ]
    if not results:
        raise RuntimeError("No successful replications are available for aggregation.")

    missing_keys = [key for key in expected_keys if key not in results_by_key]
    if missing_keys and not args.allow_partial_results:
        _write_failure_table(root, failures)
        _atomic_write_json(
            checkpoint_dir / "run_incomplete.json",
            {
                "schema_version": CHECKPOINT_SCHEMA_VERSION,
                "fingerprint": signature["fingerprint"],
                "successful_replications": len(results),
                "missing_replications": len(missing_keys),
                "total_expected_replications": total_expected,
                "updated_at_utc": _utc_timestamp(),
                "next_missing_keys": missing_keys[:20],
                "recovery_command": (
                    "python simulations.py --output-dir "
                    f"{config.output_dir}"
                ),
            },
        )
        raise RuntimeError(
            f"{len(missing_keys)} configured replications are still missing. "
            "All successful checkpoints were preserved; rerun the same command "
            "to execute only the missing replications."
        )

    _finalize_results(
        core,
        results,
        failures,
        config,
        root,
        checkpoint_dir,
        signature,
        api_audit,
        total_expected=total_expected,
        hard_exit=hard_exit,
    )
    LOGGER.info(
        "Completed %d replications (%d failed) in %.2f seconds. Results: %s",
        len(results),
        len(failures),
        time.perf_counter() - suite_start,
        root,
    )
    # Explicit reference documents why the descriptor remains live until here.
    _ = run_lock
    if hard_exit:
        # Some combinations of PyTorch, OpenMP, and scikit-learn can stall
        # while destroying native thread pools after every result has already
        # been written.  The command-line entry point exits here, after all
        # files are closed and streams flushed; imported use keeps normal
        # Python teardown semantics.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main(hard_exit=True)

