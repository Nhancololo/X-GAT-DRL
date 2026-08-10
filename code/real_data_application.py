"""Real-data application for X-GAT-DRL and controlled portfolio benchmarks.

The default experiment uses a diversified set of liquid U.S.-listed ETFs,
a three-month Treasury cash proxy, and daily macro-financial indicators. The
primary evaluation is a fixed holdout beginning in 2019. An annual-refit mode
is also available as a sensitivity analysis.

This module is intentionally separate from ``simulations.py`` and
``xgat_drl_code.py``. It imports their public and research-facing functions,
builds the real-data dataset, performs descriptive and regime analyses, trains
all three X-GAT temporal encoders and the six retained controlled benchmarks, and
creates tables and figures from the strictly out-of-sample period.

The code writes hashes and retrieval metadata for every cached source file.
"""

from __future__ import annotations

import argparse
import hashlib
from functools import wraps
import importlib.util
import inspect
import io
import json
import logging
import math
import os
import random
import re
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import asdict, dataclass, fields, replace
from datetime import date, datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

# Keep numerical backends from creating excessive CPU thread pools.
_NATIVE_THREADS = os.getenv("XGAT_NATIVE_THREADS", "1")
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = _NATIVE_THREADS

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy.stats import kurtosis, skew

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
    from pgmpy.models import DiscreteBayesianNetwork
    from sklearn.preprocessing import KBinsDiscretizer
except ImportError:
    HillClimbSearch = None
    MaximumLikelihoodEstimator = None
    DiscreteBayesianNetwork = None
    KBinsDiscretizer = None

try:
    from statsmodels.tsa.stattools import adfuller, kpss
except Exception:  
    adfuller = None
    kpss = None

LOGGER = logging.getLogger("xgat.real_data")

# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class AssetSpec:
    ticker: str
    label: str
    group: str

DEFAULT_ASSETS: tuple[AssetSpec, ...] = (
    AssetSpec("SPY", "U.S. large-cap equity", "Equity"),
    AssetSpec("QQQ", "U.S. growth equity", "Equity"),
    AssetSpec("IWM", "U.S. small-cap equity", "Equity"),
    AssetSpec("EFA", "Developed ex-U.S. equity", "Equity"),
    AssetSpec("EEM", "Emerging-market equity", "Equity"),
    AssetSpec("IEF", "7--10 year U.S. Treasury", "Treasury"),
    AssetSpec("TLT", "20+ year U.S. Treasury", "Treasury"),
    AssetSpec("LQD", "Investment-grade credit", "Credit"),
    AssetSpec("HYG", "High-yield credit", "Credit"),
    AssetSpec("GLD", "Gold", "Commodity"),
    AssetSpec("DBC", "Broad commodities", "Commodity"),
    AssetSpec("VNQ", "U.S. real estate", "Real estate"),
)

FRED_SERIES: Mapping[str, str] = {
    "DGS3MO": "Three-month Treasury yield",
    "VIXCLS": "CBOE VIX",
    "T10Y3M": "10-year minus 3-month Treasury spread",
    "BAMLH0A0HYM2": "U.S. high-yield option-adjusted spread",
    "DCOILBRENTEU": "Brent crude oil price",
}

@dataclass(frozen=True)
class EventWindow:
    name: str
    start: str
    end: str | None
    category: str

DEFAULT_EVENT_WINDOWS: tuple[EventWindow, ...] = (
    EventWindow("COVID-19 crash", "2020-02-19", "2020-03-23", "Pandemic"),
    EventWindow("COVID-19 initial recovery", "2020-03-24", "2020-08-18", "Pandemic"),
    EventWindow("Inflation and broad market sell-off", "2022-01-03", "2022-10-12", "Macro-financial"),
)

TEMPORAL_MODE_LABELS: Mapping[str, str] = {
    "hybrid": "X-GAT-DRL (Hybrid)",
    "gru": "X-GAT-DRL (GRU)",
    "lstm": "X-GAT-DRL (LSTM)",
}

EXPERT_METHOD_ORDER: tuple[str, ...] = (
    "1/N",
    "GMV",
    "HMM-GMV",
    "GLASSO-GAT",
    "TC-MAC",
    "JM-MPC",
)

BASE_SIMULATION_METHOD_ORDER: tuple[str, ...] = (
    "X-GAT-DRL",
    *EXPERT_METHOD_ORDER,
)

def _real_data_method_order(temporal_modes: Sequence[str]) -> tuple[str, ...]:
    return (
        *(TEMPORAL_MODE_LABELS[mode] for mode in temporal_modes),
        *EXPERT_METHOD_ORDER,
    )

def _primary_xgat_method(temporal_modes: Sequence[str]) -> str:
    preferred = "hybrid" if "hybrid" in temporal_modes else temporal_modes[0]
    return TEMPORAL_MODE_LABELS[preferred]

REQUIRED_SIMULATION_API: tuple[str, ...] = (
    "SimulationConfig",
    "MarketData",
    "EvaluationResult",
    "METHOD_ORDER",
    "prepare_data",
    "build_classical_benchmarks",
    "train_benchmark_models",
    "build_policy_base_path",
    "build_causal_teacher_path",
    "train_xgat_agent",
    "train_policy_agent",
    "evaluate_methods",
    "build_metrics_table",
)

@dataclass(frozen=True)
class RealDataConfig:
    start_date: str = "2008-01-02"
    end_date: str | None = "2026-08-04"
    out_of_sample_start: str = "2019-01-02"
    evaluation_mode: str = "fixed_holdout"
    benchmark_ticker: str = "SPY"
    risk_assets: tuple[AssetSpec, ...] = DEFAULT_ASSETS
    cash_series: str = "DGS3MO"
    fred_series: tuple[str, ...] = tuple(FRED_SERIES.keys())
    policy_seeds: tuple[int, ...] = (101, 211, 307, 401, 503)
    temporal_modes: tuple[str, ...] = ("hybrid", "gru", "lstm")
    validation_fraction: float = 0.15
    annual_training_years: int = 8
    minimum_annual_test_observations: int = 63
    minimum_training_observations: int = 360
    periods_per_year: int = 252
    transaction_cost: float = 0.0010
    slippage_coefficient: float = 0.00005
    impact_coefficient: float = 0.00020
    maximum_cash: float = 1.0
    ppo_episodes: int = 200 
    benchmark_epochs: int = 100 
    block_bootstraps: int = 5_000
    block_length: int = 20
    confidence_level: float = 0.95
    output_dir: Path = Path("real_data_results")
    cache_dir: Path = Path("real_data_cache")
    refresh_data: bool = False
    local_prices_csv: Path | None = None
    local_volumes_csv: Path | None = None
    local_macro_csv: Path | None = None
    continue_on_error: bool = False
    quick: bool = False

    @property
    def tickers(self) -> tuple[str, ...]:
        return tuple(asset.ticker for asset in self.risk_assets)

    @property
    def asset_labels(self) -> dict[str, str]:
        return {asset.ticker: asset.label for asset in self.risk_assets}

    @property
    def asset_groups(self) -> dict[str, str]:
        return {asset.ticker: asset.group for asset in self.risk_assets}

    @property
    def resolved_end_date(self) -> str:
        if self.end_date is not None:
            return pd.Timestamp(self.end_date).date().isoformat()
        return datetime.now(timezone.utc).date().isoformat()

    def validate(self) -> None:
        if self.evaluation_mode not in {"fixed_holdout", "annual_refit"}:
            raise ValueError("evaluation_mode must be fixed_holdout or annual_refit.")
        if self.benchmark_ticker not in self.tickers:
            raise ValueError("benchmark_ticker must be one of the risk-asset tickers.")
        if len(self.tickers) < 3:
            raise ValueError("At least three risk assets are required.")
        if len(set(self.tickers)) != len(self.tickers):
            raise ValueError("Risk-asset tickers must be unique.")
        if not 0.05 <= self.validation_fraction < 0.30:
            raise ValueError("validation_fraction must lie in [0.05, 0.30).")
        if not self.policy_seeds:
            raise ValueError("At least one policy seed is required.")
        if not self.temporal_modes:
            raise ValueError("At least one X-GAT temporal mode is required.")

@dataclass(frozen=True)
class AlignedData:
    prices: pd.DataFrame
    volumes: pd.DataFrame
    macro: pd.DataFrame
    log_returns: pd.DataFrame
    risk_log_returns: pd.DataFrame
    cash_log_returns: pd.Series
    phases: pd.DataFrame
    data_quality: pd.DataFrame

@dataclass
class WindowRun:
    label: str
    dates: pd.DatetimeIndex
    evaluations: dict[int, Any]
    fold_performance: pd.DataFrame
    policy_training: pd.DataFrame
    benchmark_training: pd.DataFrame
    hmm_probabilities: pd.DataFrame
    hmm_diagnostics: dict[str, Any]
    risk_path: np.ndarray
    predictive_path: np.ndarray
    feature_names: tuple[str, ...]
    training_times: pd.DataFrame

# -----------------------------------------------------------------------------
# 2. Module loading
# -----------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

def _configure_torch_threads() -> None:
    count = max(1, int(os.getenv("XGAT_TORCH_THREADS", "1")))
    torch.set_num_threads(count)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

def _load_module(path: Path, module_name: str) -> ModuleType:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Required module not found: {resolved}")
    specification = importlib.util.spec_from_file_location(module_name, resolved)
    if specification is None or specification.loader is None:
        raise ImportError(f"Unable to load module from {resolved}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module

def _configure_real_data_method_order(
    sim: ModuleType, temporal_modes: Sequence[str]
) -> tuple[str, ...]:
    method_order = _real_data_method_order(temporal_modes)
    sim.METHOD_ORDER = method_order
    return method_order

def _train_policy_with_temporal_mode(
    core: ModuleType,
    sim: ModuleType,
    market: Any,
    prepared: Any,
    teacher_path: np.ndarray,
    model_config: Any,
    device: torch.device,
    seed: int,
    *,
    label: str,
    temporal_mode: str,
    use_risk_graph: bool,
    use_predictive_graph: bool,
) -> tuple[Any, pd.DataFrame, float, dict[str, float]]:
    if temporal_mode not in TEMPORAL_MODE_LABELS:
        raise ValueError(f"Unknown temporal mode: {temporal_mode}")
    agent, diagnostics, elapsed, multipliers = sim.train_policy_agent(
        core,
        market,
        prepared,
        teacher_path,
        model_config,
        device,
        seed,
        label=label,
        use_risk_graph=use_risk_graph,
        use_predictive_graph=use_predictive_graph,
        temporal_mode=temporal_mode,
    )
    diagnostics = diagnostics.copy()
    diagnostics["Temporal mode"] = temporal_mode
    diagnostics["Uses risk graph"] = bool(use_risk_graph)
    diagnostics["Uses predictive graph"] = bool(use_predictive_graph)
    return agent, diagnostics, elapsed, multipliers

def _install_core_compatibility_adapters(core: ModuleType) -> list[str]:
    installed: list[str] = []
    function_name = "constrained_ppo_update_batch"
    original = getattr(core, function_name, None)
    if callable(original) and not bool(getattr(original, "_real_data_compatibility", False)):
        aliases = {
            "Mean anchor gate": "Mean optimiser-base reliance",
            "Mean plain gate": "Mean temporal-return gate",
            "Mean risk-graph gate": "Mean risk-expert usage",
            "Mean predictive-graph gate": "Mean predictive-return gate",
        }
        @wraps(original)
        def compatible_update(*args: Any, **kwargs: Any) -> Any:
            diagnostics = original(*args, **kwargs)
            if isinstance(diagnostics, Mapping):
                diagnostics = dict(diagnostics)
                for legacy_name, current_name in aliases.items():
                    if legacy_name not in diagnostics and current_name in diagnostics:
                        diagnostics[legacy_name] = diagnostics[current_name]
            return diagnostics

        compatible_update._real_data_compatibility = True  
        setattr(core, function_name, compatible_update)
        installed.append("Mapped current PPO gate diagnostics to the labels expected by simulations.py.")

    dsr_name = "deflated_sharpe_ratio"
    original_dsr = getattr(core, dsr_name, None)
    if callable(original_dsr) and not bool(getattr(original_dsr, "_real_data_compatibility", False)):
        dsr_parameters = inspect.signature(original_dsr).parameters
        if "observed_sr" in dsr_parameters and "expected_sr" not in dsr_parameters:
            @wraps(original_dsr)
            def compatible_dsr(*args: Any, **kwargs: Any) -> Any:
                if "expected_sr" in kwargs:
                    observed = kwargs.pop("expected_sr")
                    benchmark = kwargs.pop("benchmark_sr", 0.0)
                    kwargs["observed_sr"] = observed
                    kwargs["trial_sharpes"] = np.asarray([benchmark], dtype=float)
                return original_dsr(*args, **kwargs)

            compatible_dsr._real_data_compatibility = True  
            setattr(core, dsr_name, compatible_dsr)
            installed.append("Adapted the legacy Sharpe-ratio keyword interface used by simulations.py.")
    return installed

def _validate_runtime_interfaces(core: ModuleType, sim: ModuleType) -> dict[str, Any]:
    missing_simulation = [name for name in REQUIRED_SIMULATION_API if not hasattr(sim, name)]
    if missing_simulation:
        raise RuntimeError(
            "The simulations module is incompatible with the real-data application. "
            f"Missing symbols: {', '.join(missing_simulation)}"
        )

    method_order = tuple(getattr(sim, "METHOD_ORDER"))
    if method_order != BASE_SIMULATION_METHOD_ORDER:
        raise RuntimeError("The simulations module must expose its base controlled comparison in this order.")

    required_core = tuple(getattr(sim, "REQUIRED_CORE_API", ()))
    if not required_core:
        required_core = ("PortfolioConstraints", "seed_everything", "run_statistical_tests", "compute_model_confidence_set", "XGATDRLAgent")
    missing_core = [name for name in required_core if not hasattr(core, name)]
    if missing_core:
        raise RuntimeError(f"Missing symbols: {', '.join(missing_core)}")

    core_path = Path(getattr(core, "__file__", ""))
    sim_path = Path(getattr(sim, "__file__", ""))
    return {
        "base_simulation_method_order": list(method_order),
        "supported_temporal_modes": list(TEMPORAL_MODE_LABELS),
        "core_path": str(core_path.resolve()) if core_path.is_file() else str(core_path),
        "core_sha256": _sha256(core_path) if core_path.is_file() else None,
        "simulations_path": str(sim_path.resolve()) if sim_path.is_file() else str(sim_path),
        "simulations_sha256": _sha256(sim_path) if sim_path.is_file() else None,
    }

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)

def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "item"

# -----------------------------------------------------------------------------
# 3. Data acquisition and archiving
# -----------------------------------------------------------------------------

def _request_bytes(url: str, *, attempts: int = 7, timeout: int = 45) -> bytes:
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15",
    ]
    last_error: Exception | None = None
    for attempt in range(attempts):
        headers = {
            "User-Agent": random.choice(user_agents),
            "Accept": "application/json,text/csv,text/plain,*/*",
        }
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code in {429, 401, 403}:
                if attempt + 1 < attempts:
                    time.sleep(15.0 + 10.0 * attempt)
            else:
                if attempt + 1 < attempts:
                    time.sleep(min(10.0, 1.0 * 2**attempt))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(min(10.0, 1.0 * 2**attempt))
    raise RuntimeError(f"Unable to retrieve {url}: {last_error}")

def _unix_seconds(value: str | pd.Timestamp) -> int:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return int(stamp.timestamp())

def _download_yahoo_symbol(
    ticker: str,
    start_date: str,
    end_date: str,
    destination: Path,
    *,
    refresh: bool,
) -> pd.DataFrame:
    if destination.is_file() and not refresh:
        frame = pd.read_csv(destination, parse_dates=["Date"]).set_index("Date")
        if frame.index.min() <= pd.Timestamp(start_date) and frame.index.max() >= pd.Timestamp(end_date):
            return frame.sort_index()
        LOGGER.info("Cache for %s is incomplete. Redownloading.", ticker)

    try:
        import yfinance as yf
        yf_available = True
    except ImportError:
        yf_available = False

    frame = None
    if yf_available:
        try:
            yf_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            df = yf.download(ticker, start=start_date, end=yf_end, auto_adjust=False, progress=False)
            if not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                frame = pd.DataFrame(
                    {
                        "Adjusted close": pd.to_numeric(df.get("Adj Close", df.get("Close")), errors="coerce"),
                        "Open": pd.to_numeric(df.get("Open"), errors="coerce"),
                        "High": pd.to_numeric(df.get("High"), errors="coerce"),
                        "Low": pd.to_numeric(df.get("Low"), errors="coerce"),
                        "Close": pd.to_numeric(df.get("Close"), errors="coerce"),
                        "Volume": pd.to_numeric(df.get("Volume"), errors="coerce"),
                    },
                    index=df.index
                )
                frame.index.name = "Date"
                if frame.index.tzinfo is not None:
                    frame.index = frame.index.tz_convert(None)
                frame.index = frame.index.normalize()
        except Exception as e:
            LOGGER.warning("yfinance failed for %s: %s. Falling back to urllib.", ticker, e)

    if frame is None:
        period1 = _unix_seconds(start_date)
        period2 = _unix_seconds(pd.Timestamp(end_date) + pd.Timedelta(days=1))
        encoded = urllib.parse.quote(ticker, safe="")
        query = urllib.parse.urlencode(
            {"period1": period1, "period2": period2, "interval": "1d", "events": "div,splits", "includeAdjustedClose": "true"}
        )
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}?{query}"
        payload = json.loads(_request_bytes(url).decode("utf-8"))
        result_list = payload.get("chart", {}).get("result") or []
        if not result_list: raise RuntimeError(f"Yahoo returned no observations for {ticker}.")
        result = result_list[0]
        timestamps = result.get("timestamp") or []
        quote_list = result.get("indicators", {}).get("quote") or []
        quote = quote_list[0]
        adjusted_list = result.get("indicators", {}).get("adjclose") or []
        adjusted = adjusted_list[0].get("adjclose") if adjusted_list else None
        close = adjusted if adjusted is not None else quote.get("close")
        
        index = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize()
        frame = pd.DataFrame(
            {
                "Adjusted close": pd.to_numeric(close, errors="coerce"),
                "Open": pd.to_numeric(quote.get("open"), errors="coerce"),
                "High": pd.to_numeric(quote.get("high"), errors="coerce"),
                "Low": pd.to_numeric(quote.get("low"), errors="coerce"),
                "Close": pd.to_numeric(quote.get("close"), errors="coerce"),
                "Volume": pd.to_numeric(quote.get("volume"), errors="coerce"),
            },
            index=index,
        )
        frame.index.name = "Date"

    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
    frame = frame.dropna(subset=["Adjusted close"])
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination)
    return frame

def _download_fred_series(
    series_id: str,
    start_date: str,
    end_date: str,
    destination: Path,
    *,
    refresh: bool,
) -> pd.Series:
    if destination.is_file() and not refresh:
        frame = pd.read_csv(destination, parse_dates=["Date"])
        series = frame.set_index("Date")[series_id]
        if frame["Date"].min() <= pd.Timestamp(start_date) and frame["Date"].max() >= pd.Timestamp(end_date):
            return pd.to_numeric(series, errors="coerce").sort_index()
        LOGGER.info("Cache for %s is incomplete. Redownloading.", series_id)

    query = urllib.parse.urlencode({"id": series_id, "cosd": start_date, "coed": end_date})
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{query}"
    
    try:
        raw = _request_bytes(url)
        frame = pd.read_csv(io.BytesIO(raw))
        date_column = "DATE" if "DATE" in frame.columns else frame.columns[0]
        value_column = series_id if series_id in frame.columns else [c for c in frame.columns if c != date_column][0]
        frame = frame.rename(columns={date_column: "Date", value_column: series_id})
    except Exception as e:
        api_key = os.getenv("FRED_API_KEY")
        if not api_key: raise RuntimeError(f"FRED CSV failed, no API key: {e}")
        api_query = urllib.parse.urlencode({"series_id": series_id, "api_key": api_key, "file_type": "json", "observation_start": start_date, "observation_end": end_date})
        raw = _request_bytes(f"https://api.stlouisfed.org/fred/series/observations?{api_query}")
        observations = json.loads(raw.decode("utf-8")).get("observations", [])
        frame = pd.DataFrame(observations)[["date", "value"]].rename(columns={"date": "Date", "value": series_id})

    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame[series_id] = pd.to_numeric(frame[series_id], errors="coerce")
    frame = frame.dropna(subset=["Date", series_id]).drop_duplicates("Date", keep="last").sort_values("Date")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame[["Date", series_id]].to_csv(destination, index=False)
    return frame.set_index("Date")[series_id]

def _read_wide_csv(path: Path, expected_columns: Sequence[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    date_column = [c for c in frame.columns if c.lower() in {"date", "datetime", "timestamp"}][0]
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    frame = frame.dropna(subset=[date_column]).set_index(date_column).sort_index()
    return frame.loc[:, list(expected_columns)].apply(pd.to_numeric, errors="coerce")

def acquire_data(config: RealDataConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw_dir = config.cache_dir / "raw"
    market_dir = raw_dir / "market"
    fred_dir = raw_dir / "fred"
    market_dir.mkdir(parents=True, exist_ok=True)
    fred_dir.mkdir(parents=True, exist_ok=True)

    if config.local_prices_csv is not None:
        prices = _read_wide_csv(config.local_prices_csv, config.tickers)
        volumes = _read_wide_csv(config.local_volumes_csv, config.tickers) if config.local_volumes_csv else pd.DataFrame(index=prices.index, columns=config.tickers, dtype=float)
    else:
        price_parts, volume_parts = {}, {}
        for ticker in config.tickers:
            frame = _download_yahoo_symbol(ticker, config.start_date, config.resolved_end_date, market_dir / f"{_safe_name(ticker)}.csv", refresh=config.refresh_data)
            price_parts[ticker] = frame["Adjusted close"].rename(ticker)
            volume_parts[ticker] = frame["Volume"].rename(ticker)
        prices, volumes = pd.concat(price_parts.values(), axis=1).sort_index(), pd.concat(volume_parts.values(), axis=1).sort_index()

    if config.local_macro_csv is not None:
        macro = _read_wide_csv(config.local_macro_csv, config.fred_series)
    else:
        macro_parts = {}
        for series_id in config.fred_series:
            macro_parts[series_id] = _download_fred_series(series_id, config.start_date, config.resolved_end_date, fred_dir / f"{_safe_name(series_id)}.csv", refresh=config.refresh_data).rename(series_id)
        macro = pd.concat(macro_parts.values(), axis=1).sort_index()

    source_rows = []
    source_paths = set(raw_dir.rglob("*.csv"))
    for path in sorted(source_paths, key=lambda item: str(item)):
        source_rows.append({"File": str(path), "SHA256": _sha256(path), "Bytes": path.stat().st_size, "Modified UTC": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()})
    return prices, volumes, macro, pd.DataFrame(source_rows)

# -----------------------------------------------------------------------------
# 4. Alignment, data quality, descriptive statistics, and market phases
# -----------------------------------------------------------------------------

def _longest_zero_run(values: pd.Series | np.ndarray) -> int:
    zero = np.asarray(values, dtype=bool) if not isinstance(values, pd.Series) else values.fillna(False).astype(bool).to_numpy()
    longest = current = 0
    for flag in zero:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)

def _data_quality_report(raw_prices: pd.DataFrame, aligned_prices: pd.DataFrame, log_returns: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in aligned_prices.columns:
        raw = raw_prices[ticker] if ticker in raw_prices else pd.Series(dtype=float)
        ret = log_returns[ticker]
        vol = volumes[ticker] if ticker in volumes else pd.Series(index=aligned_prices.index, dtype=float)
        rows.append(
            {
                "Asset": ticker,
                "Raw observations": int(raw.notna().sum()),
                "Aligned observations": int(aligned_prices[ticker].notna().sum()),
                "Raw missing fraction": float(raw.isna().mean()) if len(raw) else np.nan,
                "Zero-return fraction": float(np.isclose(ret.fillna(0.0), 0.0).mean()),
                "Longest zero-return run": _longest_zero_run(np.isclose(ret, 0.0)),
                "Non-positive prices": int((aligned_prices[ticker] <= 0.0).sum()),
                "Missing volume fraction": float(vol.isna().mean()),
                "Start": aligned_prices[ticker].first_valid_index(),
                "End": aligned_prices[ticker].last_valid_index(),
            }
        )
    return pd.DataFrame(rows)

def _rolling_trend(log_price: pd.Series, window: int = 63) -> pd.DataFrame:
    values = log_price.to_numpy(dtype=float)
    slope = np.full(values.size, np.nan)
    r_squared = np.full(values.size, np.nan)
    x = np.arange(window, dtype=float)
    x_centered = x - x.mean()
    x_ss = float(np.dot(x_centered, x_centered))
    for end in range(window, values.size + 1):
        y = values[end - window : end]
        if not np.all(np.isfinite(y)): continue
        y_centered = y - y.mean()
        beta = float(np.dot(x_centered, y_centered) / max(x_ss, 1e-12))
        residual_ss = float(np.sum((y - (y.mean() + beta * x_centered)) ** 2))
        total_ss = float(np.sum(y_centered**2))
        slope[end - 1] = beta
        r_squared[end - 1] = 1.0 - residual_ss / max(total_ss, 1e-12)
    return pd.DataFrame({"Trend annualised": np.expm1(slope * 252.0), "Trend R2": r_squared}, index=log_price.index)

def identify_market_phases(
    prices: pd.DataFrame,
    risk_log_returns: pd.DataFrame,
    macro: pd.DataFrame,
    benchmark_ticker: str,
) -> pd.DataFrame:
    benchmark = prices[benchmark_ticker].astype(float)
    log_price = np.log(benchmark)
    returns = risk_log_returns[benchmark_ticker]
    drawdown = benchmark / benchmark.cummax() - 1.0
    realised_volatility = returns.rolling(21, min_periods=10).std(ddof=1) * math.sqrt(252.0)
    
    ma50 = benchmark.rolling(50, min_periods=20).mean()
    ma200 = benchmark.rolling(200, min_periods=60).mean()
    return_63 = np.expm1(log_price.diff(63))
    return_126 = np.expm1(log_price.diff(126))
    trend = _rolling_trend(log_price, window=63)
    
    early = ma200.isna()
    above_trend = np.where(early, benchmark > ma50, benchmark > ma200)

    phase = pd.Series("Sideways", index=benchmark.index, dtype="object")

    positive_return = np.where(early, return_63 > 0.0, return_126 > 0.0)
    negative_return = np.where(early, return_63 < 0.0, return_126 < 0.0)
    bear = (drawdown <= -0.20) | (
        (~pd.Series(above_trend, index=benchmark.index))
        & pd.Series(negative_return, index=benchmark.index)
        & (realised_volatility >= 0.20)
    )
    bull = (
        pd.Series(above_trend, index=benchmark.index)
        & pd.Series(positive_return, index=benchmark.index)
        & (drawdown > -0.10)
    )
    phase.loc[bull.fillna(False)] = "Bull"
    phase.loc[bear.fillna(False)] = "Bear"

    vix = macro.get("VIXCLS", pd.Series(index=benchmark.index, dtype=float)).reindex(benchmark.index).ffill()
    hy_spread = macro.get("BAMLH0A0HYM2", pd.Series(index=benchmark.index, dtype=float)).reindex(benchmark.index).ffill()
    
    vix_signal = vix.shift(1)
    vol_signal = realised_volatility.shift(1)
    spread_signal = hy_spread.shift(1)
    
    raw_stress = ((vix_signal >= 30.0) | (vol_signal >= 0.30) | (spread_signal >= 7.0))
    reference_stress = raw_stress.astype(float).rolling(5, min_periods=5).mean().ge(0.60)

    result = pd.DataFrame(
        {
            "Price": benchmark,
            "Log return": returns,
            "MA 50": ma50,
            "MA 200": ma200,
            "63-day return": return_63,
            "126-day return": return_126,
            "21-day annualised volatility": realised_volatility,
            "Drawdown": drawdown,
            "Phase": phase,
            "Reference stress": reference_stress.astype(int),
            "VIX": vix,
            "High-yield spread": hy_spread,
        }
    )
    return result.join(trend)


def align_and_transform_data(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    macro: pd.DataFrame,
    config: RealDataConfig,
) -> AlignedData:
    prices = prices.copy().sort_index().replace([np.inf, -np.inf], np.nan)
    prices = prices.loc[pd.Timestamp(config.start_date) : pd.Timestamp(config.resolved_end_date)]
    prices = prices.loc[:, list(config.tickers)]
    raw_prices = prices.copy()
    prices = prices.dropna(how="any")
    if prices.empty:
        raise RuntimeError("No common trading dates remain after price alignment.")

    volumes = volumes.reindex(prices.index).loc[:, list(config.tickers)]
    macro = macro.reindex(prices.index).ffill()
    first_cash = macro[config.cash_series].first_valid_index()
    prices = prices.loc[first_cash:]
    volumes = volumes.reindex(prices.index)
    macro = macro.reindex(prices.index).ffill()

    risk_log_returns = np.log(prices / prices.shift(1)).dropna(how="any")
    
    cash_yield = macro[config.cash_series].reindex(risk_log_returns.index).ffill().shift(1)
    cash_yield.iloc[0] = cash_yield.iloc[1] if len(cash_yield) > 1 else 0.0
    
    if cash_yield.isna().any():
        first_valid = cash_yield.first_valid_index()
        risk_log_returns = risk_log_returns.loc[first_valid:]
        cash_yield = cash_yield.loc[first_valid:]
        
    annual_cash_rate = np.clip(cash_yield.to_numpy(dtype=float) / 100.0, -0.999999, None)
    cash_log_returns = np.log1p(annual_cash_rate) / float(config.periods_per_year)
    cash_series = pd.Series(cash_log_returns, index=risk_log_returns.index, name="Cash")
    log_returns = pd.concat([risk_log_returns, cash_series], axis=1).dropna(how="any")
    
    prices = prices.reindex(log_returns.index)
    volumes = volumes.reindex(log_returns.index)
    macro = macro.reindex(log_returns.index).ffill()
    phases = identify_market_phases(prices, risk_log_returns.reindex(log_returns.index), macro, config.benchmark_ticker)
    quality = _data_quality_report(raw_prices, prices, log_returns.loc[:, list(config.tickers)], volumes)

    return AlignedData(
        prices=prices,
        volumes=volumes,
        macro=macro,
        log_returns=log_returns,
        risk_log_returns=log_returns.loc[:, list(config.tickers)],
        cash_log_returns=log_returns["Cash"],
        phases=phases.reindex(log_returns.index),
        data_quality=quality,
    )


def _maximum_drawdown(log_returns: Sequence[float]) -> float:
    values = np.asarray(log_returns, dtype=float)
    wealth = np.concatenate([[1.0], np.exp(np.cumsum(values))])
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def _empirical_cvar(values: Sequence[float], alpha: float = 0.05) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return np.nan
    threshold = np.quantile(array, alpha)
    tail = array[array <= threshold]
    return float(tail.mean()) if tail.size else float(threshold)


def _holm_adjust(values: Sequence[float]) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    result = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return result
    indices = np.where(finite)[0]
    ordered_local = np.argsort(p[finite])
    ordered = indices[ordered_local]
    running = 0.0
    total = ordered.size
    for rank, index in enumerate(ordered):
        candidate = min(1.0, (total - rank) * p[index])
        running = max(running, candidate)
        result[index] = running
    return result


def descriptive_statistics(
    core: ModuleType,
    data: AlignedData,
    config: RealDataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cash = data.cash_log_returns.reindex(data.risk_log_returns.index)
    rows = []
    for ticker in config.tickers:
        values = data.risk_log_returns[ticker].dropna()
        excess = values - cash.reindex(values.index)
        annual_return = math.exp(float(values.mean()) * 252.0) - 1.0
        annual_vol = float(values.std(ddof=1) * math.sqrt(252.0))
        excess_vol = float(excess.std(ddof=1))
        sharpe = float(excess.mean() / excess_vol * math.sqrt(252.0)) if excess_vol > 1e-12 else np.nan
        rows.append(
            {
                "Asset": ticker,
                "Label": config.asset_labels[ticker],
                "Group": config.asset_groups[ticker],
                "Observations": int(values.size),
                "Mean daily log return": float(values.mean()),
                "Median daily log return": float(values.median()),
                "Daily volatility": float(values.std(ddof=1)),
                "Annualised return": annual_return,
                "Annualised volatility": annual_vol,
                "Sharpe ratio": sharpe,
                "Skewness": float(skew(values, bias=False)),
                "Excess kurtosis": float(kurtosis(values, fisher=True, bias=False)),
                "Minimum daily return": float(values.min()),
                "Maximum daily return": float(values.max()),
                "Daily VaR 95%": float(np.quantile(values, 0.05)),
                "Daily CVaR 95%": _empirical_cvar(values),
                "Maximum drawdown": _maximum_drawdown(values),
            }
        )
    summary = pd.DataFrame(rows)

    tests = core.run_statistical_tests(data.risk_log_returns)
    stationarity_rows = []
    for ticker in config.tickers:
        returns = data.risk_log_returns[ticker].dropna()
        log_price = np.log(data.prices[ticker].dropna())
        row: dict[str, Any] = {"Asset": ticker}
        if adfuller is not None:
            try:
                row["ADF return statistic"] = float(adfuller(returns, autolag="AIC")[0])
                row["ADF return p-value"] = float(adfuller(returns, autolag="AIC")[1])
                row["ADF log-price statistic"] = float(adfuller(log_price, autolag="AIC")[0])
                row["ADF log-price p-value"] = float(adfuller(log_price, autolag="AIC")[1])
            except Exception:
                row.update({"ADF return statistic": np.nan, "ADF return p-value": np.nan, "ADF log-price statistic": np.nan, "ADF log-price p-value": np.nan})
        if kpss is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    kpss_return = kpss(returns, regression="c", nlags="auto")
                    kpss_price = kpss(log_price, regression="ct", nlags="auto")
                row["KPSS return statistic"] = float(kpss_return[0])
                row["KPSS return p-value"] = float(kpss_return[1])
                row["KPSS log-price statistic"] = float(kpss_price[0])
                row["KPSS log-price p-value"] = float(kpss_price[1])
            except Exception:
                row.update({"KPSS return statistic": np.nan, "KPSS return p-value": np.nan, "KPSS log-price statistic": np.nan, "KPSS log-price p-value": np.nan})
        stationarity_rows.append(row)
    stationarity = pd.DataFrame(stationarity_rows)
    merged = tests.merge(stationarity, on="Asset", how="left")
    for column in ["ADF return p-value", "ADF log-price p-value", "KPSS return p-value", "KPSS log-price p-value"]:
        if column in merged:
            merged[f"{column} Holm"] = _holm_adjust(merged[column])
    return summary, merged


def phase_episodes(phases: pd.DataFrame) -> pd.DataFrame:
    labels = phases["Phase"].fillna("Unclassified")
    group = labels.ne(labels.shift()).cumsum()
    rows = []
    for _, frame in phases.assign(_group=group).groupby("_group"):
        label = str(frame["Phase"].iloc[0])
        rows.append({
            "Phase": label, "Start": frame.index[0], "End": frame.index[-1], "Trading days": int(frame.shape[0]),
            "Benchmark cumulative return": float(np.expm1(frame["Log return"].sum())), "Minimum drawdown": float(frame["Drawdown"].min()),
            "Mean annualised volatility": float(frame["21-day annualised volatility"].mean()), "Mean VIX": float(frame["VIX"].mean()),
            "Mean high-yield spread": float(frame["High-yield spread"].mean()),
        })
    return pd.DataFrame(rows)


def phase_summary(phases: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, frame in phases.groupby("Phase", dropna=False):
        values = frame["Log return"].dropna()
        rows.append({
            "Phase": label, "Trading days": int(values.size), "Fraction of sample": float(values.size / max(1, phases["Log return"].notna().sum())),
            "Benchmark annualised return": math.exp(float(values.mean()) * 252.0) - 1.0, "Benchmark annualised volatility": float(values.std(ddof=1) * math.sqrt(252.0)),
            "Benchmark Sharpe": float(values.mean() / values.std(ddof=1) * math.sqrt(252.0)) if values.std(ddof=1) > 1e-12 else np.nan,
            "Mean drawdown": float(frame["Drawdown"].mean()), "Worst drawdown": float(frame["Drawdown"].min()), "Mean VIX": float(frame["VIX"].mean()),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 5. Descriptive figures
# -----------------------------------------------------------------------------

def _shade_phases(ax: plt.Axes, phases: pd.Series) -> None:
    palette = {"Bull": ("#2ca02c", 0.10), "Bear": ("#d62728", 0.12), "Sideways": ("#7f7f7f", 0.05)}
    labels = phases.fillna("Sideways")
    group = labels.ne(labels.shift()).cumsum()
    for _, segment in labels.groupby(group):
        label = str(segment.iloc[0])
        colour, alpha = palette.get(label, ("#7f7f7f", 0.03))
        ax.axvspan(segment.index[0], segment.index[-1], color=colour, alpha=alpha)


def plot_descriptive_analysis(data: AlignedData, config: RealDataConfig, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    normalised = data.prices / data.prices.iloc[0] * 100.0
    fig, ax = plt.subplots(figsize=(13, 7))
    for ticker in config.tickers:
        ax.plot(normalised.index, normalised[ticker], linewidth=1.0, label=ticker)
    ax.set_yscale("log")
    ax.set_title("Adjusted ETF prices normalised to 100")
    ax.set_ylabel("Normalised adjusted price, logarithmic scale")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=45) 
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "normalised_adjusted_prices.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    phase = data.phases
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    _shade_phases(axes[0], phase["Phase"])
    axes[0].plot(phase.index, phase["Price"], linewidth=1.2, label=config.benchmark_ticker)
    axes[0].plot(phase.index, phase["MA 200"], linewidth=1.0, linestyle="--", label="200-day moving average")
    axes[0].set_yscale("log")
    axes[0].set_title("Causal bull, bear, and sideways classification")
    axes[0].set_ylabel("Adjusted price")
    
    line_handles, line_labels = axes[0].get_legend_handles_labels()
    leg1 = axes[0].legend(handles=line_handles, loc="upper left")
    axes[0].add_artist(leg1)
    
    bull_patch = mpatches.Patch(color='#2ca02c', alpha=0.10, label='Bull Phase')
    bear_patch = mpatches.Patch(color='#d62728', alpha=0.12, label='Bear Phase')
    sideways_patch = mpatches.Patch(color='#7f7f7f', alpha=0.05, label='Sideways Phase')
    axes[0].legend(handles=[bull_patch, bear_patch, sideways_patch], loc="lower center", ncol=3)
    
    axes[0].grid(alpha=0.25)
    axes[1].plot(phase.index, phase["VIX"], linewidth=1.0, label="VIX")
    axes[1].axhline(30.0, linestyle="--", linewidth=0.9, label="Stress threshold")
    axes[1].set_ylabel("VIX")
    axes[1].legend(loc="upper left")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "benchmark_market_phases.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    if sns is not None:
        corr_matrix = data.risk_log_returns.corr(method='spearman')
        clustermap = sns.clustermap(corr_matrix, cmap='coolwarm', center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, linewidths=0.5, figsize=(12, 10), cbar_kws={'label': 'Spearman Correlation'})
        plt.setp(clustermap.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
        clustermap.fig.suptitle("Hierarchical Clustered Correlation Matrix (Spearman Rank)", y=1.02, fontweight='bold')
        clustermap.savefig(figure_dir / "return_correlation_clustered.png", dpi=260, bbox_inches="tight")
        plt.close(clustermap.fig)
    else:
        correlation = data.risk_log_returns.corr()
        fig, ax = plt.subplots(figsize=(9, 8))
        image = ax.imshow(correlation.to_numpy(), vmin=-1.0, vmax=1.0, aspect="equal")
        ax.set_xticks(np.arange(len(correlation.columns)))
        ax.set_yticks(np.arange(len(correlation.index)))
        ax.set_xticklabels(correlation.columns, rotation=45, ha="right")
        ax.set_yticklabels(correlation.index)
        for i in range(correlation.shape[0]):
            for j in range(correlation.shape[1]):
                ax.text(j, i, f"{correlation.iloc[i, j]:.2f}", ha="center", va="center", fontsize=7)
        ax.set_title("Full-sample Pearson correlation of daily log returns")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(figure_dir / "return_correlation_heatmap.png", dpi=260, bbox_inches="tight")
        plt.close(fig)

    rolling = data.risk_log_returns.rolling(63, min_periods=42).corr(data.risk_log_returns[config.benchmark_ticker])
    fig, ax = plt.subplots(figsize=(13, 7))
    for ticker in config.tickers:
        if ticker == config.benchmark_ticker: continue
        ax.plot(rolling.index, rolling[ticker], linewidth=0.9, label=ticker)
    ax.axhline(0.0, linewidth=0.8, linestyle=":")
    ax.set_ylim(-1.0, 1.0)
    ax.set_title(f"63-day rolling correlations with {config.benchmark_ticker}")
    ax.set_ylabel("Correlation")
    ax.legend(ncol=4, fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "rolling_correlations_with_benchmark.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    macro_columns = [column for column in ("VIXCLS", "T10Y3M", "BAMLH0A0HYM2", "DCOILBRENTEU") if column in data.macro.columns]
    if macro_columns:
        fig, axes = plt.subplots(len(macro_columns), 1, figsize=(13, 2.5 * len(macro_columns)), sharex=True)
        if len(macro_columns) == 1: axes = [axes]
        for ax, column in zip(axes, macro_columns):
            ax.plot(data.macro.index, data.macro[column], linewidth=1.0)
            ax.set_ylabel(column)
            ax.grid(alpha=0.25)
        axes[0].set_title("Macro-financial indicators")
        fig.tight_layout()
        fig.savefig(figure_dir / "macro_financial_indicators.png", dpi=260, bbox_inches="tight")
        plt.close(fig)

    wealth = np.exp(data.risk_log_returns.cumsum())
    drawdowns = wealth / wealth.cummax() - 1.0
    fig, ax = plt.subplots(figsize=(13, 7))
    for ticker in config.tickers:
        ax.plot(drawdowns.index, drawdowns[ticker], linewidth=0.9, label=ticker)
    ax.set_title("Asset-level drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend(ncol=4, fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "asset_drawdowns.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    if DiscreteBayesianNetwork is not None and KBinsDiscretizer is not None:
        try:
            discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='kmeans')
            df_discrete = pd.DataFrame(discretizer.fit_transform(data.risk_log_returns.dropna()), columns=data.risk_log_returns.columns, index=data.risk_log_returns.dropna().index).astype(int)
            hc_search = HillClimbSearch(df_discrete)
            best_model_structure = hc_search.estimate(scoring_method='bic-g', show_progress=False)
            causal_dag = DiscreteBayesianNetwork(best_model_structure.edges())
            causal_dag.fit(df_discrete, estimator=MaximumLikelihoodEstimator)
            
            target = config.benchmark_ticker
            if target in causal_dag.nodes():
                parents = causal_dag.get_parents(target)
                children = causal_dag.get_children(target)
                spouses = list(set([p for c in children for p in causal_dag.get_parents(c)]) - {target})

                G = nx.DiGraph(causal_dag.edges())
                pos = nx.spring_layout(G, k=1.5, seed=42)
                node_colors = ['#8B0000' if n == target else '#228B22' if n in parents else '#4682B4' if n in children else '#DAA520' if n in spouses else '#E0E0E0' for n in G.nodes()]

                fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
                nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=node_colors, edgecolors='black', ax=ax)
                nx.draw_networkx_edges(G, pos, edge_color='lightgray', arrows=True, arrowsize=15, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
                
                legend_elements = [mpatches.Patch(facecolor='#8B0000', label='Target (Benchmark)'), mpatches.Patch(facecolor='#228B22', label='Parents (Direct Causes)'), mpatches.Patch(facecolor='#4682B4', label='Children (Direct Effects)'), mpatches.Patch(facecolor='#DAA520', label='Spouses (Co-parents)'), mpatches.Patch(facecolor='#E0E0E0', label='Independent Nodes')]
                ax.legend(handles=legend_elements, loc='upper right', title="Markov Blanket Topology")
                ax.set_title(f"Markov Blanket Isolation for {target}", fontweight='bold')
                plt.axis('off')
                fig.savefig(figure_dir / "causal_markov_blanket.png", dpi=260, bbox_inches="tight")
                plt.close(fig)
        except Exception as e:
            LOGGER.warning(f"Causal modeling failed: {e}")

# -----------------------------------------------------------------------------
# 6. Model configuration and execution
# -----------------------------------------------------------------------------

def build_model_config(
    sim: ModuleType,
    real_config: RealDataConfig,
    *,
    n_samples: int,
    n_risk_assets: int,
    train_end_index: int,
    output_dir: Path,
) -> Any:
    base = sim.SimulationConfig()
    risk_degree = min(int(base.graph_maximum_degree), n_risk_assets - 1)
    predictive_degree = min(int(base.predictive_maximum_in_degree), n_risk_assets - 1)
    risk_density = min(float(base.graph_maximum_density), risk_degree / max(1, n_risk_assets - 1))
    predictive_density = min(float(base.predictive_maximum_density), predictive_degree / max(1, n_risk_assets - 1))
    policy_seeds = real_config.policy_seeds[:1] if real_config.quick else real_config.policy_seeds

    fraction = (train_end_index + 0.25) / n_samples
    if int(math.floor(n_samples * fraction)) != train_end_index:
        fraction = np.nextafter((train_end_index + 1.0) / n_samples, 0.0)

    updates: dict[str, Any] = {
        "n_samples": int(n_samples), "n_risk_assets": int(n_risk_assets),
        "train_fraction": float(fraction), "validation_fraction": real_config.validation_fraction,
        "scenario": "graph_predictive", "scenarios": ("graph_predictive",),
        "market_seeds": (17,), "policy_seeds": policy_seeds,
        "graph_maximum_degree": risk_degree, "graph_maximum_density": risk_density,
        "predictive_maximum_in_degree": predictive_degree, "predictive_maximum_density": predictive_density,
        "transaction_cost": real_config.transaction_cost, "slippage_coefficient": real_config.slippage_coefficient,
        "impact_coefficient": real_config.impact_coefficient, "maximum_cash": real_config.maximum_cash,
        "ppo_episodes": int(real_config.ppo_episodes), "benchmark_epochs": int(real_config.benchmark_epochs),
        "periods_per_year": int(real_config.periods_per_year), "continue_on_error": bool(real_config.continue_on_error),
        "output_dir": output_dir,
    }

    if real_config.quick:
        updates.update({
            "ppo_episodes": 1, "ppo_update_epochs": 1, "episode_length": min(int(base.episode_length), 32),
            "batch_size": min(int(base.batch_size), 16), "benchmark_epochs": 1, "benchmark_minimum_epochs": 1,
            "benchmark_validation_patience": 1, "hidden_gru": min(int(base.hidden_gru), 8), "hidden_gat": min(int(base.hidden_gat), 8),
            "hidden_tcmac": min(int(base.hidden_tcmac), 8), "validation_interval": 1, "validation_patience": 1,
            "critic_updates_per_actor": 1, "encoder_freeze_episodes": 0, "behaviour_epochs": 1, "behaviour_batches_per_epoch": 1,
            "representation_pretrain_epochs": 1, "representation_batches_per_epoch": 1, "future_risk_horizon": min(int(base.future_risk_horizon), 10),
            "graph_window": min(int(base.graph_window), 120), "fast_graph_window": min(int(base.fast_graph_window), 60),
            "graph_min_history": min(int(base.graph_min_history), 50), "fast_graph_min_history": min(int(base.fast_graph_min_history), 36),
            "graph_update_interval": max(int(base.graph_update_interval), 160), "graph_alpha_refresh_interval": max(int(base.graph_alpha_refresh_interval), 320),
            "predictive_lags": 1, "use_multiscale_graphs": False, "graph_bootstrap_replicates": 1, "rolling_graph_bootstrap_replicates": 0,
            "predictive_bootstrap_replicates": 2, "predictive_report_bootstrap_replicates": 4, "predictive_null_replicates": 8,
            "predictive_report_null_replicates": 12, "mcs_bootstraps": min(int(base.mcs_bootstraps), 20),
        })

    candidate = replace(base, **updates)
    candidate.validate()
    return candidate


def _run_window(
    core: ModuleType,
    sim: ModuleType,
    returns: pd.DataFrame,
    phases: pd.DataFrame,
    train_end_index: int,
    real_config: RealDataConfig,
    device: torch.device,
    window_label: str,
    output_dir: Path,
) -> WindowRun:
    model_config = build_model_config(sim, real_config, n_samples=len(returns), n_risk_assets=len(real_config.tickers), train_end_index=train_end_index, output_dir=output_dir)
    true_regimes = phases["Reference stress"].reindex(returns.index).fillna(0).astype(int).to_numpy()
    
    market = sim.MarketData(
        log_returns=returns.to_numpy(dtype=float), true_regimes=true_regimes,
        parameters={"type": "real_data", "tickers": list(real_config.tickers), "cash": "Cash", "start": returns.index.min().isoformat(), "end": returns.index.max().isoformat(), "out_of_sample_start": returns.index[train_end_index].isoformat()}
    )
    prepared = sim.prepare_data(core, market, model_config, seed=17, device=device)
    oos_dates = returns.index[model_config.train_end :]

    hmm_frame = pd.DataFrame(prepared.hmm_probabilities[model_config.train_end :], index=oos_dates, columns=[f"HMM state {index}" for index in range(model_config.n_regimes)])
    hmm_frame["Reference stress"] = true_regimes[model_config.train_end :]
    hmm_frame["Market phase"] = phases["Phase"].reindex(oos_dates).to_numpy()
    hmm_frame["Window"] = window_label

    evaluations, performance_frames, policy_frames, benchmark_frames, time_rows = {}, [], [], [], []
    seeds = real_config.policy_seeds[:1] if real_config.quick else real_config.policy_seeds
    for seed in seeds:
        LOGGER.info("%s | policy seed %d", window_label, seed)
        try:
            core.seed_everything(seed, deterministic=True)
            classical = sim.build_classical_benchmarks(core, market, prepared, model_config, seed)
            bundle = sim.train_benchmark_models(core, market, prepared, model_config, device, seed)
            teacher_path = sim.build_causal_teacher_path(core, bundle, classical, market, prepared, model_config, device)[0] if bool(model_config.use_behaviour_cloning) else sim.build_policy_base_path(classical, prepared, model_config)

            agents, policy_diagnostic_parts, policy_times = {}, [], {}
            for temporal_mode in real_config.temporal_modes:
                method = TEMPORAL_MODE_LABELS[temporal_mode]
                agent, diagnostics_frame, elapsed, multipliers = _train_policy_with_temporal_mode(core, sim, market, prepared, teacher_path, model_config, device, seed, label=method, temporal_mode=temporal_mode, use_risk_graph=bool(model_config.use_risk_graph), use_predictive_graph=bool(model_config.use_predictive_graph))
                agents[method] = agent
                policy_diagnostic_parts.append(diagnostics_frame)
                policy_times[method] = (elapsed, multipliers)

            evaluation = sim.evaluate_methods(core, market, prepared, agents, bundle, classical, teacher_path, model_config, device)
            performance = sim.build_metrics_table(core, evaluation, model_config)
            performance["Policy seed"], performance["Window"], performance["OOS start"], performance["OOS end"] = seed, window_label, oos_dates.min(), oos_dates.max()
            performance_frames.append(performance)

            policy_frames.append(pd.concat(policy_diagnostic_parts, ignore_index=True, sort=False).assign(**{"Policy seed": seed, "Window": window_label}))
            benchmark_frames.append(bundle.training_diagnostics.assign(**{"Policy seed": seed, "Window": window_label}))

            for method, (seconds, multipliers) in policy_times.items():
                time_rows.append({"Window": window_label, "Policy seed": seed, "Method": method, "Training seconds": float(seconds), "Final multipliers": json.dumps(multipliers, default=_json_default, sort_keys=True)})
            for method, seconds in bundle.training_times.items():
                time_rows.append({"Window": window_label, "Policy seed": seed, "Method": method, "Training seconds": float(seconds), "Final multipliers": ""})
            evaluations[seed] = evaluation
        except Exception as exc:
            LOGGER.error("Window %s, seed %d failed: %s", window_label, seed, exc)
            if not real_config.continue_on_error: raise
        finally:
            if device.type == "cuda": torch.cuda.empty_cache()

    return WindowRun(
        label=window_label, dates=oos_dates, evaluations=evaluations, fold_performance=pd.concat(performance_frames, ignore_index=True),
        policy_training=pd.concat(policy_frames, ignore_index=True, sort=False), benchmark_training=pd.concat(benchmark_frames, ignore_index=True, sort=False),
        hmm_probabilities=hmm_frame, hmm_diagnostics=dict(prepared.hmm_diagnostics), risk_path=prepared.dynamic_risk_adjacencies[model_config.train_end :].detach().cpu().numpy(),
        predictive_path=prepared.dynamic_predictive_adjacencies[model_config.train_end :].detach().cpu().numpy(), feature_names=prepared.feature_names, training_times=pd.DataFrame(time_rows),
    )


def _evaluation_folds(index: pd.DatetimeIndex, config: RealDataConfig) -> list[tuple[str, int, int, int]]:
    oos_start = int(index.searchsorted(pd.Timestamp(config.out_of_sample_start), side="left"))
    if oos_start <= 0 or oos_start >= len(index): raise ValueError("out_of_sample_start is outside the aligned return sample.")

    def bounded_fold(label: str, test_start: int, test_end: int, desired_training: int) -> tuple[str, int, int, int] | None:
        test_size = int(test_end - test_start)
        if test_size < config.minimum_annual_test_observations: return None
        training_size = min(int(test_start), int(desired_training), max(1, int(math.floor(8.5 * test_size))))
        if training_size < config.minimum_training_observations: return None
        return label, int(test_start - training_size), int(test_start), int(test_end)

    if config.evaluation_mode == "fixed_holdout":
        fold = bounded_fold("fixed holdout", oos_start, len(index), desired_training=oos_start)
        if fold is None: raise RuntimeError("The fixed holdout does not contain enough observations.")
        return [fold]

    folds = []
    desired_training = int(config.annual_training_years * config.periods_per_year)
    for year in range(pd.Timestamp(config.out_of_sample_start).year, index[-1].year + 1):
        start = int(index.searchsorted(pd.Timestamp(f"{year}-01-01"), side="left"))
        end = min(int(index.searchsorted(pd.Timestamp(f"{year + 1}-01-01"), side="left")), len(index))
        if start >= end: continue
        fold = bounded_fold(str(year), start, end, desired_training)
        if fold is not None: folds.append(fold)
    if not folds: raise RuntimeError("No annual-refit fold satisfies the predeclared sample requirements.")
    return folds


def run_model_application(core: ModuleType, sim: ModuleType, data: AlignedData, config: RealDataConfig, device: torch.device, output_dir: Path) -> dict[str, Any]:
    folds = _evaluation_folds(data.log_returns.index, config)
    window_runs, fold_performance, policy_training, benchmark_training, hmm_frames, training_times = [], [], [], [], [], []

    for label, history_start, start, end in folds:
        subset_returns = data.log_returns.iloc[history_start:end].copy()
        subset_phases = data.phases.reindex(subset_returns.index)
        local_train_end = start - history_start
        run = _run_window(core, sim, subset_returns, subset_phases, local_train_end, config, device, label, output_dir / "windows" / _safe_name(label))
        window_runs.append(run)
        fold_performance.append(run.fold_performance)
        policy_training.append(run.policy_training)
        benchmark_training.append(run.benchmark_training)
        hmm_frames.append(run.hmm_probabilities)
        training_times.append(run.training_times)

    successful_seeds = sorted(set.intersection(*(set(run.evaluations) for run in window_runs)))
    
    # Require strictly identical completed seeds
    expected_seeds = set(config.policy_seeds[:1] if config.quick else config.policy_seeds)
    if set(successful_seeds) != expected_seeds:
        raise RuntimeError(f"Missing policy seeds: {sorted(expected_seeds - set(successful_seeds))}")

    combined_evaluations: dict[int, Any] = {}
    combined_dates = pd.DatetimeIndex(np.concatenate([run.dates.to_numpy() for run in window_runs]))
    for seed in successful_seeds:
        fields = {field: {method: np.concatenate([getattr(run.evaluations[seed], field)[method] for run in window_runs], axis=0) for method in sim.METHOD_ORDER} for field in ("returns", "turnover", "costs", "weights", "hhi", "cash_weights")}
        combined_evaluations[seed] = sim.EvaluationResult(**fields)

    final_performance = []
    for seed, evaluation in combined_evaluations.items():
        metric_config = replace(sim.SimulationConfig(), periods_per_year=config.periods_per_year, n_risk_assets=len(config.tickers), n_samples=max(360, len(combined_dates) + 1))
        frame = sim.build_metrics_table(core, evaluation, metric_config)
        frame["Policy seed"] = seed
        final_performance.append(frame)

    return {
        "dates": combined_dates,
        "evaluations": combined_evaluations,
        "performance_by_seed": pd.concat(final_performance, ignore_index=True),
        "fold_performance": pd.concat(fold_performance, ignore_index=True),
        "policy_training": pd.concat(policy_training, ignore_index=True, sort=False),
        "benchmark_training": pd.concat(benchmark_training, ignore_index=True, sort=False),
        "hmm_probabilities": pd.concat(hmm_frames).sort_index(),
        "hmm_diagnostics": pd.DataFrame([{"Window": run.label, **run.hmm_diagnostics} for run in window_runs]),
        "training_times": pd.concat(training_times, ignore_index=True),
        "window_runs": window_runs,
        "successful_seeds": successful_seeds,
    }

# -----------------------------------------------------------------------------
# 7. Aggregation and inferential summaries
# -----------------------------------------------------------------------------

def _path_metrics(log_returns: np.ndarray, cash_log_returns: np.ndarray, turnover: np.ndarray | None = None) -> dict[str, float]:
    ret_simple = np.expm1(log_returns)
    cash_simple = np.expm1(cash_log_returns)
    excess = ret_simple - cash_simple
    turn = np.asarray(turnover, dtype=float) if turnover is not None else np.zeros(ret_simple.size)
    
    ann_ret = math.exp(float(np.mean(log_returns)) * 252.0) - 1.0
    excess_vol = float(np.std(excess, ddof=1)) if excess.size > 1 else 1e-12
    sharpe = float(np.mean(excess) / excess_vol * math.sqrt(252.0))
    
    downside = np.minimum(excess, 0.0)
    downside_dev = math.sqrt(float(np.mean(downside**2))) if excess.size > 0 else 1e-12
    sortino = float(np.mean(excess) / downside_dev * math.sqrt(252.0)) if downside_dev > 1e-12 else 0.0
    
    return {
        "Annualised return": ann_ret,
        "Sharpe ratio": sharpe,
        "Sortino ratio": sortino,
        "Daily CVaR 95%": _empirical_cvar(log_returns),
        "Maximum drawdown": _maximum_drawdown(log_returns),
        "Mean turnover": float(turn.mean()) if turn.size else 0.0,
    }

def _policy_seed_summary(performance: pd.DataFrame) -> pd.DataFrame:
    metrics = ["Annualised return", "Annualised volatility", "Sharpe ratio", "Sortino ratio", "Maximum drawdown", "Daily CVaR 95%", "Final wealth", "Mean turnover", "Total transaction cost", "Mean HHI", "Mean cash weight"]
    rows = []
    for method, frame in performance.groupby("Method", sort=False):
        for metric in metrics:
            if metric in frame.columns:
                values = pd.to_numeric(frame[metric], errors="coerce").dropna()
                rows.append({"Method": method, "Metric": metric, "Policy seeds": int(values.size), "Mean": float(values.mean()), "Median": float(values.median()), "Standard deviation": float(values.std(ddof=1)) if values.size > 1 else 0.0, "10th percentile": float(values.quantile(0.10)), "90th percentile": float(values.quantile(0.90)), "Minimum": float(values.min()), "Maximum": float(values.max())})
    return pd.DataFrame(rows)

def _average_paths(evaluations: Mapping[int, Any], method_order: Sequence[str]) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {"returns": {}, "turnover": {}, "costs": {}, "weights": {}, "hhi": {}, "cash_weights": {}}
    for method in method_order:
        # Simulate an equal capital ensemble by averaging simple returns
        simple_returns = np.stack([np.expm1(ev.returns[method]) for ev in evaluations.values()])
        result["returns"][method] = np.log1p(simple_returns.mean(axis=0))
        
        # Average weights to produce the ensemble portfolio
        weights = np.stack([ev.weights[method] for ev in evaluations.values()])
        result["weights"][method] = weights.mean(axis=0)
        
        for field in ("turnover", "costs", "hhi", "cash_weights"):
            stacked = np.stack([getattr(ev, field)[method] for ev in evaluations.values()])
            result[field][method] = stacked.mean(axis=0)
    return result

def _circular_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    blocks = int(math.ceil(n / block_length))
    starts = rng.integers(0, n, size=blocks)
    return np.concatenate([(start + np.arange(block_length)) % n for start in starts])[:n]

def paired_block_bootstrap(
    averaged_paths: Mapping[str, Mapping[str, np.ndarray]],
    cash_returns: np.ndarray,
    method_order: Sequence[str],
    *,
    repetitions: int,
    block_length: int,
    confidence_level: float = 0.95,
    seed: int = 20260723,
    reference: str = "X-GAT-DRL (Hybrid)",
) -> pd.DataFrame:
    if reference not in method_order: raise ValueError(f"Bootstrap reference {reference!r} is not in method_order.")
    metrics = ["Annualised return", "Sharpe ratio", "Daily CVaR 95%", "Maximum drawdown", "Mean turnover"]
    n = len(averaged_paths["returns"][reference])
    if n < 2: raise ValueError("The OOS path is too short for block bootstrap inference.")
    block_length = min(max(2, int(block_length)), n)
    repetitions = max(1, int(repetitions))
    alpha = 0.5 * (1.0 - confidence_level)
    rng = np.random.default_rng(seed)
    rows = []

    def effect(comparator: str, indices: np.ndarray | None = None) -> dict[str, float]:
        selector = slice(None) if indices is None else indices
        ref_metrics = _path_metrics(averaged_paths["returns"][reference][selector], cash_returns[selector], averaged_paths["turnover"][reference][selector])
        cmp_metrics = _path_metrics(averaged_paths["returns"][comparator][selector], cash_returns[selector], averaged_paths["turnover"][comparator][selector])
        return {m: (cmp_metrics[m] - ref_metrics[m] if m == "Mean turnover" else ref_metrics[m] - cmp_metrics[m]) for m in metrics}

    for comparator in method_order:
        if comparator == reference: continue
        observed = effect(comparator)
        distributions = {metric: np.empty(repetitions, dtype=float) for metric in metrics}
        for replicate in range(repetitions):
            indices = _circular_block_indices(n, block_length, rng)
            values = effect(comparator, indices)
            for metric in metrics: distributions[metric][replicate] = values[metric]

        for metric in metrics:
            distribution = distributions[metric]
            probability_positive = float(np.mean(distribution > 0.0))
            centred = distribution - float(np.mean(distribution))
            p_value = float((1.0 + np.sum(np.abs(centred) >= abs(observed[metric]))) / (distribution.size + 1.0))
            rows.append({
                "Reference": reference, "Comparator": comparator, "Metric": metric, "Direction-adjusted effect": observed[metric],
                "Confidence low": float(np.quantile(distribution, alpha)), "Confidence high": float(np.quantile(distribution, 1.0 - alpha)),
                "Probability reference favourable": probability_positive, "Two-sided centred-bootstrap p-value": p_value,
                "Bootstrap repetitions": repetitions, "Block length": block_length,
            })

    result = pd.DataFrame(rows)
    result["Holm p-value"] = np.nan
    for _, indices in result.groupby("Metric").groups.items():
        result.loc[indices, "Holm p-value"] = _holm_adjust(result.loc[indices, "Two-sided centred-bootstrap p-value"])
    return result

def _model_confidence_set(
    core: ModuleType,
    averaged_paths: Mapping[str, Mapping[str, np.ndarray]],
    cash_returns: np.ndarray,
    method_order: Sequence[str],
    block_length: int,
    *,
    bootstraps: int = 1_000,
) -> pd.DataFrame:
    # Use negative excess returns to measure comparative model loss
    losses = []
    for method in method_order:
        excess = np.expm1(averaged_paths["returns"][method]) - np.expm1(cash_returns)
        losses.append(-excess)
    losses_arr = np.column_stack(losses)
    
    survivors = core.compute_model_confidence_set(
        losses_arr, alpha=0.10, block_length=min(block_length, max(2, losses_arr.shape[0] // 2)),
        bootstraps=max(50, int(bootstraps)), random_state=20260723,
    )
    return pd.DataFrame({"Method": method_order, "MCS survivor": [index in survivors for index in range(len(method_order))]})

def performance_by_phase(
    dates: pd.DatetimeIndex, phases: pd.Series, averaged_paths: Mapping[str, Mapping[str, np.ndarray]],
    method_order: Sequence[str],
) -> pd.DataFrame:
    labels = phases.reindex(dates)
    rows = []
    for phase in ("Bull", "Sideways", "Bear"):
        mask = labels.eq(phase).to_numpy()
        if not mask.any(): continue
        for method in method_order:
            returns, turnover, cash = averaged_paths["returns"][method][mask], averaged_paths["turnover"][method][mask], averaged_paths["cash_weights"][method][mask]
            volatility = float(returns.std(ddof=1)) if returns.size > 1 else np.nan
            rows.append({
                "Phase": phase, "Method": method, "Trading days": int(mask.sum()), "Conditional annualised return": math.exp(float(returns.mean()) * 252.0) - 1.0,
                "Conditional annualised volatility": volatility * math.sqrt(252.0), "Conditional Sharpe": float(returns.mean() / volatility * math.sqrt(252.0)) if volatility and volatility > 1e-12 else np.nan,
                "Conditional CVaR 95%": _empirical_cvar(returns), "Mean turnover": float(turnover.mean()), "Mean cash weight": float(cash.mean()),
            })
    return pd.DataFrame(rows)

def event_window_performance(
    dates: pd.DatetimeIndex, averaged_paths: Mapping[str, Mapping[str, np.ndarray]], method_order: Sequence[str], events: Sequence[EventWindow],
) -> pd.DataFrame:
    rows = []
    for event in events:
        start = pd.Timestamp(event.start)
        end = pd.Timestamp(event.end) if event.end is not None else dates.max()
        mask = (dates >= start) & (dates <= end)
        if not mask.any(): continue
        for method in method_order:
            returns, turnover, cash = averaged_paths["returns"][method][mask], averaged_paths["turnover"][method][mask], averaged_paths["cash_weights"][method][mask]
            rows.append({
                "Event": event.name, "Category": event.category, "Start": dates[mask][0], "End": dates[mask][-1], "Method": method,
                "Trading days": int(mask.sum()), "Cumulative return": float(np.expm1(returns.sum())), "Annualised volatility": float(returns.std(ddof=1) * math.sqrt(252.0)) if returns.size > 1 else np.nan,
                "Daily CVaR 95%": _empirical_cvar(returns), "Maximum drawdown": _maximum_drawdown(returns), "Mean turnover": float(turnover.mean()), "Mean cash weight": float(cash.mean()),
            })
    return pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# 8. Graph summaries
# -----------------------------------------------------------------------------

def _graph_edge_tables(risk_paths: Sequence[np.ndarray], predictive_paths: Sequence[np.ndarray], tickers: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    risk = np.concatenate(risk_paths, axis=0)
    predictive = np.concatenate(predictive_paths, axis=0)
    if predictive.ndim == 3:
        predictive = predictive[:, None, :, :]
        predictive_lag_labels: tuple[str | int, ...] = ("Aggregated",)
    elif predictive.ndim == 4:
        predictive_lag_labels = tuple(range(1, predictive.shape[1] + 1))
    else: raise ValueError("Predictive adjacency paths must have shape [time, asset, asset] or [time, lag, asset, asset].")
    
    n = len(tickers)
    risk_rows = []
    for i in range(n):
        for j in range(i + 1, n):
            values = risk[:, i, j]
            risk_rows.append({"Asset 1": tickers[i], "Asset 2": tickers[j], "Mean signed weight": float(values.mean()), "Mean absolute weight": float(np.abs(values).mean()), "Edge persistence": float(np.mean(np.abs(values) > 1e-12))})
            
    predictive_rows = []
    for lag in range(predictive.shape[1]):
        for target in range(n):
            for source in range(n):
                if source == target: continue
                values = predictive[:, lag, target, source]
                predictive_rows.append({"Lag": predictive_lag_labels[lag], "Source": tickers[source], "Target": tickers[target], "Mean signed weight": float(values.mean()), "Mean absolute weight": float(np.abs(values).mean()), "Edge persistence": float(np.mean(np.abs(values) > 1e-12))})
    
    risk_table = pd.DataFrame(risk_rows).sort_values(["Edge persistence", "Mean absolute weight"], ascending=False) if risk_rows else pd.DataFrame(risk_rows)
    predictive_table = pd.DataFrame(predictive_rows).sort_values(["Edge persistence", "Mean absolute weight"], ascending=False) if predictive_rows else pd.DataFrame(predictive_rows)
        
    off_diagonal = ~np.eye(n, dtype=bool)
    risk_off_diagonal = risk[:, off_diagonal]
    predictive_off_diagonal = predictive[:, :, off_diagonal]
    summary = pd.DataFrame([
        {"Graph": "Risk", "Mean density": float(np.mean(np.abs(risk_off_diagonal) > 1e-12)), "Mean absolute off-diagonal weight": float(np.abs(risk_off_diagonal).mean())},
        {"Graph": "Predictive", "Mean density": float(np.mean(np.abs(predictive_off_diagonal) > 1e-12)), "Mean absolute off-diagonal weight": float(np.abs(predictive_off_diagonal).mean())},
    ])
    return risk_table, predictive_table, summary

def _plot_network(matrix: np.ndarray, tickers: Sequence[str], path: Path, title: str, *, directed: bool) -> None:
    values = np.asarray(matrix, dtype=float).copy()
    np.fill_diagonal(values, 0.0)
    
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(tickers)
    n = len(tickers)
    
    # Filter 0 weights before finding quantile threshold
    flat_vals = np.abs(values[~np.eye(n, dtype=bool)])
    nonzero = flat_vals[flat_vals > 1e-12]
    threshold = np.percentile(nonzero, 70) if nonzero.size > 0 else 0.0

    if directed:
        for target in range(n):
            for source in range(n):
                val = values[target, source]
                if source != target and abs(val) > max(1e-12, threshold):
                    graph.add_edge(tickers[source], tickers[target], weight=float(val))
    else:
        for i in range(n):
            for j in range(i + 1, n):
                val = 0.5 * (values[i, j] + values[j, i])
                if abs(val) > max(1e-12, threshold):
                    graph.add_edge(tickers[i], tickers[j], weight=float(val))

    positions = nx.circular_layout(graph)
    fig, ax = plt.subplots(figsize=(11, 11))
    
    asset_groups = {"SPY": "Equity", "QQQ": "Equity", "IWM": "Equity", "EFA": "Equity", "EEM": "Equity", "IEF": "Treasury", "TLT": "Treasury", "LQD": "Credit", "HYG": "Credit", "GLD": "Commodity", "DBC": "Commodity", "VNQ": "Real Estate"}
    group_colors = {"Equity": "#1f77b4", "Treasury": "#2ca02c", "Credit": "#ff7f0e", "Commodity": "#d62728", "Real Estate": "#9467bd"}
    node_colors = [group_colors.get(asset_groups.get(node, "Equity"), "#1f77b4") for node in graph.nodes()]

    nx.draw_networkx_nodes(graph, positions, node_size=2200, node_color=node_colors, edgecolors="black", linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(graph, positions, font_size=10, font_weight="bold", font_color="white", ax=ax)
    
    positive = [(u, v) for u, v, d in graph.edges(data=True) if d["weight"] >= 0.0]
    negative = [(u, v) for u, v, d in graph.edges(data=True) if d["weight"] < 0.0]
    max_weight = max([abs(d["weight"]) for _, _, d in graph.edges(data=True)] + [1e-12])
    
    for edges, style in ((positive, "solid"), (negative, "dashed")):
        if not edges: continue
        widths = [1.5 + 3.5 * abs(graph[u][v]["weight"]) / max_weight for u, v in edges]
        edge_kwargs: dict[str, Any] = {"edgelist": edges, "width": widths, "style": style, "arrows": directed, "connectionstyle": "arc3,rad=0.08" if directed else "arc3", "edge_color": "gray" if style == "solid" else "orange", "ax": ax}
        if directed: edge_kwargs["arrowsize"] = 18
        nx.draw_networkx_edges(graph, positions, **edge_kwargs)

    legend_elements = [mpatches.Patch(color='gray', label='Strong Positive Link'), mpatches.Patch(color='orange', label='Strong Negative Link')]
    for group, color in group_colors.items(): legend_elements.append(mpatches.Patch(color=color, label=group))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, frameon=True, bbox_to_anchor=(1.15, 1.05))

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=260, bbox_inches="tight")
    plt.close(fig)

# -----------------------------------------------------------------------------
# 9. Model figures
# -----------------------------------------------------------------------------

def plot_model_results(
    dates: pd.DatetimeIndex,
    averaged_paths: Mapping[str, Mapping[str, np.ndarray]],
    performance_by_seed: pd.DataFrame,
    hmm_probabilities: pd.DataFrame,
    phases: pd.DataFrame,
    method_order: Sequence[str],
    tickers: Sequence[str],
    events: Sequence[EventWindow],
    figure_dir: Path,
    *,
    primary_xgat_method: str,
    temporal_methods: Sequence[str],
) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 7))
    for method in method_order:
        wealth = np.exp(np.cumsum(averaged_paths["returns"][method]))
        ax.plot(dates, wealth, linewidth=1.1 if method != primary_xgat_method else 2.2, label=method)
    ax.set_title("Out-of-sample cumulative wealth")
    ax.set_ylabel("Portfolio wealth")
    ax.legend(ncol=4, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "out_of_sample_cumulative_wealth.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    for method in temporal_methods:
        wealth = np.exp(np.cumsum(averaged_paths["returns"][method]))
        ax.plot(dates, wealth, linewidth=2.0 if method == primary_xgat_method else 1.4, label=method)
    ax.set_title("X-GAT temporal-encoder comparison")
    ax.set_ylabel("Portfolio wealth")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "xgat_temporal_mode_cumulative_wealth.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 7))
    for method in method_order:
        wealth = np.exp(np.cumsum(averaged_paths["returns"][method]))
        drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
        ax.plot(dates, drawdown, linewidth=1.1 if method != primary_xgat_method else 2.2, label=method)
    ax.set_title("Out-of-sample drawdowns")
    ax.set_ylabel("Drawdown")
    ax.legend(ncol=4, fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "out_of_sample_drawdowns.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    median_perf = performance_by_seed.groupby("Method", sort=False).median(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    maximum_turnover = max(float(median_perf["Mean turnover"].max()), 1e-12)
    
    offsets = [(10, 10), (-40, -15), (10, -15), (-40, 10), (0, 20), (0, -25), (20, 0), (-20, 0)]
    for i, method in enumerate(method_order):
        row = median_perf.loc[method]
        size = 70 + 300 * math.sqrt(max(float(row["Mean turnover"]), 0.0) / maximum_turnover)
        ax.scatter(row["Daily CVaR 95%"], row["Annualised return"], s=size, edgecolors="black", alpha=0.7)
        offset = offsets[i % len(offsets)]
        ax.annotate(method, (row["Daily CVaR 95%"], row["Annualised return"]), xytext=offset, textcoords="offset points", fontsize=9, bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, lw=0))
        
    ax.axhline(0.0, linestyle=":", linewidth=0.8, color='grey')
    ax.set_title("Return--CVaR Trade-off (Marker size proportional to turnover)")
    ax.set_xlabel("Daily CVaR 95% (closer to zero is better)")
    ax.set_ylabel("Annualised Return")
    ax.grid(alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(figure_dir / "return_cvar_tradeoff.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    weights = averaged_paths["weights"][primary_xgat_method]
    labels = list(tickers) + ["Cash"]
    fig, ax = plt.subplots(figsize=(15, 8)) 
    ax.stackplot(dates, weights.T, labels=labels, alpha=0.85)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{primary_xgat_method} Out-of-Sample Allocation")
    ax.set_ylabel("Portfolio Weight")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate(rotation=45)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=13, fontsize=9, frameon=False)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(figure_dir / "xgat_out_of_sample_allocation.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(dates, averaged_paths["cash_weights"][primary_xgat_method], linewidth=1.2, label="Cash weight")
    ax.plot(hmm_probabilities.index, hmm_probabilities["HMM state 1"], linewidth=1.0, label="Filtered crisis-state probability")
    ax.step(phases.reindex(dates).index, phases.reindex(dates)["Reference stress"], where="post", linewidth=0.8, label="Reference stress label")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Cash exposure, HMM crisis probability, and descriptive stress reference")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout(rect=[0, 0.05, 1, 1]) 
    fig.savefig(figure_dir / "cash_hmm_and_reference_stress.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5))
    for method in temporal_methods:
        ax.plot(dates, averaged_paths["cash_weights"][method], linewidth=1.1, label=method)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Cash exposure by X-GAT temporal encoder")
    ax.set_ylabel("Cash weight")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "xgat_temporal_mode_cash_weights.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    metrics = ["Annualised return", "Sharpe ratio", "Daily CVaR 95%", "Maximum drawdown", "Mean turnover"]
    for metric in metrics:
        data = [performance_by_seed.loc[performance_by_seed["Method"].eq(method), metric].to_numpy() for method in method_order]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.boxplot(data, tick_labels=method_order, showfliers=True)
        ax.set_title(f"Policy-seed dispersion: {metric}")
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(figure_dir / f"policy_seed_{_safe_name(metric).lower()}.png", dpi=260, bbox_inches="tight")
        plt.close(fig)

    for event in events:
        start = pd.Timestamp(event.start)
        end = pd.Timestamp(event.end) if event.end is not None else dates.max()
        mask = (dates >= start) & (dates <= end)
        if not mask.any(): continue
        event_dates = dates[mask]
        fig, ax = plt.subplots(figsize=(11, 6))
        for method in method_order:
            wealth = np.exp(np.cumsum(averaged_paths["returns"][method][mask]))
            ax.plot(event_dates, wealth, linewidth=1.0 if method != primary_xgat_method else 2.0, label=method)
        ax.set_title(event.name)
        ax.set_ylabel("Wealth from event start")
        ax.legend(ncol=4, fontsize=8, loc="best")
        ax.grid(alpha=0.25)
        fig.autofmt_xdate(rotation=45)
        fig.tight_layout()
        fig.savefig(figure_dir / f"event_{_safe_name(event.name).lower()}_wealth.png", dpi=260, bbox_inches="tight")
        plt.close(fig)
        
# -----------------------------------------------------------------------------
# 10. Output
# -----------------------------------------------------------------------------

def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")

def _write_manifests(root: Path) -> None:
    table_rows, figure_rows = [], []
    for path in sorted(root.rglob("*.csv")): table_rows.append({"File": str(path.relative_to(root)), "Bytes": path.stat().st_size, "SHA256": _sha256(path)})
    for path in sorted((root / "figures").rglob("*.png")): figure_rows.append({"File": str(path.relative_to(root)), "Bytes": path.stat().st_size, "SHA256": _sha256(path)})
    pd.DataFrame(table_rows).to_csv(root / "table_manifest.csv", index=False)
    pd.DataFrame(figure_rows).to_csv(root / "figure_manifest.csv", index=False)

def save_results(
    core: ModuleType, sim: ModuleType, data: AlignedData, descriptive: pd.DataFrame, tests: pd.DataFrame, source_manifest: pd.DataFrame,
    model: dict[str, Any], config: RealDataConfig, code_paths: Mapping[str, Path],
) -> None:
    root = config.output_dir
    tables, figures, paths, graphs, diagnostics = root / "tables", root / "figures", root / "paths", root / "graphs", root / "diagnostics"
    for directory in (tables, figures, paths, graphs, diagnostics): directory.mkdir(parents=True, exist_ok=True)

    data.prices.to_csv(tables / "adjusted_prices.csv")
    data.volumes.to_csv(tables / "trading_volume.csv")
    data.macro.to_csv(tables / "macro_financial_indicators.csv")
    data.log_returns.to_csv(tables / "daily_log_returns.csv")
    data.phases.to_csv(tables / "market_phase_daily.csv")
    data.data_quality.to_csv(tables / "data_quality_report.csv", index=False)
    descriptive.to_csv(tables / "descriptive_statistics.csv", index=False)
    tests.to_csv(tables / "distribution_and_stationarity_tests.csv", index=False)
    data.risk_log_returns.corr().to_csv(tables / "pearson_correlation.csv")
    data.risk_log_returns.corr(method="spearman").to_csv(tables / "spearman_correlation.csv")
    downside_mask = data.risk_log_returns[config.benchmark_ticker] <= data.risk_log_returns[config.benchmark_ticker].quantile(0.10)
    data.risk_log_returns.loc[downside_mask].corr().to_csv(tables / "downside_correlation.csv")
    phase_episodes(data.phases).to_csv(tables / "market_phase_episodes.csv", index=False)
    phase_summary(data.phases).to_csv(tables / "market_phase_summary.csv", index=False)
    source_manifest.to_csv(tables / "source_file_manifest.csv", index=False)

    performance = model["performance_by_seed"]
    performance.to_csv(tables / "performance_by_policy_seed.csv", index=False)
    seed_summary = _policy_seed_summary(performance)
    seed_summary.to_csv(tables / "performance_policy_seed_summary.csv", index=False)
    temporal_methods = tuple(TEMPORAL_MODE_LABELS[mode] for mode in config.temporal_modes)
    performance.loc[performance["Method"].isin(temporal_methods)].to_csv(tables / "xgat_temporal_mode_performance_by_seed.csv", index=False)
    seed_summary.loc[seed_summary["Method"].isin(temporal_methods)].to_csv(tables / "xgat_temporal_mode_summary.csv", index=False)
    model["fold_performance"].to_csv(tables / "fold_performance.csv", index=False)
    model["policy_training"].to_csv(diagnostics / "policy_training_diagnostics.csv", index=False)
    model["benchmark_training"].to_csv(diagnostics / "benchmark_training_diagnostics.csv", index=False)
    model["training_times"].to_csv(diagnostics / "training_times.csv", index=False)
    model["hmm_probabilities"].to_csv(paths / "hmm_filtered_probabilities.csv")
    model["hmm_diagnostics"].to_csv(diagnostics / "hmm_diagnostics.csv", index=False)

    averaged = _average_paths(model["evaluations"], sim.METHOD_ORDER)
    dates = model["dates"]
    
    oos_cash_returns = data.cash_log_returns.loc[dates].to_numpy()

    pd.DataFrame({method: averaged["returns"][method] for method in sim.METHOD_ORDER}, index=dates).to_csv(paths / "oos_daily_method_returns.csv")
    pd.DataFrame({method: np.exp(np.cumsum(averaged["returns"][method])) for method in sim.METHOD_ORDER}, index=dates).to_csv(paths / "oos_daily_method_wealth.csv")
    pd.DataFrame({method: (np.exp(np.cumsum(averaged["returns"][method])) / np.maximum.accumulate(np.exp(np.cumsum(averaged["returns"][method]))) - 1.0) for method in sim.METHOD_ORDER}, index=dates,).to_csv(paths / "oos_daily_method_drawdowns.csv")
    pd.DataFrame({method: averaged["turnover"][method] for method in sim.METHOD_ORDER}, index=dates).to_csv(paths / "oos_daily_method_turnover.csv")
    pd.DataFrame({method: averaged["cash_weights"][method] for method in sim.METHOD_ORDER}, index=dates).to_csv(paths / "oos_daily_method_cash_weights.csv")
    for temporal_mode in config.temporal_modes:
        method = TEMPORAL_MODE_LABELS[temporal_mode]
        weights = pd.DataFrame(averaged["weights"][method], index=dates, columns=[*config.tickers, "Cash"])
        weights.to_csv(paths / f"oos_xgat_{temporal_mode}_weights.csv")
        if temporal_mode == "hybrid": weights.to_csv(paths / "oos_xgat_weights.csv")

    paired = paired_block_bootstrap(
        averaged, oos_cash_returns, sim.METHOD_ORDER,
        repetitions=min(config.block_bootstraps, 200) if config.quick else config.block_bootstraps,
        block_length=config.block_length,
        confidence_level=config.confidence_level,
        reference=_primary_xgat_method(config.temporal_modes),
    )
    paired.to_csv(tables / "paired_moving_block_bootstrap.csv", index=False)
    
    _model_confidence_set(
        core, averaged, oos_cash_returns, sim.METHOD_ORDER, config.block_length, bootstraps=50 if config.quick else 1_000,
    ).to_csv(tables / "model_confidence_set.csv", index=False)
    
    performance_by_phase(dates, data.phases["Phase"], averaged, sim.METHOD_ORDER).to_csv(tables / "performance_by_market_phase.csv", index=False)
    event_window_performance(dates, averaged, sim.METHOD_ORDER, DEFAULT_EVENT_WINDOWS).to_csv(tables / "event_window_performance.csv", index=False)

    risk_paths = [run.risk_path for run in model["window_runs"]]
    predictive_paths = [run.predictive_path for run in model["window_runs"]]
    risk_edges, predictive_edges, graph_summary = _graph_edge_tables(risk_paths, predictive_paths, config.tickers)
    risk_edges.to_csv(graphs / "risk_graph_edges.csv", index=False)
    predictive_edges.to_csv(graphs / "predictive_graph_edges.csv", index=False)
    graph_summary.to_csv(graphs / "graph_summary.csv", index=False)

    risk_all = np.concatenate(risk_paths, axis=0)
    predictive_all = np.concatenate(predictive_paths, axis=0)
    average_risk = risk_all.mean(axis=0)
    if predictive_all.ndim == 3: average_predictive = predictive_all.mean(axis=0)
    elif predictive_all.ndim == 4:
        lag_discount = 1.0 / np.arange(1, predictive_all.shape[1] + 1, dtype=float)
        average_predictive = np.einsum("tlij,l->ij", predictive_all, lag_discount) / (predictive_all.shape[0] * lag_discount.sum())
    else: raise ValueError("Unexpected predictive adjacency path shape.")
    
    pd.DataFrame(average_risk, index=config.tickers, columns=config.tickers).to_csv(graphs / "average_risk_adjacency.csv")
    pd.DataFrame(average_predictive, index=config.tickers, columns=config.tickers).to_csv(graphs / "average_predictive_adjacency.csv")
    _plot_network(average_risk, config.tickers, figures / "average_risk_graph.png", "Average out-of-sample risk graph", directed=False)
    _plot_network(average_predictive, config.tickers, figures / "average_predictive_graph.png", "Average out-of-sample directed predictive graph", directed=True)

    plot_descriptive_analysis(data, config, figures)
    plot_model_results(
        dates, averaged, performance, model["hmm_probabilities"], data.phases, sim.METHOD_ORDER, config.tickers,
        DEFAULT_EVENT_WINDOWS, figures, primary_xgat_method=_primary_xgat_method(config.temporal_modes), temporal_methods=tuple(TEMPORAL_MODE_LABELS[mode] for mode in config.temporal_modes),
    )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "configuration": asdict(config), "methods": list(sim.METHOD_ORDER),
        "xgat_temporal_modes": list(config.temporal_modes), "xgat_temporal_method_labels": {mode: TEMPORAL_MODE_LABELS[mode] for mode in config.temporal_modes},
        "successful_policy_seeds": model["successful_seeds"], "sample_start": data.log_returns.index.min(), "sample_end": data.log_returns.index.max(),
        "out_of_sample_start": dates.min(), "out_of_sample_end": dates.max(), "data_sources": {"market": "Yahoo Finance chart endpoint", "cash_and_macro": "FRED CSV endpoint"},
        "code_hashes": {name: _sha256(path) for name, path in code_paths.items()}, "interface_audit": model.get("interface_audit", {}),
    }
    _write_json(root / "run_metadata.json", metadata)
    _write_manifests(root)

# -----------------------------------------------------------------------------
# 11. Command-line interface
# -----------------------------------------------------------------------------

def _parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())

def _parse_temporal_modes(value: str) -> tuple[str, ...]:
    return tuple(item.strip().lower() for item in value.split(",") if item.strip())

def _parse_tickers(value: str) -> tuple[AssetSpec, ...]:
    tickers = tuple(item.strip().upper() for item in value.split(",") if item.strip())
    known = {asset.ticker: asset for asset in DEFAULT_ASSETS}
    return tuple(known.get(ticker, AssetSpec(ticker, ticker, "Other")) for ticker in tickers)

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the X-GAT-DRL real-data application.")
    parser.add_argument("--core-path", default="xgat_drl_code.py")
    parser.add_argument("--simulations-path", default="simulations.py")
    parser.add_argument("--output-dir", default="real_data_results")
    parser.add_argument("--cache-dir", default="real_data_cache")
    parser.add_argument("--start-date", default="2008-01-02")
    parser.add_argument("--end-date", default="2026-08-04")
    parser.add_argument("--oos-start", default="2019-01-02")
    parser.add_argument("--evaluation-mode", choices=("fixed_holdout", "annual_refit"), default="fixed_holdout")
    parser.add_argument("--annual-training-years", type=int, default=8)
    parser.add_argument("--minimum-annual-test-observations", type=int, default=63)
    parser.add_argument("--tickers", default=",".join(asset.ticker for asset in DEFAULT_ASSETS))
    parser.add_argument("--benchmark-ticker", default="SPY")
    parser.add_argument("--policy-seeds", default="101,211,307,401,503")
    parser.add_argument("--temporal-modes", default="hybrid,gru,lstm")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--refresh-data", action="store_true")
    parser.add_argument("--prices-csv", default=None)
    parser.add_argument("--volumes-csv", default=None)
    parser.add_argument("--macro-csv", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    _configure_torch_threads()

    config = RealDataConfig(
        start_date=args.start_date, end_date=args.end_date, out_of_sample_start=args.oos_start, evaluation_mode=args.evaluation_mode,
        annual_training_years=int(args.annual_training_years), minimum_annual_test_observations=int(args.minimum_annual_test_observations),
        benchmark_ticker=args.benchmark_ticker, risk_assets=_parse_tickers(args.tickers), policy_seeds=_parse_ints(args.policy_seeds),
        temporal_modes=_parse_temporal_modes(args.temporal_modes), output_dir=Path(args.output_dir), cache_dir=Path(args.cache_dir),
        refresh_data=bool(args.refresh_data), local_prices_csv=Path(args.prices_csv) if args.prices_csv else None, local_volumes_csv=Path(args.volumes_csv) if args.volumes_csv else None,
        local_macro_csv=Path(args.macro_csv) if args.macro_csv else None, quick=bool(args.quick), continue_on_error=False,
    )
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)

    core_path = Path(args.core_path)
    simulations_path = Path(args.simulations_path)
    core = _load_module(core_path, "xgat_drl_code")
    sim = _load_module(simulations_path, "xgat_simulations")
    compatibility_adapters = _install_core_compatibility_adapters(core)
    interface_audit = _validate_runtime_interfaces(core, sim)
    method_order = _configure_real_data_method_order(sim, config.temporal_modes)
    interface_audit["compatibility_adapters"] = compatibility_adapters
    interface_audit["real_data_method_order"] = list(method_order)
    LOGGER.info("Validated X-GAT-DRL interfaces and %d real-data methods (%s).", len(method_order), ", ".join(config.temporal_modes))

    LOGGER.info("Acquiring market and macro-financial data")
    prices, volumes, macro, source_manifest = acquire_data(config)
    data = align_and_transform_data(prices, volumes, macro, config)
    LOGGER.info("Aligned sample: %s to %s (%d daily returns)", data.log_returns.index.min().date(), data.log_returns.index.max().date(), len(data.log_returns))

    descriptive, tests = descriptive_statistics(core, data, config)
    preliminary_root = config.output_dir / "pre_model"
    preliminary_root.mkdir(parents=True, exist_ok=True)
    descriptive.to_csv(preliminary_root / "descriptive_statistics.csv", index=False)
    tests.to_csv(preliminary_root / "distribution_and_stationarity_tests.csv", index=False)
    data.data_quality.to_csv(preliminary_root / "data_quality_report.csv", index=False)
    data.phases.to_csv(preliminary_root / "market_phase_daily.csv")
    plot_descriptive_analysis(data, config, config.output_dir / "figures" / "descriptive")

    if args.analysis_only:
        LOGGER.info("Analysis-only mode completed; model training was skipped.")
        return

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu" if args.device == "cpu" else "cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    model = run_model_application(core, sim, data, config, device, config.output_dir)
    model["interface_audit"] = interface_audit
    save_results(core, sim, data, descriptive, tests, source_manifest, model, config, {"real_data_application.py": Path(__file__), "simulations.py": simulations_path, "xgat_drl_code.py": core_path})
    LOGGER.info("Completed. Results were written to %s", config.output_dir.resolve())

if __name__ == "__main__":
    main()
