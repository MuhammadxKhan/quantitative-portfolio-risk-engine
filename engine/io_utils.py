"""Loading config + price data.

I put config parsing and data loading in the same file because they're both
just "get the inputs ready before any of the actual maths happens".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---- config ---------------------------------------------------------------

@dataclass
class Config:
    output_dir: Path
    random_seed: int
    save_html: bool
    save_csv: bool
    auto_open_html: bool
    tickers: list[str]
    start_date: str
    end_date: str
    source: str
    price_field: str
    trading_days: int
    mean_shrinkage: float
    covariance_estimator: str
    ewma_decay: float
    horizon_years: float
    n_steps: int
    n_scenarios: int
    t_dof: int
    confidence_level: float
    risk_free_rate: float
    min_weight: float
    max_weight: float
    n_random_portfolios: int
    frontier_points: int
    backtest_enabled: bool
    lookback_days: int
    rebalance_days: int
    strategy: str
    transaction_cost_bps: float


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    p, d, e = raw["project"], raw["data"], raw["estimation"]
    s, o, b = raw["scenarios"], raw["optimisation"], raw["backtest"]

    return Config(
        output_dir=Path(p["output_dir"]),
        random_seed=int(p["random_seed"]),
        save_html=bool(p["save_html"]),
        save_csv=bool(p["save_csv"]),
        auto_open_html=bool(p["auto_open_html"]),
        tickers=list(d["tickers"]),
        start_date=str(d["start_date"]),
        end_date=str(d["end_date"]),
        source=str(d["source"]),
        price_field=str(d["price_field"]),
        trading_days=int(e["trading_days"]),
        mean_shrinkage=float(e["mean_shrinkage"]),
        covariance_estimator=str(e["covariance_estimator"]),
        ewma_decay=float(e.get("ewma_decay", 0.94)),
        horizon_years=float(s["horizon_years"]),
        n_steps=int(s["n_steps"]),
        n_scenarios=int(s["n_scenarios"]),
        t_dof=int(s["t_degrees_of_freedom"]),
        confidence_level=float(s["confidence_level"]),
        risk_free_rate=float(o["risk_free_rate"]),
        min_weight=float(o["min_weight"]),
        max_weight=float(o["max_weight"]),
        n_random_portfolios=int(o["n_random_portfolios"]),
        frontier_points=int(o["frontier_points"]),
        backtest_enabled=bool(b["enabled"]),
        lookback_days=int(b["lookback_days"]),
        rebalance_days=int(b["rebalance_frequency_days"]),
        strategy=str(b["strategy"]),
        transaction_cost_bps=float(b["transaction_cost_bps"]),
    )


# ---- price data -----------------------------------------------------------

@dataclass
class PriceData:
    prices: pd.DataFrame
    log_returns: pd.DataFrame
    simple_returns: pd.DataFrame


def load_prices(cfg: Config) -> PriceData:
    if cfg.source == "sample":
        prices = _sample_prices(cfg.tickers, cfg.start_date, cfg.end_date, cfg.random_seed)
    elif cfg.source == "yfinance":
        prices = _yfinance_prices(cfg.tickers, cfg.start_date, cfg.end_date, cfg.price_field)
    else:
        raise ValueError(f"unknown data source '{cfg.source}' (use 'sample' or 'yfinance')")

    prices = prices.sort_index().dropna(how="any")
    if prices.empty:
        raise ValueError("no price data left after dropping NaNs")

    log_returns = np.log(prices / prices.shift(1)).dropna()
    simple_returns = prices.pct_change().dropna()
    return PriceData(prices, log_returns, simple_returns)


def _yfinance_prices(tickers, start, end, field):
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance not installed - use source: sample instead") from exc

    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError("yfinance returned nothing - check tickers/dates")

    if isinstance(raw.columns, pd.MultiIndex):  # happens with multiple tickers
        prices = raw[field]
    else:
        prices = raw[[field]]
        prices.columns = tickers
    return prices[tickers].dropna(how="any")


def _sample_prices(tickers, start, end, seed):
    # Synthetic correlated prices so the repo runs offline with no API key.
    # It's basically a correlated GBM - same idea as the scenario engine.
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end)
    n_days, n_assets = len(dates), len(tickers)

    annual_mu = np.linspace(0.07, 0.13, n_assets)
    annual_vol = np.linspace(0.18, 0.30, n_assets)

    corr = np.full((n_assets, n_assets), 0.45)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(annual_vol, annual_vol) * corr / 252
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(n_assets))

    drift = (annual_mu - 0.5 * annual_vol ** 2) / 252
    shocks = rng.normal(size=(n_days, n_assets)) @ chol.T
    log_r = drift + shocks
    prices = 100 * np.exp(np.cumsum(log_r, axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)
