"""Scenario engines + portfolio optimisation.

Two related jobs live here:
  1. simulate possible future returns (GBM, fat-tailed t, bootstrap)
  2. given simulated returns, solve for portfolios (min-var, max-Sharpe, frontier)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from engine.analytics import portfolio_metrics


# ---- scenario engines ----------------------------------------------------

@dataclass
class ScenarioSet:
    name: str
    returns: np.ndarray
    paths: np.ndarray | None = None


def correlated_gbm(s0, mu, cov, horizon_years, n_steps, n_scenarios, rng):
 # Standard multi-asset GBM in log space. Cholesky gives correlated normal
    # shocks. `mu` is already the mean of *log* returns (see shrunk_mean), i.e.
    # the log-drift (mu_arith - 0.5*sigma^2), so we must NOT subtract 0.5*sigma^2
    # again here -- doing so double-counts the Ito term and biases returns low.
    n_assets = len(s0)
    dt = horizon_years / n_steps
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(n_assets))

    paths = np.zeros((n_scenarios, n_steps + 1, n_assets))
    paths[:, 0, :] = s0
    drift = mu * dt

    for t in range(1, n_steps + 1):
        z = rng.normal(size=(n_scenarios, n_assets)) @ chol.T
        paths[:, t, :] = paths[:, t - 1, :] * np.exp(drift + z * np.sqrt(dt))

    terminal = paths[:, -1, :] / s0 - 1.0
    return ScenarioSet("gbm", terminal, paths)


def multivariate_t(mu, cov, horizon_years, n_scenarios, dof, rng):
    # Fat-tailed version: a normal shock divided by a chi-square factor.
   # smaller degrees of freedom create heavier-tailed shocks
    n_assets = len(mu)
    z = rng.normal(size=(n_scenarios, n_assets))
    chi = rng.chisquare(dof, size=n_scenarios)
    scale = np.sqrt(dof / chi).reshape(-1, 1)

    chol = np.linalg.cholesky(cov * horizon_years + 1e-12 * np.eye(n_assets))
    shocks = (z @ chol.T) * scale
    return ScenarioSet("fat_tailed_t", mu * horizon_years + shocks)


def historical_bootstrap(log_returns, horizon_years, n_scenarios, trading_days, rng):
    # Resample real daily log-returns with replacement and compound them.
    # No distributional assumption - it just reuses what actually happened.
    n_days = round(horizon_years * trading_days)
    data = log_returns.to_numpy()
    idx = rng.integers(0, data.shape[0], size=(n_scenarios, n_days))
    terminal = np.exp(data[idx].sum(axis=1)) - 1.0
    return ScenarioSet("historical_bootstrap", terminal)


# ---- optimisation --------------------------------------------------------

@dataclass
class Settings:
    min_weight: float
    max_weight: float
    risk_free_rate: float


def _check_bounds(n, lo, hi):
    if lo * n > 1 or hi * n < 1 or lo < 0 or hi > 1 or lo > hi:
        raise ValueError("weight bounds can't sum to 1 - check min/max_weight")


def _bounds(n, lo, hi):
    return [(lo, hi)] * n


def _sum_to_one():
    return {"type": "eq", "fun": lambda w: w.sum() - 1.0}


def _solve(objective, n, settings, extra_constraints=None):
    _check_bounds(n, settings.min_weight, settings.max_weight)
    cons = [_sum_to_one()] + (extra_constraints or [])
    x0 = np.full(n, 1.0 / n)
    res = minimize(objective, x0, method="SLSQP",
                   bounds=_bounds(n, settings.min_weight, settings.max_weight),
                   constraints=cons)
    if not res.success:
        raise RuntimeError(f"optimiser failed: {res.message}")
    return res.x


def min_variance(mu, cov, settings):
    return _solve(lambda w: w @ cov @ w, len(mu), settings)


def max_sharpe(mu, cov, settings):
    def neg_sharpe(w):
        r = w @ mu
        v = np.sqrt(max(w @ cov @ w, 1e-18))
        return -(r - settings.risk_free_rate) / v
    return _solve(neg_sharpe, len(mu), settings)


def max_return(mu, settings):
    return _solve(lambda w: -(w @ mu), len(mu), settings)


def _moments(scenario_returns):
    mu = scenario_returns.mean(axis=0)
    cov = np.cov(scenario_returns, rowvar=False, ddof=1)
    return mu, cov + 1e-12 * np.eye(cov.shape[0])


def random_cloud(scenario_returns, tickers, n_portfolios, settings, confidence, rng):
    # Random long-only weights via a Dirichlet, rejecting any that break bounds.
    # random feasible portfolios for comparison with the solved frontier
    n = scenario_returns.shape[1]
    _check_bounds(n, settings.min_weight, settings.max_weight)

    kept = []
    while sum(len(b) for b in kept) < n_portfolios:
        draw = rng.dirichlet(np.ones(n), size=max(n_portfolios * 3, 5000))
        ok = draw[(draw >= settings.min_weight).all(1) & (draw <= settings.max_weight).all(1)]
        if len(ok):
            kept.append(ok)
    weights = np.vstack(kept)[:n_portfolios]

    m = portfolio_metrics(scenario_returns, weights, confidence, settings.risk_free_rate)
    df = pd.DataFrame(m)
    for i, t in enumerate(tickers):
        df[f"{t}_weight"] = weights[:, i]
    return df


def efficient_frontier(scenario_returns, settings, n_points, confidence):
    # solve minimum-variance portfolios across a grid of target returns
    mu, cov = _moments(scenario_returns)
    w_minvar = min_variance(mu, cov, settings)
    w_maxsharpe = max_sharpe(mu, cov, settings)
    w_maxret = max_return(mu, settings)

    lo, hi = float(mu @ w_minvar), float(mu @ w_maxret)
    targets = np.linspace(lo, hi, n_points)

    solved = []
    for target in targets:
        # warm-start by blending the two endpoint portfolios
        blend = 0.0 if hi - lo < 1e-12 else (target - lo) / (hi - lo)
        x0 = np.clip((1 - blend) * w_minvar + blend * w_maxret,
                     settings.min_weight, settings.max_weight)
        x0 /= x0.sum()
        cons = [_sum_to_one(), {"type": "eq", "fun": lambda w, t=target: w @ mu - t}]
        res = minimize(lambda w: w @ cov @ w, x0, method="SLSQP",
                       bounds=_bounds(len(mu), settings.min_weight, settings.max_weight),
                       constraints=cons)
        if res.success:
            solved.append(res.x)

    if not solved:
        raise RuntimeError("no frontier points solved")

    weights = np.vstack(solved)
    m = portfolio_metrics(scenario_returns, weights, confidence, settings.risk_free_rate)
    df = pd.DataFrame(m)
    df["weights"] = list(weights)
    df = df.sort_values(["volatility", "expected_return"]).reset_index(drop=True)
    return df, w_minvar, w_maxsharpe
