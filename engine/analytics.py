"""Estimators + risk metrics.

Everything in here turns return data into numbers: expected returns,
covariance matrices, and the portfolio risk stats (vol, VaR, CVaR, Sharpe).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


# ---- estimating mu and covariance ----------------------------------------

def shrunk_mean(log_returns, trading_days=252, shrinkage=0.35):
    # Shrink each asset's sample mean towards the cross-sectional average.
    # Sample means are really noisy, so pulling them together helps a lot.
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must be in [0, 1]")
    sample_mu = log_returns.mean().to_numpy() * trading_days
    grand_mean = sample_mu.mean()
    return shrinkage * grand_mean + (1 - shrinkage) * sample_mu


def _ewma_cov(log_returns, decay):
    if not 0 < decay < 1:
        raise ValueError("ewma decay must be in (0, 1)")
    x = log_returns.to_numpy()
    x = x - x.mean(axis=0)
    n = x.shape[0]
    # most recent obs gets the biggest weight
    w = np.array([(1 - decay) * decay ** i for i in range(n - 1, -1, -1)])
    w /= w.sum()
    xw = x * np.sqrt(w[:, None])
    return xw.T @ xw


def covariance(log_returns, trading_days=252, method="ledoit_wolf", ewma_decay=0.94):
    if method == "ledoit_wolf":
        cov = LedoitWolf().fit(log_returns.to_numpy()).covariance_
    elif method == "ewma":
        cov = _ewma_cov(log_returns, ewma_decay)
    elif method == "sample":
        cov = np.cov(log_returns.to_numpy(), rowvar=False, ddof=1)
    else:
        raise ValueError("method must be ledoit_wolf, ewma or sample")
    # annualise, plus a tiny ridge so it's always positive definite
    return cov * trading_days + 1e-12 * np.eye(log_returns.shape[1])


def estimate_inputs(log_returns, trading_days, mean_shrinkage, cov_method, ewma_decay=0.94):
    mu = shrunk_mean(log_returns, trading_days, mean_shrinkage)
    cov = covariance(log_returns, trading_days, cov_method, ewma_decay)
    return mu, cov


# ---- portfolio risk metrics ----------------------------------------------

def portfolio_metrics(scenario_returns, weights, confidence=0.95, rf=0.04):
    """Risk/return stats for one or many weight vectors at once.

    scenario_returns : (n_scenarios, n_assets)
    weights          : (n_assets,) or (n_portfolios, n_assets)
    """
    weights = np.atleast_2d(weights)
    port = scenario_returns @ weights.T          # (n_scenarios, n_portfolios)

    exp_ret = port.mean(axis=0)
    vol = port.std(axis=0, ddof=1)

    # VaR = the loss quantile; CVaR = average loss in the tail beyond VaR
    losses = -port
    var = np.quantile(losses, confidence, axis=0)
    tail = losses >= var[None, :]
    counts = tail.sum(axis=0)
    cvar = np.divide((losses * tail).sum(axis=0), counts,
                     out=var.copy(), where=counts > 0)

    sharpe = np.divide(exp_ret - rf, vol,
                       out=np.full_like(exp_ret, np.nan), where=vol > 0)

    return {"expected_return": exp_ret, "volatility": vol,
            "var": var, "cvar": cvar, "sharpe": sharpe}


# ---- realised stats for the backtest -------------------------------------

def annualised_return(daily, trading_days=252):
    if daily.empty:
        return float("nan")
    growth = float((1 + daily).prod())
    return growth ** (trading_days / len(daily)) - 1


def annualised_vol(daily, trading_days=252):
    return float(daily.std(ddof=1) * np.sqrt(trading_days))


def sharpe_ratio(daily, rf=0.04, trading_days=252):
    vol = annualised_vol(daily, trading_days)
    if vol == 0 or np.isnan(vol):
        return float("nan")
    return (annualised_return(daily, trading_days) - rf) / vol


def max_drawdown(daily):
    wealth = (1 + daily).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def summarise(daily, rf, trading_days, confidence):
    losses = -daily.to_numpy()
    var = float(np.quantile(losses, confidence))
    tail = losses[losses >= var]
    cvar = float(tail.mean()) if len(tail) else var
    return {
        "annualised_return": annualised_return(daily, trading_days),
        "annualised_volatility": annualised_vol(daily, trading_days),
        "sharpe": sharpe_ratio(daily, rf, trading_days),
        "max_drawdown": max_drawdown(daily),
        "daily_var": var,
        "daily_cvar": cvar,
    }
