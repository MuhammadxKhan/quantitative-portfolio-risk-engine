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





# ---- out-of-sample VaR backtest ------------------------------------------

def kupiec_pof(n_obs, n_breaches, expected_rate):
    """Kupiec proportion-of-failures (POF) likelihood-ratio test.

    H0: the true breach probability equals `expected_rate` (= 1 - confidence).
    Returns (LR statistic, p-value) against a chi-square with 1 d.o.f.
    A high p-value means the VaR model's breach rate is statistically
    consistent with its nominal level (well-calibrated).
    """
    from scipy import stats

    N, x, p = int(n_obs), int(n_breaches), float(expected_rate)
    if N == 0:
        return float("nan"), float("nan")
    pi = x / N
    ll_null = (N - x) * np.log(1 - p) + x * np.log(p)
    if x == 0:
        ll_alt = N * np.log(1 - pi + 1e-12)
    elif x == N:
        ll_alt = N * np.log(pi + 1e-12)
    else:
        ll_alt = (N - x) * np.log(1 - pi) + x * np.log(pi)
    lr = -2.0 * (ll_null - ll_alt)
    pval = float(1 - stats.chi2.cdf(lr, df=1))
    return float(lr), pval


def var_backtest(portfolio_returns, window, confidence=0.95):
    """One-step-ahead historical-simulation VaR backtest (out-of-sample).

    At each date, the VaR forecast uses ONLY the trailing `window` returns
    (data strictly before the tested day); the realised next-day return is
    then checked against it. This is genuinely out-of-sample: no future
    information enters the forecast. Returns the breach counts and the
    Kupiec POF calibration test.
    """
    r = np.asarray(portfolio_returns, dtype=float)
    n = len(r)
    if n <= window + 1:
        raise ValueError("not enough data for this VaR backtest window")

    tail_prob = 1.0 - confidence
    breaches = np.empty(n - window, dtype=bool)
    for t in range(window, n):
        past = r[t - window:t]                      # strictly pre-t data
        var = -np.quantile(past, tail_prob)         # positive loss threshold
        breaches[t - window] = r[t] < -var          # realised loss beyond VaR

    n_obs = int(breaches.size)
    n_breach = int(breaches.sum())
    lr, pval = kupiec_pof(n_obs, n_breach, tail_prob)
    return {
        "confidence": confidence,
        "window": window,
        "observations": n_obs,
        "breaches": n_breach,
        "expected_rate": tail_prob,
        "realised_rate": n_breach / n_obs if n_obs else float("nan"),
        "kupiec_lr": lr,
        "kupiec_p": pval,
    }
