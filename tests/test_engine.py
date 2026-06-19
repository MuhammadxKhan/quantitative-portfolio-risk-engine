"""Tests for the core engine pieces.

Run with:  pytest    (from the project root)
"""
import numpy as np
import pandas as pd
import pytest

from engine.analytics import portfolio_metrics, covariance, shrunk_mean
from engine.models import Settings, min_variance, max_sharpe, efficient_frontier


@pytest.fixture
def returns():
    # small reproducible return sample to test on
    rng = np.random.default_rng(0)
    return rng.normal(0.0005, 0.01, size=(500, 3))


@pytest.fixture
def settings():
    return Settings(min_weight=0.0, max_weight=0.8, risk_free_rate=0.04)


def test_weights_sum_to_one(returns, settings):
    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False)
    for solver in (min_variance, max_sharpe):
        w = solver(mu, cov, settings)
        assert np.isclose(w.sum(), 1.0)
        assert (w >= -1e-9).all() and (w <= 0.8 + 1e-9).all()


def test_min_variance_is_lowest_variance(returns, settings):
    # the min-variance portfolio should have <= variance than equal weight
    mu = returns.mean(axis=0)
    cov = np.cov(returns, rowvar=False)
    w = min_variance(mu, cov, settings)
    eq = np.full(3, 1 / 3)
    assert w @ cov @ w <= eq @ cov @ eq + 1e-9


def test_cvar_at_least_var(returns):
    # CVaR is an average of the worst losses, so it must be >= VaR
    w = np.full(3, 1 / 3)
    m = portfolio_metrics(returns, w)
    assert m["cvar"][0] >= m["var"][0] - 1e-9


def test_shrinkage_pulls_towards_mean(returns):
    df = pd.DataFrame(returns)
    full_shrink = shrunk_mean(df, shrinkage=1.0)
    # full shrinkage => every asset gets the same (grand mean) value
    assert np.allclose(full_shrink, full_shrink[0])


def test_covariance_is_symmetric(returns):
    cov = covariance(pd.DataFrame(returns), method="ledoit_wolf")
    assert np.allclose(cov, cov.T)


def test_frontier_returns_increase(returns, settings):
    df, _, _ = efficient_frontier(returns, settings, n_points=10, confidence=0.95)
    # sorted by volatility, so returns should be broadly non-decreasing
    assert len(df) > 1
    assert df["expected_return"].iloc[-1] >= df["expected_return"].iloc[0]
