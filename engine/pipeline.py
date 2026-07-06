"""Wires everything together and writes the outputs.

I kept the plotting helpers in this file since they're only ever called from
the pipeline anyway - no point having them in their own module.
"""
from __future__ import annotations

import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from engine.analytics import covariance, estimate_inputs, portfolio_metrics, var_backtest
from engine.backtest import run_backtest
from engine.io_utils import Config, load_prices
from engine.models import (
    Settings, correlated_gbm, efficient_frontier, historical_bootstrap,
    max_sharpe, min_variance, multivariate_t, random_cloud,
)


# ---- plotting helpers ----------------------------------------------------

def _save(fig, path: Path, auto_open: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")
    if auto_open:
        webbrowser.open(path.resolve().as_uri())


def _style(fig, title, xlab, ylab):
    fig.update_layout(title={"text": title, "x": 0.5}, template="plotly_white",
                      width=1400, height=800, font={"size": 15},
                      xaxis_title=xlab, yaxis_title=ylab,
                      legend={"orientation": "h", "y": 1.03})


def plot_gbm(paths, tickers, max_paths=20):
    fig = go.Figure()
    shown = min(max_paths, paths.shape[0])
    for a, t in enumerate(tickers):
        for p in range(shown):
            fig.add_trace(go.Scatter(x=np.arange(paths.shape[1]), y=paths[p, :, a],
                                     mode="lines", opacity=0.15, line={"width": 1},
                                     name=t if p == 0 else None, showlegend=p == 0,
                                     hoverinfo="skip"))
    _style(fig, "Correlated GBM sample paths", "Time step", "Simulated price")
    return fig


def plot_frontier(cloud, frontier, color_col, minvar, maxsharpe, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cloud["volatility"], y=cloud["expected_return"],
                             mode="markers",
                             marker={"size": 4.5, "color": cloud[color_col],
                                     "colorscale": "Viridis", "opacity": 0.8,
                                     "showscale": True, "colorbar": {"title": color_col}},
                             name="random portfolios", showlegend=False))
    fig.add_trace(go.Scatter(x=frontier["volatility"], y=frontier["expected_return"],
                             mode="lines", line={"width": 3}, name="efficient frontier"))
    fig.add_trace(go.Scatter(x=[minvar["volatility"]], y=[minvar["expected_return"]],
                             mode="markers", marker={"size": 13, "symbol": "x"},
                             name="min variance"))
    fig.add_trace(go.Scatter(x=[maxsharpe["volatility"]], y=[maxsharpe["expected_return"]],
                             mode="markers", marker={"size": 15, "symbol": "star"},
                             name="max Sharpe"))
    _style(fig, title, "Volatility", "Expected return")
    return fig


def plot_equity(returns):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns.index, y=returns["strategy_equity"],
                             mode="lines", name="optimised"))
    fig.add_trace(go.Scatter(x=returns.index, y=returns["equal_weight_equity"],
                             mode="lines", name="equal weight"))
    _style(fig, "Walk-forward equity curve", "Date", "Growth of 1 unit")
    return fig


# ---- the actual pipeline -------------------------------------------------

def _single_metrics(scenario_returns, w, confidence, rf):
    return {k: float(v[0]) for k, v in
            portfolio_metrics(scenario_returns, w, confidence, rf).items()}


def run_pipeline(cfg: Config) -> dict[str, Path]:
    rng = np.random.default_rng(cfg.random_seed)
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    data = load_prices(cfg)
    mu, cov = estimate_inputs(data.log_returns, cfg.trading_days,
                              cfg.mean_shrinkage, cfg.covariance_estimator, cfg.ewma_decay)
    s0 = data.prices.iloc[-1].to_numpy(dtype=float)

    settings = Settings(cfg.min_weight, cfg.max_weight, cfg.risk_free_rate)

    scenarios = [
        correlated_gbm(s0, mu, cov, cfg.horizon_years, cfg.n_steps, cfg.n_scenarios, rng),
        multivariate_t(mu, cov, cfg.horizon_years, cfg.n_scenarios, cfg.t_dof, rng),
        historical_bootstrap(data.log_returns, cfg.horizon_years, cfg.n_scenarios,
                             cfg.trading_days, rng),
    ]

    summary_rows = []
    for idx, sc in enumerate(scenarios, start=2):
        cloud = random_cloud(sc.returns, cfg.tickers, cfg.n_random_portfolios,
                             settings, cfg.confidence_level, rng)
        frontier, w_minvar, w_maxsharpe = efficient_frontier(
            sc.returns, settings, cfg.frontier_points, cfg.confidence_level)

        mv = _single_metrics(sc.returns, w_minvar, cfg.confidence_level, cfg.risk_free_rate)
        ms = _single_metrics(sc.returns, w_maxsharpe, cfg.confidence_level, cfg.risk_free_rate)

        if cfg.save_html:
            if idx == 2 and scenarios[0].paths is not None:
                p = out / "01_gbm_paths.html"
                _save(plot_gbm(scenarios[0].paths, cfg.tickers), p, cfg.auto_open_html)
                written["gbm_paths"] = p
            color = "expected_return" if sc.name == "gbm" else "cvar"
            p = out / f"{idx:02d}_{sc.name}_frontier.html"
            _save(plot_frontier(cloud, frontier, color, mv, ms,
                                f"{sc.name.replace('_', ' ').title()} frontier"),
                  p, cfg.auto_open_html)
            written[f"{sc.name}_frontier"] = p

        row = {"model": sc.name,
               "minvar_return": mv["expected_return"], "minvar_vol": mv["volatility"],
               "minvar_cvar": mv["cvar"],
               "maxsharpe_return": ms["expected_return"], "maxsharpe_vol": ms["volatility"],
               "maxsharpe_cvar": ms["cvar"], "maxsharpe_sharpe": ms["sharpe"]}
        for t, w in zip(cfg.tickers, w_maxsharpe):
            row[f"maxsharpe_{t}_weight"] = float(w)
        summary_rows.append(row)

    if cfg.save_csv:
        p = out / "scenario_summary.csv"
        pd.DataFrame(summary_rows).to_csv(p, index=False)
        written["scenario_summary"] = p

        # quick covariance estimator comparison, handy as a sanity check
        comp = pd.DataFrame({
            "ticker": cfg.tickers,
            "sample_vol": np.sqrt(np.diag(covariance(data.log_returns, cfg.trading_days, "sample"))),
            "ewma_vol": np.sqrt(np.diag(covariance(data.log_returns, cfg.trading_days, "ewma", cfg.ewma_decay))),
            "ledoit_wolf_vol": np.sqrt(np.diag(covariance(data.log_returns, cfg.trading_days, "ledoit_wolf"))),
        })
        p = out / "covariance_comparison.csv"
        comp.to_csv(p, index=False)
        written["covariance_comparison"] = p

    if cfg.backtest_enabled:
        bt = run_backtest(data.log_returns, data.simple_returns, cfg, settings)
        if cfg.save_csv:
            for name, frame in [("walk_forward_returns", bt.returns),
                                ("walk_forward_weights", bt.weights),
                                ("walk_forward_summary", bt.summary)]:
                p = out / f"{name}.csv"
                frame.to_csv(p, index=(name != "walk_forward_summary"))
                written[name] = p
        if cfg.save_html:
            p = out / "05_walk_forward_equity.html"
            _save(plot_equity(bt.returns), p, cfg.auto_open_html)
            written["walk_forward_equity"] = p

    # Out-of-sample VaR backtest on the equal-weight portfolio. This is
    # independent of the scenario engines (no circularity): the VaR forecast
    # at each date uses only trailing data, then is scored on the next day.
    ew_returns = data.simple_returns.mean(axis=1).to_numpy()
    vb = var_backtest(ew_returns, cfg.lookback_days, cfg.confidence_level)
    print(f"\nOut-of-sample VaR backtest (equal-weight, {cfg.confidence_level:.0%} "
          f"historical, {cfg.lookback_days}-day window):")
    print(f"  breaches {vb['breaches']}/{vb['observations']} = {vb['realised_rate']:.2%} "
          f"(expected {vb['expected_rate']:.2%})  |  Kupiec p = {vb['kupiec_p']:.3f}")
    if cfg.save_csv:
        p = out / "var_backtest.csv"
        pd.DataFrame([vb]).to_csv(p, index=False)
        written["var_backtest"] = p

    return written
