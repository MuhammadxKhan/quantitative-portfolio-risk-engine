"""Walk-forward backtest.

Roll through history: every `rebalance_days` days, estimate mu/cov on the
trailing `lookback_days` window, re-optimise, pay turnover costs, then hold.
Compare against an equal-weight benchmark.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.analytics import estimate_inputs, summarise
from engine.models import Settings, max_sharpe, min_variance


@dataclass
class BacktestResult:
    returns: pd.DataFrame
    weights: pd.DataFrame
    summary: pd.DataFrame


def run_backtest(log_returns, simple_returns, cfg, settings):
    if len(simple_returns) <= cfg.lookback_days + cfg.rebalance_days:
        raise ValueError("not enough data for this lookback/rebalance window")

    tickers = list(simple_returns.columns)
    n = len(tickers)
    cost_rate = cfg.transaction_cost_bps / 10_000

    weights = np.full(n, 1.0 / n)         # current strategy weights
    benchmark = np.full(n, 1.0 / n)       # equal-weight, never changes

    strat_rows, bench_rows, weight_rows = [], [], []
    rebalance_on = set(range(cfg.lookback_days, len(simple_returns), cfg.rebalance_days))

    for i in range(cfg.lookback_days, len(simple_returns)):
        date = simple_returns.index[i]
        todays = simple_returns.iloc[i].to_numpy()
        cost = 0.0

        if i in rebalance_on:
            window = log_returns.iloc[i - cfg.lookback_days:i]
            mu, cov = estimate_inputs(window, cfg.trading_days,
                                      cfg.mean_shrinkage, cfg.covariance_estimator,
                                      cfg.ewma_decay)
            if cfg.strategy == "max_sharpe":
                new_w = max_sharpe(mu, cov, settings)
            elif cfg.strategy == "min_variance":
                new_w = min_variance(mu, cov, settings)
            else:
                raise ValueError("strategy must be max_sharpe or min_variance")

            turnover = float(np.abs(new_w - weights).sum())
            cost = turnover * cost_rate
            weights = new_w

            row = {"date": date, "turnover": turnover}
            row.update({f"{t}_weight": float(w) for t, w in zip(tickers, weights)})
            weight_rows.append(row)

        strat_rows.append((date, float(weights @ todays - cost)))
        bench_rows.append((date, float(benchmark @ todays)))

    rdf = pd.DataFrame(strat_rows, columns=["date", "strategy_return"]).set_index("date")
    rdf["equal_weight_return"] = pd.Series(dict(bench_rows), dtype=float)
    rdf["strategy_equity"] = (1 + rdf["strategy_return"]).cumprod()
    rdf["equal_weight_equity"] = (1 + rdf["equal_weight_return"]).cumprod()

    wdf = pd.DataFrame(weight_rows).set_index("date") if weight_rows else pd.DataFrame()
    avg_turnover = float(wdf["turnover"].mean()) if not wdf.empty else 0.0

    strat_stats = summarise(rdf["strategy_return"], settings.risk_free_rate,
                            cfg.trading_days, cfg.confidence_level)
    bench_stats = summarise(rdf["equal_weight_return"], settings.risk_free_rate,
                            cfg.trading_days, cfg.confidence_level)

    summary = pd.DataFrame([
        {"portfolio": cfg.strategy, **strat_stats, "avg_turnover": avg_turnover},
        {"portfolio": "equal_weight", **bench_stats, "avg_turnover": 0.0},
    ])
    return BacktestResult(rdf, wdf, summary)
