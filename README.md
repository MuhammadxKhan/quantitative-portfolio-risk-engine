# Quantitative Portfolio Risk Engine

A Python project for testing portfolio risk and allocation methods under different scenario assumptions.

I built this to understand how portfolio risk estimates change when moving from simple Gaussian assumptions to heavier-tailed or empirical return distributions. The default universe is deliberately small, so the model is easier to inspect and debug, but the structure is intended to be extended toward yield-bearing portfolios.

## What it currently does

* Loads market prices through `yfinance`
* Converts prices into log returns and simple returns
* Estimates expected returns using shrinkage toward the cross-sectional mean
* Estimates covariance using:
  * sample covariance
  * EWMA covariance
  * Ledoit-Wolf shrinkage
* Simulates one-year terminal returns using:
  * correlated geometric Brownian motion
  * multivariate-t shocks
  * historical bootstrap
* Computes portfolio-level:
  * expected return
  * volatility
  * VaR
  * CVaR
  * Sharpe ratio
* Solves long-only portfolio allocations using:
  * minimum variance
  * maximum Sharpe
  * an efficient frontier solved with SLSQP
* Runs a walk-forward backtest with rolling estimation windows and turnover costs

## Why I built it

The main question I wanted to test was:

> How sensitive are portfolio risk and allocation results to the scenario model and covariance estimator used?

The Gaussian model is useful as a baseline, but it can understate tail risk. The fat-tailed and bootstrap models make the tail-risk assumptions more visible.

## Running

```bash
pip install -r requirements.txt
python main.py
```

The default config uses synthetic data, so the project runs offline. To use live market data, change this in `config/default.yml`:

```yaml
data:
  source: "yfinance"
```

Outputs are written to the `outputs/` folder, which is ignored by Git.

## Project layout

```text
main.py                 entry point
config/default.yml      model parameters
engine/io_utils.py      config parsing and price loading
engine/analytics.py     estimators and risk metrics
engine/models.py        scenario generation and optimisation
engine/backtest.py      walk-forward backtest
engine/pipeline.py      pipeline and plotting
tests/test_engine.py    core tests
```

## Current limitations

This is a research/learning project, not a production portfolio system.

The main limitations are:

* the default universe is only three liquid equities
* expected returns are noisy and heavily assumption-driven
* transaction costs are simplified
* no liquidity, borrow, funding, or execution model is included
* no explicit factor model is used
* no DeFi-specific risks are modelled yet, such as smart contract risk, oracle risk, liquidation risk, pool depth, or APY regime changes
* the optimiser is long-only with simple bounds

## Next steps

The extension I am most interested in is a DeFi yield-risk module. The idea would be to model lending/yield positions using APY history, utilisation, TVL/liquidity, drawdown behaviour, and protocol-specific risk flags, then compare allocations using CVaR and walk-forward performance rather than headline APY alone.

## Future additions

The next step is to make the engine more robust as the asset universe grows. In particular, I want to compare mean-variance / max-Sharpe optimisation against allocation methods that are less sensitive to noisy expected return and covariance estimates, such as Hierarchical Risk Parity.
* CVaR-aware allocation across yield venues
* stress scenarios for liquidity shocks and APY compression
* comparison between mean-variance allocation and more robust methods such as HRP
 
