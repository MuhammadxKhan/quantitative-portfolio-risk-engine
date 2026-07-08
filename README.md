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

> How sensitive are portfolio **risk** estimates to the scenario model and covariance estimator used?

The Gaussian model is useful as a baseline, but it can understate tail risk. The fat-tailed and bootstrap models make the tail-risk assumptions more visible.


## Running

```bash
pip install -r requirements.txt
python main.py
```

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

The main limitations are:

* the default universe is only three large, highly liquid US equities
  (AAPL, MSFT, GOOGL); so covariance shrinkage (Ledoit-Wolf) is close to
  decorative here and only starts to earn its place as the universe grows
* GBM/t scenario *returns* are circular by construction (see above); treat
  them as a risk-sensitivity layer, not a performance result
* expected returns are noisy and heavily assumption-driven, which is why the
  default allocation is min-variance rather than max-Sharpe
* transaction costs are simplified
* no liquidity, borrow, funding, or execution model is included
* no explicit factor model is used
* the optimiser is long-only with simple bounds


## Future additions

The next step is to make the engine more robust as the asset universe grows. In particular, I want to compare mean-variance / max-Sharpe optimisation against allocation methods that are less sensitive to noisy expected return and covariance estimates, such as Hierarchical Risk Parity.
* CVaR-aware allocation across yield venues
* stress scenarios for liquidity shocks and APY compression
* comparison between mean-variance allocation and more robust methods such as HRP
 
