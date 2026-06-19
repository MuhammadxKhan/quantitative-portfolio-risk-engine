# Quantitative Portfolio Risk Engine

A Python project for portfolio risk modelling, scenario simulation, efficient-frontier
optimisation, and walk-forward backtesting. It runs offline out of the box on synthetic
data, or on live market data via yfinance.

## What it does

- **Estimation** – shrunk expected returns and three covariance estimators (sample, EWMA, Ledoit-Wolf)
- **Scenario engines** – correlated GBM (Cholesky shocks + Ito drift), a fat-tailed multivariate-t, and a historical bootstrap
- **Risk metrics** – 95% VaR and CVaR, volatility, and Sharpe across thousands of sampled portfolios
- **Optimisation** – min-variance, max-Sharpe and a solved efficient frontier (SLSQP, long-only, weight bounds)
- **Backtest** – walk-forward test (126-day lookback, 21-day rebalance, 5bps turnover cost) vs an equal-weight benchmark


## Notes on method

- CVaR is the average loss beyond the VaR quantile, computed directly from the
  simulated loss distribution rather than a parametric formula.
- The fat-tailed-t scenario produces visibly larger VaR/CVaR than GBM, which is the
  whole reason for including it – Gaussian assumptions understate tail risk.
- Ledoit-Wolf shrinkage is the default covariance estimator since the sample covariance
  is unstable on short windows with few assets.
