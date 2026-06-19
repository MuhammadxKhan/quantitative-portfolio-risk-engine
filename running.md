## Running it

```bash
pip install -r requirements.txt
python main.py
```

It uses offline sample data by default, so it runs immediately with no API key.
To use live prices, set `data.source: yfinance` in `config/default.yml`.

Charts (Plotly HTML) and CSV summaries are written to `outputs/`. Set
`auto_open_html: true` in the config to have the charts pop open in your browser.

## Layout

```
main.py              entry point
config/default.yml   all parameters live here
engine/
  io_utils.py        config parsing + price data (sample or yfinance)
  analytics.py       estimators (mu, covariance) + risk metrics (VaR/CVaR/Sharpe)
  models.py          scenario engines + portfolio optimisation
  backtest.py        walk-forward backtest
  pipeline.py        ties it together and draws the charts
tests/test_engine.py
```

## Tests

```bash
pytest
```****
