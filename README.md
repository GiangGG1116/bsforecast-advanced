# BSFA — Balance Sheet Forecast (advanced)

This project reorganizes your single-file script into a clean package with:
- **Original-space accounting constraint** (`total_assets = total_liabilities + total_equity`)
- **Weighted Huber + core sMAPE** loss
- Robust **masked MAPE** metric on core features
- CLI via `bsfa`

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Train (default tickers)
bsfa train --epochs 80

# Train (custom tickers)
bsfa train --tickers AAPL MSFT GOOGL AMZN TSLA --epochs 80

# Forecast 1 period for AAPL using latest scaler & weights learned in-session
bsfa forecast --ticker AAPL --n-periods 1
```

Artifacts (history, weights, meta) are saved in `artifacts/`.
