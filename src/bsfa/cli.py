# src/bsfa/cli.py
import json, os
from typing import List
import typer
import numpy as np
from .data import BalanceSheetDataProcessor, FEATURE_NAMES_25
from .forecaster import BalanceSheetForecaster
from .training import train_pipeline

app = typer.Typer(add_completion=False)

@app.command()
def train(
    tickers: List[str] = typer.Option(
        ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        "--tickers", "-t",
        help="Danh sách tickers. Dùng nhiều lần: -t AAPL -t MSFT ...",
        show_default=True,
    ),
    lookback: int = typer.Option(2, "--lookback", help="Số bước lịch sử cho LSTM"),
    epochs: int = typer.Option(80, "--epochs", help="Số epoch huấn luyện"),
    val_ratio: float = typer.Option(0.2, "--val-ratio", help="Tỷ lệ validation"),
    batch_size: int = typer.Option(16, "--batch-size", help="Batch size"),
    patience: int = typer.Option(10, "--patience", help="EarlyStopping patience"),
    smape_alpha: float = typer.Option(0.4, "--smape-alpha", help="Hệ số α cho sMAPE core"),
):
    """Train the model with original-space constraints and robust loss."""
    train_pipeline(tickers, lookback, epochs, val_ratio, batch_size, patience, smape_alpha)

@app.command()
def forecast(
    ticker: str = typer.Option("AAPL", "--ticker"),
    n_periods: int = typer.Option(1, "--n-periods"),
    lookback: int = typer.Option(2, "--lookback"),
):
    """Quick forecast using freshly-fetched data and a just-in-time trained model on that ticker only."""
    proc = BalanceSheetDataProcessor()
    feature_names = FEATURE_NAMES_25
    df = proc.get_balance_sheet_data(ticker)
    if df.empty or len(df) <= lookback:
        typer.echo(f"Dữ liệu quá ít cho {ticker}")
        raise typer.Exit(code=1)

    fc = BalanceSheetForecaster(lookback_periods=lookback)
    # train nhanh trên 1 ticker để khởi tạo scaler/model
    fc.train({ticker: df}, feature_names=feature_names, epochs=5, val_ratio=0.2, batch_size=8, patience=3, smape_alpha=0.4)

    hist_window = df[feature_names].values[-(lookback+1):-1]
    pred = fc.forecast(hist_window, n_periods=n_periods)
    typer.echo(json.dumps({"ticker": ticker, "forecast": pred.tolist(), "features": feature_names}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    app()
