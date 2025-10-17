import json, os
from typing import List, Dict
from .data import BalanceSheetDataProcessor, FEATURE_NAMES_25
from .forecaster import BalanceSheetForecaster

ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)


def train_pipeline(
    tickers: List[str],
    lookback: int = 2,
    epochs: int = 80,
    val_ratio: float = 0.2,
    batch_size: int = 16,
    patience: int = 10,
    smape_alpha: float = 0.4
):
    """
    Full training pipeline for the Balance Sheet Forecaster model.
    """
    processor = BalanceSheetDataProcessor()
    feature_names = FEATURE_NAMES_25
    all_data: Dict[str, "pd.DataFrame"] = {}

    print("=== DATA COLLECTION ===")
    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        df = processor.get_balance_sheet_data(ticker)
        if not df.empty and len(df) > lookback:
            validation = processor.validate_data_quality(df)
            df = df.reindex(columns=feature_names, fill_value=0)
            all_data[ticker] = df
            print(f"✓ Data points: {validation.get('data_points', len(df))} periods")
            print(f"✓ Time span: {validation.get('time_span', 'N/A')}")
            if 'max_identity_error_pct' in validation:
                print(f"✓ Max accounting identity error: {validation['max_identity_error_pct']:.2f}%")

    if not all_data:
        raise SystemExit("No valid data found for training!")

    print("\n=== MODEL TRAINING ===")
    print(f"Companies: {len(all_data)} | Lookback: {lookback} | Epochs: {epochs}")

    forecaster = BalanceSheetForecaster(lookback_periods=lookback)
    history = forecaster.train(
        data_dict=all_data,
        feature_names=feature_names,
        epochs=epochs,
        val_ratio=val_ratio,
        batch_size=batch_size,
        patience=patience,
        smape_alpha=smape_alpha
    )

    # Save training artifacts (only history for simplicity)
    hist_path = os.path.join(ART_DIR, "training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: [float(x) for x in v] for k, v in history.history.items()},
            f,
            ensure_ascii=False,
            indent=2
        )
    print(f"\nTraining history saved to: {hist_path}")
    return {"history_path": hist_path}
