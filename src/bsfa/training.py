import json, os
from typing import List, Dict
from .data import BalanceSheetDataProcessor, FEATURE_NAMES_25
from .forecaster import BalanceSheetForecaster

ART_DIR = "artifacts"
os.makedirs(ART_DIR, exist_ok=True)

def train_pipeline(tickers: List[str], lookback: int = 2, epochs: int = 80,
                   val_ratio: float = 0.2, batch_size: int = 16, patience: int = 10, smape_alpha: float = 0.4):
    proc = BalanceSheetDataProcessor()
    feature_names = FEATURE_NAMES_25
    all_data: Dict[str, "pd.DataFrame"] = {}

    print("=== THU THẬP DỮ LIỆU ===")
    for t in tickers:
        print(f"\nĐang xử lý {t}...")
        df = proc.get_balance_sheet_data(t)
        if not df.empty and len(df) > lookback:
            valid = proc.validate_data_quality(df)
            df = df.reindex(columns=feature_names, fill_value=0)
            all_data[t] = df
            print(f"✓ Dữ liệu: {valid.get('data_points', len(df))} periods")
            print(f"✓ Time span: {valid.get('time_span', 'N/A')}")
            if 'max_identity_error_pct' in valid:
                print(f"✓ Lỗi identity tối đa: {valid['max_identity_error_pct']:.2f}%")

    if not all_data:
        raise SystemExit("Không có dữ liệu nào để training!")

    print("\n=== HUẤN LUYỆN MÔ HÌNH ===")
    print(f"Số công ty: {len(all_data)} | Lookback: {lookback} | Epochs: {epochs}")
    fc = BalanceSheetForecaster(lookback_periods=lookback)
    history = fc.train(
        data_dict=all_data,
        feature_names=feature_names,
        epochs=epochs,
        val_ratio=val_ratio,
        batch_size=batch_size,
        patience=patience,
        smape_alpha=smape_alpha
    )

    # Save artifacts (history only – scalers/weights are kept in-memory for simplicity)
    hist_path = os.path.join(ART_DIR, "training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu history: {hist_path}")
    return {"history_path": hist_path}
