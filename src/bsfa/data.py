import warnings
from typing import Dict, List
import yfinance as yf
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

CORE_FIELDS = {
    # Assets
    "Cash And Cash Equivalents": "cash",
    "Accounts Receivable": "receivables",
    "Inventory": "inventory",
    "Other Current Assets": "other_current_assets",
    "Net PPE": "net_ppe",
    "Other Non Current Assets": "other_non_current_assets",
    "Total Assets": "total_assets",
    # Liabilities
    "Accounts Payable": "accounts_payable",
    "Current Debt": "current_debt",
    "Other Current Liabilities": "other_current_liabilities",
    "Long Term Debt": "long_term_debt",
    "Other Non Current Liabilities": "other_non_current_liabilities",
    "Total Liabilities Net Minority Interest": "total_liabilities",
    # Equity
    "Common Stock": "common_stock",
    "Retained Earnings": "retained_earnings",
    "Other Equity Adjustments": "other_equity",
    "Total Equity Gross Minority Interest": "total_equity",
}

DERIVED_FIELDS = [
    "total_current_assets",
    "total_current_liabilities",
    "calculated_total_assets",
    "assets_identity_error",
    "assets_identity_error_pct",
    "working_capital",
    "current_ratio",
    "debt_to_equity",
]

def build_feature_names():
    core_names = list(CORE_FIELDS.values())
    primary3 = ["total_assets","total_liabilities","total_equity"]
    rest_core = [c for c in core_names if c not in primary3]
    return primary3 + rest_core + DERIVED_FIELDS  # 3 + 14 + 8 = 25

FEATURE_NAMES_25 = build_feature_names()

class BalanceSheetDataProcessor:
    def __init__(self):
        self.core_fields = CORE_FIELDS
        self.derived_fields = DERIVED_FIELDS
        self.feature_names_25 = FEATURE_NAMES_25

    def get_feature_names(self):
        return self.feature_names_25

    def get_balance_sheet_data(self, ticker: str, period: str = "quarterly") -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            balance_sheet = stock.balance_sheet
            if balance_sheet is None or balance_sheet.empty:
                raise ValueError(f"Balance sheet data not found for {ticker}")

            filtered = {}
            for yahoo_field, our_field in self.core_fields.items():
                if yahoo_field in balance_sheet.index:
                    filtered[our_field] = balance_sheet.loc[yahoo_field]
                else:
                    print(f"Warning: Missing field {yahoo_field} for {ticker}")
                    filtered[our_field] = pd.Series([np.nan] * len(balance_sheet.columns), index=balance_sheet.columns)

            df = pd.DataFrame(filtered).sort_index(ascending=True)
            df = self.calculate_derived_fields(df)
            df = self.clean_data(df)
            df = df.reindex(columns=self.feature_names_25, fill_value=0)
            return df
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def calculate_derived_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df["total_current_assets"] = (
            df["cash"].fillna(0) + df["receivables"].fillna(0) + df["inventory"].fillna(0) + df["other_current_assets"].fillna(0)
        )
        df["total_current_liabilities"] = (
            df["accounts_payable"].fillna(0) + df["current_debt"].fillna(0) + df["other_current_liabilities"].fillna(0)
        )
        df["calculated_total_assets"] = df["total_liabilities"].fillna(0) + df["total_equity"].fillna(0)
        df["assets_identity_error"] = (df["total_assets"] - df["calculated_total_assets"]).abs()
        df["assets_identity_error_pct"] = (df["assets_identity_error"] / df["total_assets"].replace(0, np.nan)) * 100
        df["working_capital"] = df["total_current_assets"] - df["total_current_liabilities"]
        df["current_ratio"] = df["total_current_assets"] / df["total_current_liabilities"].replace(0, np.nan)
        df["debt_to_equity"] = df["total_liabilities"] / df["total_equity"].replace(0, np.nan)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        threshold = int(len(df.columns) * 0.3)
        df = df.dropna(thresh=threshold)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(method="ffill")
        df[numeric_cols] = df[numeric_cols].fillna(method="bfill")
        df[numeric_cols] = df[numeric_cols].fillna(0)
        for col in ["current_ratio","debt_to_equity","assets_identity_error_pct"]:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0)
        return df

    def validate_data_quality(self, df: pd.DataFrame):
        if df.empty:
            return {"error": "DataFrame is empty"}
        res = {}
        missing_pct = df.isnull().mean() * 100
        res["missing_values"] = missing_pct[missing_pct > 0]
        if "assets_identity_error_pct" in df.columns:
            mx = df["assets_identity_error_pct"].max()
            res["max_identity_error_pct"] = float(mx if pd.notna(mx) else 0.0)
        if len(df) > 1 and "total_assets" in df.columns:
            asset_growth = df["total_assets"].pct_change().dropna()
            if len(asset_growth):
                res["asset_growth_stats"] = {
                    "mean": float(asset_growth.mean()),
                    "std": float(asset_growth.std()),
                    "min": float(asset_growth.min()),
                    "max": float(asset_growth.max()),
                }
        res["data_points"] = int(len(df))
        res["time_span"] = f"{df.index[0]} to {df.index[-1]}" if len(df) else "N/A"
        return res
