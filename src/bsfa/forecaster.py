from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models, layers
import tensorflow as tf
from .layers import AccountingConstraintInOriginalSpace
from .metrics import make_weighted_huber_plus_core_smape_loss, make_masked_mape_metric_original

class BalanceSheetForecaster:
    def __init__(self, lookback_periods: int):
        self.lookback_periods = lookback_periods
        self.feature_names = None
        self.feature_dim = None
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.model = None
        self.is_trained = False
        self.y_denom_floor = None

    def build_model(self, y_mu, y_sigma, loss_fn=None, metrics_list=None):
        assert self.feature_dim is not None
        inputs = layers.Input(shape=(self.lookback_periods, self.feature_dim))
        x = layers.LSTM(64, return_sequences=True)(inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(32, return_sequences=False)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
        raw_out = layers.Dense(self.feature_dim)(x)
        outputs = AccountingConstraintInOriginalSpace(y_mu=y_mu, y_sigma=y_sigma)(raw_out)

        model = models.Model(inputs, outputs)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
        if loss_fn is None:
            loss_fn = tf.keras.losses.Huber(delta=1.0)
        if metrics_list is None:
            metrics_list = ["mae"]
        model.compile(optimizer=opt, loss=loss_fn, metrics=metrics_list)
        self.model = model
        return model

    @staticmethod
    def _prepare_sequences_from_values(values: np.ndarray, lookback: int):
        seqs, tgts = [], []
        for i in range(len(values) - lookback):
            seqs.append(values[i:i+lookback])
            tgts.append(values[i+lookback])
        return np.array(seqs), np.array(tgts)

    def _ensure_feature_order(self, df_or_arr):
        if isinstance(df_or_arr, pd.DataFrame):
            missing = [c for c in self.feature_names if c not in df_or_arr.columns]
            for c in missing:
                df_or_arr[c] = 0.0
            arr = df_or_arr[self.feature_names].values
        else:
            arr = np.asarray(df_or_arr)
            if arr.shape[1] != self.feature_dim:
                raise ValueError(f"Input has {arr.shape[1]} features, expected {self.feature_dim}.")
        return arr

    def _prepare_train_val(self, data_dict, val_ratio=0.2):
        X_tr, y_tr, X_val, y_val = [], [], [], []
        for ticker, df in data_dict.items():
            if len(df) <= self.lookback_periods:
                continue
            values = self._ensure_feature_order(df)
            seq, tgt = self._prepare_sequences_from_values(values, self.lookback_periods)
            n = len(seq)
            if n <= 1:
                X_tr.append(seq); y_tr.append(tgt)
                continue
            n_val = max(1, int(round(n * val_ratio)))
            if n_val >= n: n_val = 1
            X_tr.append(seq[:-n_val]); y_tr.append(tgt[:-n_val])
            X_val.append(seq[-n_val:]); y_val.append(tgt[-n_val:])
        X_tr = np.vstack(X_tr) if X_tr else np.empty((0, self.lookback_periods, self.feature_dim))
        y_tr = np.vstack(y_tr) if y_tr else np.empty((0, self.feature_dim))
        X_val = np.vstack(X_val) if X_val else np.empty((0, self.lookback_periods, self.feature_dim))
        y_val = np.vstack(y_val) if y_val else np.empty((0, self.feature_dim))
        return X_tr, y_tr, X_val, y_val

    def _make_feature_weights(self):
        w = np.full(self.feature_dim, 0.25, dtype=np.float32)
        name_to_idx = {n:i for i,n in enumerate(self.feature_names)}
        for n in ['total_assets', 'total_liabilities', 'total_equity']:
            w[name_to_idx[n]] = 1.0
        for n in ['cash','receivables','inventory','other_current_assets','net_ppe',
                  'other_non_current_assets','accounts_payable','current_debt',
                  'other_current_liabilities','long_term_debt','other_non_current_liabilities',
                  'common_stock','retained_earnings','other_equity']:
            if n in name_to_idx: w[name_to_idx[n]] = max(w[name_to_idx[n]], 0.5)
        for n in ['total_current_assets','total_current_liabilities','calculated_total_assets',
                  'assets_identity_error','assets_identity_error_pct','working_capital',
                  'current_ratio','debt_to_equity']:
            if n in name_to_idx: w[name_to_idx[n]] = 0.1
        return w

    def train(self, data_dict, feature_names, epochs=80, val_ratio=0.2, batch_size=16, patience=10, smape_alpha=0.35):
        self.feature_names = list(feature_names)
        self.feature_dim = len(self.feature_names)
        X_tr, y_tr, X_val, y_val = self._prepare_train_val(data_dict, val_ratio=val_ratio)
        if X_tr.shape[0] == 0:
            total = sum(len(df) for df in data_dict.values())
            raise ValueError(f"Not enough training samples. Total points: {total}, lookback={self.lookback_periods}.")

        X_tr_flat = X_tr.reshape(-1, self.feature_dim)
        self.x_scaler.fit(X_tr_flat)
        self.y_scaler.fit(y_tr)

        abs_y = np.abs(y_tr)
        p5, p95 = np.percentile(abs_y, 5, axis=0), np.percentile(abs_y, 95, axis=0)
        denom_floor = np.maximum.reduce([p5, 0.01 * p95, np.full_like(p5, 1e-6)])
        self.y_denom_floor = denom_floor

        X_tr_s = self.x_scaler.transform(X_tr_flat).reshape(X_tr.shape)
        y_tr_s = self.y_scaler.transform(y_tr)
        if X_val.shape[0] > 0:
            X_val_s = self.x_scaler.transform(X_val.reshape(-1, self.feature_dim)).reshape(X_val.shape)
            y_val_s = self.y_scaler.transform(y_val)
        else:
            X_val_s, y_val_s = None, None

        y_mu = self.y_scaler.mean_.astype(np.float32)
        y_sigma = self.y_scaler.scale_.astype(np.float32)
        y_sigma = np.where(y_sigma == 0.0, 1.0, y_sigma)
        feat_weights = self._make_feature_weights()

        loss_fn = make_weighted_huber_plus_core_smape_loss(y_mu, y_sigma, feat_weights, core_idxs=(0,1,2),
                                                           alpha=smape_alpha, delta=1.0)
        mape_core = make_masked_mape_metric_original(y_mu, y_sigma, self.y_denom_floor, indices=[0,1,2], name="mape_core")

        self.build_model(y_mu=y_mu, y_sigma=y_sigma, loss_fn=loss_fn, metrics_list=["mae", mape_core])

        monitor_metric = "val_mape_core" if X_val_s is not None else "mape_core"
        es = tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, mode="min", patience=patience,
                                              restore_best_weights=True, verbose=1)
        rlrop = tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric, mode="min",
                                                     patience=max(2, patience//2), factor=0.5, verbose=1, min_lr=1e-5)

        if X_val_s is not None:
            history = self.model.fit(X_tr_s, y_tr_s, validation_data=(X_val_s, y_val_s),
                                     epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[es, rlrop])
        else:
            history = self.model.fit(X_tr_s, y_tr_s, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[es, rlrop])
        self.is_trained = True
        return history

    def forecast(self, historical_data_array, n_periods=1):
        if not self.is_trained:
            raise ValueError("Model is not trained")
        seq = self._ensure_feature_order(historical_data_array)
        if seq.shape[0] < self.lookback_periods:
            raise ValueError(f"Need >= {self.lookback_periods} historical points to forecast")
        forecasts = []
        cur = seq[-self.lookback_periods:].copy()
        for _ in range(n_periods):
            x = self.x_scaler.transform(cur).reshape(1, self.lookback_periods, self.feature_dim)
            pred_s = self.model.predict(x, verbose=0)
            pred = self.y_scaler.inverse_transform(pred_s)[0]
            forecasts.append(pred)
            cur = np.vstack([cur[1:], pred])
        return np.array(forecasts)

    def evaluate_forecast(self, actual, predicted, feature_names):
        res = {}
        eps = 1e-8
        idx_assets = feature_names.index("total_assets")
        idx_liab   = feature_names.index("total_liabilities")
        idx_equity = feature_names.index("total_equity")

        core_idx = [idx_assets, idx_liab, idx_equity]
        core_mae, core_mape = [], []
        for i in core_idx:
            mae = float(np.mean(np.abs(actual[:, i] - predicted[:, i])))
            floor = self.y_denom_floor[i] if self.y_denom_floor is not None else 1e-6
            denom = np.maximum(np.abs(actual[:, i]), floor) + eps
            mape = float(np.mean(np.abs(actual[:, i] - predicted[:, i]) / denom) * 100.0)
            core_mae.append(mae); core_mape.append(mape)
        res["CORE"] = {"MAE_avg": float(np.mean(core_mae)), "MAPE_avg_%": float(np.mean(core_mape))}

        identity_errors = np.abs(predicted[:, idx_assets] - (predicted[:, idx_liab] + predicted[:, idx_equity]))
        res["accounting_identity_errors"] = {"max_error": float(np.max(identity_errors)),
                                             "mean_error": float(np.mean(identity_errors))}
        return res
