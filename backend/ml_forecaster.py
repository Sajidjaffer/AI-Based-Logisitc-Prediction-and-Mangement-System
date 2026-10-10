"""
AI Demand Forecasting & Machine Learning Engine for Supply Chain Risk Management.
Implements a supervised Gradient Boosting Regressor (HistGradientBoostingRegressor)
trained on historical M5 retail sales with lag features, rolling statistics,
calendar/promotional indicators, SNAP subsidies, and product price features.
Optimized for high-speed vectorized batch multi-step inference.
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(CURRENT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODELS_DIR, "demand_forecast_model.joblib")
METADATA_PATH = os.path.join(CURRENT_DIR, "ml_metadata.json")

FEATURE_NAMES = [
    "lag_7",            # Sales exactly 7 days ago (weekly seasonality)
    "lag_14",           # Sales 14 days ago
    "lag_21",           # Sales 21 days ago
    "lag_28",           # Sales 28 days ago (monthly cycle)
    "rolling_mean_7",   # 7-day moving average demand
    "rolling_std_7",    # 7-day demand volatility
    "rolling_mean_14",  # 14-day moving average demand
    "wday",             # Day of week (1=Sat, 2=Sun, ... 7=Fri)
    "is_weekend",       # Binary weekend indicator
    "month",            # Calendar month (1-12)
    "snap_active",      # Binary SNAP food stamp subsidy flag for state
    "event_memorial",   # Binary flag for Memorial Day surge window
    "event_nba_finals", # Binary flag for NBA Finals viewership surge
    "sell_price",       # Current unit selling price
    "cat_code"          # Category encoding (FOODS=0, HOUSEHOLD=1, HOBBIES=2)
]

CATEGORY_MAP = {"FOODS": 0, "HOUSEHOLD": 1, "HOBBIES": 2}


class SupplyChainMLForecaster:
    """
    Supervised Machine Learning Forecasting Engine.
    Trains on historical sales time series and performs high-speed batch multi-step forward inference.
    """

    def __init__(self):
        self.model = HistGradientBoostingRegressor(
            max_iter=100,
            learning_rate=0.08,
            max_depth=6,
            min_samples_leaf=20,
            l2_regularization=0.1,
            random_state=42
        )
        self.is_trained = False
        self.training_metadata = {}

    def train_model(self, sales_history_df, calendar_map, prices_dict, val_days=10):
        """
        Constructs training dataset from historical sales matrix and fits the Gradient Boosting Regressor.
        """
        print("Constructing Supervised ML Training Dataset with Lag & Rolling Features...")
        t0 = time.time()

        day_cols = [c for c in sales_history_df.columns if c.startswith("d_")]
        total_days = len(day_cols)

        X_train_list = []
        y_train_list = []
        X_val_list = []
        y_val_list = []

        train_end_idx = total_days - val_days

        for _, row in sales_history_df.iterrows():
            store_id = row["store_id"]
            item_id = row["item_id"]
            cat_id = row["cat_id"]
            state_id = row["state_id"]
            unit_price = float(prices_dict.get((store_id, item_id), 4.99))
            cat_code = CATEGORY_MAP.get(cat_id, 0)

            sales_series = row[day_cols].values.astype(float)

            # Sample every 2nd day to keep training dataset compact and fast
            for t_idx in range(28, total_days, 2):
                target_day_code = day_cols[t_idx]
                cal_info = calendar_map.get(target_day_code, {})
                snap_active = int(cal_info.get(f"snap_{state_id}", 0))

                wday = int(cal_info.get("wday", 4))
                is_weekend = 1 if wday in [1, 2] else 0
                month = int(cal_info.get("month", 5))
                event = cal_info.get("event_1", "")
                event_memorial = 1 if event == "MemorialDay" else 0
                event_nba_finals = 1 if event in ["NBAFinalsStart", "NBAFinalsEnd"] else 0

                hist_available = sales_series[:t_idx]
                target_val = sales_series[t_idx]

                lag_7 = float(hist_available[-7])
                lag_14 = float(hist_available[-14])
                lag_21 = float(hist_available[-21])
                lag_28 = float(hist_available[-28])

                last_7 = hist_available[-7:]
                last_14 = hist_available[-14:]
                rolling_mean_7 = float(np.mean(last_7))
                rolling_std_7 = float(np.std(last_7))
                rolling_mean_14 = float(np.mean(last_14))

                feat = [
                    lag_7, lag_14, lag_21, lag_28,
                    rolling_mean_7, rolling_std_7, rolling_mean_14,
                    wday, is_weekend, month,
                    snap_active, event_memorial, event_nba_finals,
                    unit_price, cat_code
                ]

                if t_idx < train_end_idx:
                    X_train_list.append(feat)
                    y_train_list.append(target_val)
                else:
                    X_val_list.append(feat)
                    y_val_list.append(target_val)

        X_train = np.array(X_train_list, dtype=np.float32)
        y_train = np.array(y_train_list, dtype=np.float32)
        X_val = np.array(X_val_list, dtype=np.float32)
        y_val = np.array(y_val_list, dtype=np.float32)

        print(f"Dataset compiled in {time.time()-t0:.2f}s! Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")
        print("Fitting HistGradientBoostingRegressor model...")

        fit_t0 = time.time()
        self.model.fit(X_train, y_train)
        fit_time = time.time() - fit_t0
        self.is_trained = True

        val_preds = np.clip(self.model.predict(X_val), 0, None)
        rmse = float(np.sqrt(mean_squared_error(y_val, val_preds)))
        mae = float(mean_absolute_error(y_val, val_preds))
        r2 = float(r2_score(y_val, val_preds))
        total_actual = float(np.sum(y_val))
        wape = float(np.sum(np.abs(y_val - val_preds)) / max(1.0, total_actual)) * 100.0

        print(f"Model Training Complete ({fit_time:.2f}s)!")
        print(f"Validation Metrics -> R2: {r2:.3f} | RMSE: {rmse:.2f} | MAE: {mae:.2f} | WAPE: {wape:.1f}%")

        print("Computing Feature Importances...")
        perm_res = permutation_importance(
            self.model, X_val[:1500], y_val[:1500],
            n_repeats=2, random_state=42
        )
        importances = np.maximum(0, perm_res.importances_mean)
        total_imp = max(1e-6, np.sum(importances))
        normalized_imp = {
            FEATURE_NAMES[i]: round(float(importances[i] / total_imp) * 100, 1)
            for i in range(len(FEATURE_NAMES))
        }
        sorted_imp = dict(sorted(normalized_imp.items(), key=lambda x: x[1], reverse=True))

        self.training_metadata = {
            "algorithm": "HistGradientBoostingRegressor",
            "framework": "scikit-learn 1.9.1",
            "hyperparameters": {
                "max_iter": 100,
                "learning_rate": 0.08,
                "max_depth": 6,
                "min_samples_leaf": 20,
                "l2_regularization": 0.1
            },
            "training_samples": int(X_train.shape[0]),
            "validation_samples": int(X_val.shape[0]),
            "features_count": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "feature_importances": sorted_imp,
            "metrics": {
                "r2_score": round(r2, 3),
                "rmse": round(rmse, 2),
                "mae": round(mae, 2),
                "wape_pct": round(wape, 1),
                "accuracy_pct": round(max(0.0, 100.0 - wape), 1)
            },
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        joblib.dump(self.model, MODEL_PATH)
        with open(METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self.training_metadata, f, indent=2)

        return self.training_metadata

    def predict_forward_horizon_batch(self, items_meta, history_matrix, forecast_days, calendar_map):
        """
        High-performance vectorized multi-step recursive forecasting for all SKUs simultaneously.
        items_meta: list of dicts with ['unit_price', 'cat_code', 'state_id', 'avg_demand']
        history_matrix: numpy array of shape (N_SKUS, 30)
        Returns:
            predictions_matrix: numpy array of shape (N_SKUS, 28)
            spike_flags_per_item: list of lists with detected demand spikes
        """
        N = len(items_meta)
        n_days = len(forecast_days)
        print(f"Executing batch ML prediction for {N} SKUs across {n_days} days forward...")

        # Copy running history: shape (N, 30 + 28)
        running_history = np.zeros((N, 30 + n_days), dtype=np.float32)
        running_history[:, :30] = history_matrix

        prices = np.array([m["unit_price"] for m in items_meta], dtype=np.float32)
        cat_codes = np.array([m["cat_code"] for m in items_meta], dtype=np.float32)
        states = [m["state_id"] for m in items_meta]
        avg_demands = np.array([m["avg_demand"] for m in items_meta], dtype=np.float32)

        t_start = time.time()

        for d_idx, day_code in enumerate(forecast_days):
            current_len = 30 + d_idx
            cal_info = calendar_map.get(day_code, {})

            wday = int(cal_info.get("wday", 4))
            is_weekend = 1.0 if wday in [1, 2] else 0.0
            month = float(cal_info.get("month", 5))
            event = cal_info.get("event_1", "")
            event_memorial = 1.0 if event == "MemorialDay" else 0.0
            event_nba_finals = 1.0 if event in ["NBAFinalsStart", "NBAFinalsEnd"] else 0.0

            # Vectorized feature computation across all N items:
            lag_7 = running_history[:, current_len - 7]
            lag_14 = running_history[:, current_len - 14]
            lag_21 = running_history[:, current_len - 21]
            lag_28 = running_history[:, current_len - 28]

            rolling_mean_7 = np.mean(running_history[:, current_len - 7:current_len], axis=1)
            rolling_std_7 = np.std(running_history[:, current_len - 7:current_len], axis=1)
            rolling_mean_14 = np.mean(running_history[:, current_len - 14:current_len], axis=1)

            snap_active = np.array([float(cal_info.get(f"snap_{st}", 0)) for st in states], dtype=np.float32)

            # Assemble batch feature matrix: shape (N, 15)
            X_batch = np.column_stack([
                lag_7, lag_14, lag_21, lag_28,
                rolling_mean_7, rolling_std_7, rolling_mean_14,
                np.full(N, wday, dtype=np.float32),
                np.full(N, is_weekend, dtype=np.float32),
                np.full(N, month, dtype=np.float32),
                snap_active,
                np.full(N, event_memorial, dtype=np.float32),
                np.full(N, event_nba_finals, dtype=np.float32),
                prices,
                cat_codes
            ])

            # Model Batch Inference (all N items in 1 single vectorized call)
            preds = np.clip(self.model.predict(X_batch), 0.0, None)
            preds = np.round(preds, 1)

            running_history[:, current_len] = preds

        predictions_matrix = running_history[:, 30:]
        print(f"Batch ML multi-step forward inference completed in {time.time()-t_start:.2f}s!")

        # Detect spikes
        spike_flags_per_item = []
        for i in range(N):
            item_spikes = []
            base = max(0.5, avg_demands[i])
            for d_idx, day_code in enumerate(forecast_days):
                pred_val = predictions_matrix[i, d_idx]
                if pred_val >= 1.45 * base:
                    cal_info = calendar_map.get(day_code, {})
                    spike_pct = round(((pred_val / base) - 1.0) * 100, 1)
                    reason = cal_info.get("event_1") or ("SNAP Benefit Surge" if cal_info.get(f"snap_{states[i]}", 0) else "Weekend Demand Peak")
                    item_spikes.append({
                        "day_index": d_idx + 1,
                        "date": cal_info.get("date"),
                        "spike_pct": spike_pct,
                        "reason": reason
                    })
            spike_flags_per_item.append(item_spikes)

        return predictions_matrix, spike_flags_per_item
