"""Automated Feature Engineering for Real-Time Financial Fraud Detection.

Computes:
1. Historical transaction velocity features (1-hour, 24-hour, 7-day windows)
2. Merchant amount z-scores (learned strictly from training data)
3. Time-of-day and day-of-week sine/cosine cyclical encodings
4. Device fingerprint frequency and new-device detection
5. Distance and amount log-transforms
"""

from typing import Dict, Any, Optional, Set, Tuple
import numpy as np
import pandas as pd

from shieldml.config import settings
from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.features.engineer")


class FeatureEngineer:
    """Computes fraud detection features while preventing temporal data leakage."""

    def __init__(self):
        # Learned statistics from training set
        self.merchant_stats_: Dict[str, Tuple[float, float]] = {}  # merchant_id -> (mean, std)
        self.global_merchant_mean_: float = 0.0
        self.global_merchant_std_: float = 1.0

        self.device_frequencies_: Dict[str, int] = {}
        self.user_known_devices_: Set[Tuple[str, str]] = set()

        self.is_fitted_: bool = False

    def fit(self, train_df: pd.DataFrame) -> "FeatureEngineer":
        """Fits historical merchant, device, and user profiles strictly on training data.

        Args:
            train_df: Chronological training partition.

        Returns:
            FeatureEngineer: Fitted instance.
        """
        logger.info(f"Fitting FeatureEngineer on {len(train_df)} training transactions...")
        df = train_df.copy()

        # Merchant amount statistics
        self.global_merchant_mean_ = float(df["amount"].mean())
        self.global_merchant_std_ = float(max(df["amount"].std(), 1.0))

        merchant_grouped = df.groupby("merchant_id")["amount"].agg(["mean", "std", "count"])
        for m_id, row in merchant_grouped.iterrows():
            m_mean = float(row["mean"])
            m_std = float(row["std"]) if not pd.isna(row["std"]) and row["std"] > 0 else self.global_merchant_std_
            self.merchant_stats_[str(m_id)] = (m_mean, m_std)

        # Device frequencies
        if "device_id" in df.columns:
            dev_counts = df["device_id"].value_counts().to_dict()
            self.device_frequencies_ = {str(k): int(v) for k, v in dev_counts.items()}

            # User-device associations
            pairs = set(zip(df["user_id"].astype(str), df["device_id"].astype(str)))
            self.user_known_devices_ = pairs

        self.is_fitted_ = True
        logger.info(f"FeatureEngineer fitted: {len(self.merchant_stats_)} merchants, "
                    f"{len(self.device_frequencies_)} devices profiled.")
        return self

    def transform(self, df: pd.DataFrame, is_training: bool = False) -> pd.DataFrame:
        """Transforms raw transaction data into engineered feature vectors.

        Args:
            df: Input DataFrame containing raw transaction columns.
            is_training: If True, computes rolling features across the batch.

        Returns:
            pd.DataFrame: Feature-engineered DataFrame.
        """
        if not self.is_fitted_ and not is_training:
            raise RuntimeError("FeatureEngineer must be fitted on training data before transform.")

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # 1. Cyclical Time Features
        hours = df["timestamp"].dt.hour.values
        dayofweeks = df["timestamp"].dt.dayofweek.values

        df["hour_sin"] = np.sin(2.0 * np.pi * hours / 24.0)
        df["hour_cos"] = np.cos(2.0 * np.pi * hours / 24.0)
        df["day_sin"] = np.sin(2.0 * np.pi * dayofweeks / 7.0)
        df["day_cos"] = np.cos(2.0 * np.pi * dayofweeks / 7.0)
        df["is_weekend"] = (dayofweeks >= 5).astype(int)
        df["is_night"] = ((hours >= 0) & (hours <= 5)).astype(int)

        # 2. Log amount and distance
        df["log_amount"] = np.log1p(df["amount"].clip(lower=0.0))
        if "distance_from_home" in df.columns:
            df["log_distance"] = np.log1p(df["distance_from_home"].clip(lower=0.0))
        else:
            df["log_distance"] = 0.0

        # 3. Merchant Amount Z-Score (using learned training stats)
        z_scores = []
        for m_id, amt in zip(df["merchant_id"], df["amount"]):
            mean, std = self.merchant_stats_.get(
                str(m_id), (self.global_merchant_mean_, self.global_merchant_std_)
            )
            z = (amt - mean) / max(std, 1e-4)
            z_scores.append(float(np.clip(z, -10.0, 10.0)))
        df["merchant_amount_zscore"] = z_scores

        # 4. Device features
        if "device_id" in df.columns:
            dev_freqs = [self.device_frequencies_.get(str(d), 0) for d in df["device_id"]]
            df["device_usage_count"] = dev_freqs

            # Is new device for user?
            new_dev_flags = []
            for u, d in zip(df["user_id"], df["device_id"]):
                is_known = (str(u), str(d)) in self.user_known_devices_
                new_dev_flags.append(0 if is_known else 1)
            df["is_new_device_for_user"] = new_dev_flags
        else:
            df["device_usage_count"] = 0
            df["is_new_device_for_user"] = 0

        # 5. Historical Velocity Features (1h, 24h, 7d)
        df = self._compute_historical_velocities(df)

        return df

    def _compute_historical_velocities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Computes backward-looking rolling velocity metrics without forward leakage.

        For each row, metrics reflect transactions that occurred strictly prior to this transaction.
        """
        # Ensure chronological order
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        timestamps = df["timestamp"].values
        user_ids = df["user_id"].values
        amounts = df["amount"].values
        n = len(df)

        # Output arrays
        count_1h = np.zeros(n, dtype=np.int32)
        sum_1h = np.zeros(n, dtype=np.float32)
        mean_1h = np.zeros(n, dtype=np.float32)

        count_24h = np.zeros(n, dtype=np.int32)
        sum_24h = np.zeros(n, dtype=np.float32)
        mean_24h = np.zeros(n, dtype=np.float32)

        count_7d = np.zeros(n, dtype=np.int32)
        sum_7d = np.zeros(n, dtype=np.float32)
        mean_7d = np.zeros(n, dtype=np.float32)

        ratio_to_mean_24h = np.ones(n, dtype=np.float32)

        sec_1h = np.timedelta64(settings.WINDOW_1H_SECS, "s")
        sec_24h = np.timedelta64(settings.WINDOW_24H_SECS, "s")
        sec_7d = np.timedelta64(settings.WINDOW_7D_SECS, "s")

        # Group indices by user for fast backward search
        from collections import defaultdict
        user_history = defaultdict(list)  # user -> list of (timestamp, amount)

        for i in range(n):
            u = user_ids[i]
            t = timestamps[i]
            curr_amt = amounts[i]

            hist = user_history[u]
            if not hist:
                # First transaction for this user
                user_history[u].append((t, curr_amt))
                continue

            # Binary search or backward traversal for window cutoffs
            t_1h_cutoff = t - sec_1h
            t_24h_cutoff = t - sec_24h
            t_7d_cutoff = t - sec_7d

            c1, s1 = 0, 0.0
            c24, s24 = 0, 0.0
            c7, s7 = 0, 0.0

            # Traverse backwards (most recent to oldest)
            for prev_t, prev_amt in reversed(hist):
                if prev_t >= t_7d_cutoff:
                    c7 += 1
                    s7 += prev_amt
                    if prev_t >= t_24h_cutoff:
                        c24 += 1
                        s24 += prev_amt
                        if prev_t >= t_1h_cutoff:
                            c1 += 1
                            s1 += prev_amt
                else:
                    # Outside 7-day window, older records won't match
                    break

            count_1h[i] = c1
            sum_1h[i] = s1
            mean_1h[i] = s1 / c1 if c1 > 0 else 0.0

            count_24h[i] = c24
            sum_24h[i] = s24
            m24 = s24 / c24 if c24 > 0 else 0.0
            mean_24h[i] = m24
            ratio_to_mean_24h[i] = curr_amt / (m24 + 1e-4) if c24 > 0 else 1.0

            count_7d[i] = c7
            sum_7d[i] = s7
            mean_7d[i] = s7 / c7 if c7 > 0 else 0.0

            # Append current to history
            hist.append((t, curr_amt))

        df["tx_count_user_1h"] = count_1h
        df["tx_amount_sum_user_1h"] = sum_1h
        df["tx_amount_mean_user_1h"] = mean_1h

        df["tx_count_user_24h"] = count_24h
        df["tx_amount_sum_user_24h"] = sum_24h
        df["tx_amount_mean_user_24h"] = mean_24h
        df["amount_to_mean_ratio_24h"] = ratio_to_mean_24h

        df["tx_count_user_7d"] = count_7d
        df["tx_amount_sum_user_7d"] = sum_7d
        df["tx_amount_mean_user_7d"] = mean_7d

        return df

    def transform_single(
        self,
        transaction: Dict[str, Any],
        historical_user_txs: Optional[list] = None
    ) -> Dict[str, Any]:
        """Ultra-fast feature extraction for a single real-time transaction (< 1ms).

        Args:
            transaction: Dictionary of incoming transaction payload.
            historical_user_txs: Optional recent transactions for this user.

        Returns:
            dict: Computed feature map ready for model input.
        """
        ts = pd.to_datetime(transaction.get("timestamp", pd.Timestamp.now()))
        amount = float(transaction.get("amount", 0.0))
        m_id = str(transaction.get("merchant_id", "unknown"))
        u_id = str(transaction.get("user_id", "unknown"))
        d_id = str(transaction.get("device_id", "unknown"))
        cat = str(transaction.get("merchant_category", "retail"))
        channel = str(transaction.get("channel", "mobile_app"))
        dist = float(transaction.get("distance_from_home", 0.0))

        # Time cyclics
        hour = ts.hour
        dayofweek = ts.dayofweek
        hour_sin = float(np.sin(2.0 * np.pi * hour / 24.0))
        hour_cos = float(np.cos(2.0 * np.pi * hour / 24.0))
        day_sin = float(np.sin(2.0 * np.pi * dayofweek / 7.0))
        day_cos = float(np.cos(2.0 * np.pi * dayofweek / 7.0))
        is_weekend = int(dayofweek >= 5)
        is_night = int(0 <= hour <= 5)

        # Log amounts
        log_amount = float(np.log1p(max(0.0, amount)))
        log_distance = float(np.log1p(max(0.0, dist)))

        # Merchant Z-score
        mean, std = self.merchant_stats_.get(
            m_id, (self.global_merchant_mean_, self.global_merchant_std_)
        )
        z = float(np.clip((amount - mean) / max(std, 1e-4), -10.0, 10.0))

        # Device features
        dev_usage = self.device_frequencies_.get(d_id, 0)
        is_new_dev = 0 if (u_id, d_id) in self.user_known_devices_ else 1

        # Velocity features
        # If caller provides pre-computed or past list of txs:
        c1, s1 = 0, 0.0
        c24, s24 = 0, 0.0
        c7, s7 = 0, 0.0

        if historical_user_txs:
            t_curr = ts.to_datetime64()
            for prev_ts_str, prev_amt in historical_user_txs:
                prev_t = pd.to_datetime(prev_ts_str).to_datetime64()
                diff_sec = (t_curr - prev_t).astype("timedelta64[s]").astype(float)
                if 0 < diff_sec <= 604800:
                    c7 += 1
                    s7 += prev_amt
                    if diff_sec <= 86400:
                        c24 += 1
                        s24 += prev_amt
                        if diff_sec <= 3600:
                            c1 += 1
                            s1 += prev_amt
        else:
            # Fallback to direct field inputs if passed in payload
            c1 = int(transaction.get("tx_count_user_1h", 0))
            s1 = float(transaction.get("tx_amount_sum_user_1h", 0.0))
            c24 = int(transaction.get("tx_count_user_24h", 0))
            s24 = float(transaction.get("tx_amount_sum_user_24h", 0.0))
            c7 = int(transaction.get("tx_count_user_7d", 0))
            s7 = float(transaction.get("tx_amount_sum_user_7d", 0.0))

        m1 = s1 / c1 if c1 > 0 else 0.0
        m24 = s24 / c24 if c24 > 0 else 0.0
        m7 = s7 / c7 if c7 > 0 else 0.0
        ratio_24 = amount / (m24 + 1e-4) if c24 > 0 else 1.0

        return {
            "amount": amount,
            "log_amount": log_amount,
            "distance_from_home": dist,
            "log_distance": log_distance,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "is_weekend": is_weekend,
            "is_night": is_night,
            "merchant_amount_zscore": z,
            "device_usage_count": dev_usage,
            "is_new_device_for_user": is_new_dev,
            "tx_count_user_1h": c1,
            "tx_amount_sum_user_1h": s1,
            "tx_amount_mean_user_1h": m1,
            "tx_count_user_24h": c24,
            "tx_amount_sum_user_24h": s24,
            "tx_amount_mean_user_24h": m24,
            "amount_to_mean_ratio_24h": ratio_24,
            "tx_count_user_7d": c7,
            "tx_amount_sum_user_7d": s7,
            "tx_amount_mean_user_7d": m7,
            "merchant_category": cat,
            "channel": channel,
        }
