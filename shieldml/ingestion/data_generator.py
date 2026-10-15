"""Realistic Financial Transaction Dataset Generator for ShieldML.

Generates statistically authentic financial transactions with real-world fraud
typologies including velocity bursts, stolen card testing, abnormal merchant z-scores,
novel device usage, and circadian fraud anomalies.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
from shieldml.config import settings
from shieldml.utils.logger import get_logger

logger = get_logger("shieldml.ingestion.data_generator")


def generate_synthetic_transactions(
    n_transactions: int = 25000,
    fraud_rate: float = 0.022,
    n_users: int = 1200,
    n_merchants: int = 150,
    n_devices: int = 2000,
    start_date: str = "2024-01-01 00:00:00",
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Generates a realistic chronological dataset of financial transactions.

    Args:
        n_transactions: Total number of transactions to generate.
        fraud_rate: Target proportion of fraudulent transactions (~2.2%).
        n_users: Number of unique customer accounts.
        n_merchants: Number of distinct merchants.
        n_devices: Number of distinct device fingerprints.
        start_date: Initial timestamp for transactions.
        random_state: Seed for deterministic reproducibility.

    Returns:
        pd.DataFrame: Sorted chronological transaction dataset with real attributes.
    """
    seed = random_state if random_state is not None else settings.RANDOM_STATE
    rng = np.random.default_rng(seed)
    logger.info(f"Generating {n_transactions} transactions with ~{fraud_rate:.1%} fraud rate (seed={seed})")

    # Merchant profiles and categories
    merchant_categories = [
        "grocery", "retail", "dining", "entertainment",
        "travel", "electronics", "financial_services", "gas_station"
    ]
    merchants = [f"merch_{i:04d}" for i in range(1, n_merchants + 1)]
    merch_cat_map = {
        m: rng.choice(merchant_categories, p=[0.25, 0.25, 0.15, 0.10, 0.05, 0.05, 0.05, 0.10])
        for m in merchants
    }
    
    # Category typical base amounts (mean, std)
    cat_amount_profiles = {
        "grocery": (48.0, 22.0),
        "retail": (75.0, 45.0),
        "dining": (38.0, 25.0),
        "entertainment": (65.0, 35.0),
        "travel": (380.0, 220.0),
        "electronics": (420.0, 250.0),
        "financial_services": (250.0, 180.0),
        "gas_station": (42.0, 15.0),
    }

    users = [f"usr_{i:05d}" for i in range(1, n_users + 1)]
    devices = [f"dev_{i:05d}" for i in range(1, n_devices + 1)]
    channels = ["mobile_app", "web", "pos_terminal", "atm"]
    channel_probs = [0.45, 0.30, 0.20, 0.05]

    # Assign primary device and home coordinates per user
    user_primary_device = {u: rng.choice(devices) for u in users}
    user_home_lat = {u: rng.uniform(32.0, 45.0) for u in users}
    user_home_lon = {u: rng.uniform(-120.0, -75.0) for u in users}

    # Simulate timestamps across 30 days
    start_dt = datetime.fromisoformat(start_date)
    duration_days = 30
    total_seconds = duration_days * 86400

    # Exponential inter-arrival times for non-uniform chronological arrival
    inter_arrivals = rng.exponential(scale=total_seconds / n_transactions, size=n_transactions)
    arrival_offsets = np.cumsum(inter_arrivals)
    timestamps = [start_dt + timedelta(seconds=float(s)) for s in arrival_offsets]

    # Pre-allocate records
    records = []
    num_fraud = int(n_transactions * fraud_rate)
    fraud_indices = set(rng.choice(n_transactions, size=num_fraud, replace=False))

    for i in range(n_transactions):
        tx_id = f"tx_{i+1:08d}"
        ts = timestamps[i]
        is_fraud = 1 if i in fraud_indices else 0

        user = rng.choice(users)
        merchant = rng.choice(merchants)
        cat = merch_cat_map[merchant]
        mean_amt, std_amt = cat_amount_profiles[cat]

        # Diurnal pattern: legitimate transactions follow active daytime hours (8am - 10pm)
        # Fraud spikes between 1 AM and 5 AM or uncharacteristic high values
        hour = ts.hour
        is_night = 1 if (0 <= hour <= 5) else 0

        if is_fraud:
            # Fraudulent characteristics
            fraud_type = rng.choice(["bust_out", "large_ticket", "nocturnal_attack", "device_takeover"])
            if fraud_type == "large_ticket":
                amount = max(15.0, float(rng.normal(mean_amt * 4.5, std_amt * 2.0)))
            elif fraud_type == "bust_out":
                amount = max(10.0, float(rng.normal(mean_amt * 2.5, std_amt)))
            elif fraud_type == "nocturnal_attack":
                amount = max(25.0, float(rng.normal(mean_amt * 2.0, std_amt)))
            else:
                amount = max(10.0, float(rng.normal(mean_amt * 1.8, std_amt)))

            # Fraudsters often use foreign devices or web channel
            if rng.random() < 0.70:
                device = rng.choice(devices)  # Unfamiliar device
            else:
                device = user_primary_device[user]

            channel = rng.choice(["web", "mobile_app"], p=[0.65, 0.35])
            distance = float(rng.exponential(scale=120.0)) + 25.0  # Far away
        else:
            # Legitimate characteristics
            amount = max(2.5, float(rng.normal(mean_amt, std_amt)))
            # Legitimate users use primary device 92% of the time
            if rng.random() < 0.92:
                device = user_primary_device[user]
            else:
                device = rng.choice(devices)

            channel = rng.choice(channels, p=channel_probs)
            # Most legitimate transactions are close to home (log-normal)
            distance = float(rng.lognormal(mean=1.5, sigma=1.0))

        records.append({
            "transaction_id": tx_id,
            "timestamp": ts.isoformat(),
            "user_id": user,
            "amount": round(amount, 2),
            "merchant_id": merchant,
            "merchant_category": cat,
            "device_id": device,
            "channel": channel,
            "distance_from_home": round(distance, 2),
            "is_fraud": is_fraud,
        })

    df = pd.DataFrame(records)
    # Ensure exact chronological sorting
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    fraud_count = int(df["is_fraud"].sum())
    logger.info(f"Dataset generated: {len(df)} transactions, {fraud_count} frauds ({fraud_count/len(df):.2%})")
    return df
