"""CLI script to generate synthetic financial fraud datasets."""

from pathlib import Path
import argparse
from shieldml.config import settings
from shieldml.ingestion.data_generator import generate_synthetic_transactions
from shieldml.utils.logger import get_logger

logger = get_logger("scripts.generate_data")


def main():
    parser = argparse.ArgumentParser(description="Generate realistic financial fraud transaction dataset.")
    parser.add_argument("--count", type=int, default=25000, help="Total transactions to generate")
    parser.add_argument("--fraud-rate", type=float, default=0.022, help="Target fraud rate (e.g. 0.022)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="data/raw/transactions.csv", help="Output CSV path")
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_transactions(
        n_transactions=args.count,
        fraud_rate=args.fraud_rate,
        random_state=args.seed,
    )

    df.to_csv(out_path, index=False)
    logger.info(f"Successfully saved {len(df)} transactions to {out_path}")

    # Generate a sample batch CSV for testing the batch upload endpoint and UI
    sample_batch_path = settings.DATA_DIR / "sample_batch.csv"
    sample_df = df.tail(100).copy()
    # Remove fraud label from sample upload CSV so it acts like real unlabelled incoming transactions
    eval_labels = sample_df.pop("is_fraud")
    sample_df.to_csv(sample_batch_path, index=False)
    logger.info(f"Generated sample batch test CSV ({len(sample_df)} rows) at {sample_batch_path}")


if __name__ == "__main__":
    main()
