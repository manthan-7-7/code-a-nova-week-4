"""CLI script to run latency and SLA benchmark against ShieldML inference API."""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from shieldml.config import settings
from shieldml.benchmarking.latency_benchmark import run_latency_benchmark
from shieldml.utils.logger import get_logger

logger = get_logger("scripts.run_benchmark")


def main():
    parser = argparse.ArgumentParser(description="Run ShieldML latency and SLA benchmark.")
    parser.add_argument("--requests", type=int, default=500, help="Number of benchmark requests (default: 500)")
    parser.add_argument("--warmup", type=int, default=25, help="Number of warmup requests (default: 25)")
    parser.add_argument("--target-ms", type=float, default=settings.TARGET_LATENCY_MS, help="SLA target ms (default: 50.0)")
    args = parser.parse_args()

    json_path = settings.ARTIFACTS_DIR / "benchmark_results.json"
    md_path = settings.BASE_DIR / "BENCHMARK_REPORT.md"

    run_latency_benchmark(
        n_requests=args.requests,
        warmup_requests=args.warmup,
        target_sla_ms=args.target_ms,
        output_json_path=json_path,
        output_markdown_path=md_path,
    )


if __name__ == "__main__":
    main()
