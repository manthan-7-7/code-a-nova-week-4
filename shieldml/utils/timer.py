"""High-precision latency measurement utilities."""

import time
from typing import Optional


class Timer:
    """Context manager and stopwatch for millisecond latency benchmarking.

    Example:
        with Timer() as t:
            run_inference()
        print(f"Latency: {t.elapsed_ms:.2f} ms")
    """

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.elapsed_ms: float = 0.0

    def start(self) -> "Timer":
        """Start the timer."""
        self.start_time = time.perf_counter()
        self.end_time = None
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed milliseconds."""
        if self.start_time is None:
            raise RuntimeError("Timer was not started.")
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000.0
        return self.elapsed_ms

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
