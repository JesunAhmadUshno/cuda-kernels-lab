"""Verification analysis for Level 2 Build GPU benchmarks.

Loads results_torch_bench.json, the measured PyTorch native op
benchmarks from the fleet GPU node (BlackBox, NVIDIA GeForce RTX
5050 Laptop GPU), and recomputes every throughput from the stored
size and median_ms:

- vector add: bytes moved = 3 * n * 4 (two reads, one write,
  float32), GB/s = bytes / seconds / 1e9.
- matmul: FLOPs = 2 * size^3, GFLOP/s = FLOPs / seconds / 1e9.
- softmax rows: bytes moved = 2 * rows * cols * 4 (one read, one
  write, float32), GB/s as above.

A stored value verifies when the recomputed value matches it
within 1 percent (the stored medians are rounded to 4 significant
figures, so exact equality is not expected). Every row must also
carry correct = true. These are PyTorch native ops, never custom
kernel results.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_torch_bench.json")
TOLERANCE = 0.01


def vector_gb_per_s(n, median_ms):
    return 3.0 * n * 4 / (median_ms * 1e-3) / 1e9


def matmul_gflops(size, median_ms):
    return 2.0 * size ** 3 / (median_ms * 1e-3) / 1e9


def softmax_gb_per_s(rows, cols, median_ms):
    return 2.0 * rows * cols * 4 / (median_ms * 1e-3) / 1e9


def recompute(row):
    """Return (kind, recomputed value, stored value) for one row."""
    kernel = row["kernel"]
    if kernel == "vector_add_torch":
        return "gb_per_s", vector_gb_per_s(row["n"], row["median_ms"]), row["gb_per_s"]
    if kernel == "matmul_torch":
        return "gflops", matmul_gflops(row["size"], row["median_ms"]), row["gflops"]
    if kernel == "softmax_rows_torch":
        return "gb_per_s", softmax_gb_per_s(row["rows"], row["cols"], row["median_ms"]), row["gb_per_s"]
    raise ValueError(f"unknown kernel row: {kernel}")


def within_tolerance(recomputed, stored, tolerance=TOLERANCE):
    if stored == 0:
        return recomputed == 0
    return abs(recomputed - stored) / abs(stored) <= tolerance


def load_results(path=RESULTS):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def verify(data, tolerance=TOLERANCE):
    """Assert every row recomputes within tolerance and is correct.

    Returns a list of dicts, one per row, with the recomputed value
    and the relative difference. Raises AssertionError on any
    mismatch or any correct flag that is not true.
    """
    checked = []
    for row in data["benchmarks"]:
        assert row["correct"] is True, f"row not correct: {row}"
        kind, recomputed, stored = recompute(row)
        assert within_tolerance(recomputed, stored, tolerance), (
            f"{row['kernel']}: recomputed {recomputed:.2f} {kind} vs "
            f"stored {stored} differs by more than {tolerance:.0%}")
        checked.append({
            "kernel": row["kernel"],
            "median_ms": row["median_ms"],
            "metric": kind,
            "stored": stored,
            "recomputed": round(recomputed, 2),
            "rel_diff": abs(recomputed - stored) / abs(stored),
            "correct": True,
        })
    return checked


def format_table(data, checked):
    lines = [
        f"device: {data['device']} (compute capability "
        f"{data['compute_capability']}, torch {data['torch_version']})",
        "PyTorch native ops, CUDA events, median. Not custom kernels.",
        f"{'kernel':<22} {'median_ms':>10} {'metric':>9} "
        f"{'stored':>10} {'recomputed':>11} {'correct':>8}",
    ]
    for row, chk in zip(data["benchmarks"], checked):
        lines.append(
            f"{row['kernel']:<22} {row['median_ms']:>10} "
            f"{chk['metric']:>9} {chk['stored']:>10} "
            f"{chk['recomputed']:>11} {str(row['correct']):>8}")
    return "\n".join(lines)


if __name__ == "__main__":
    data = load_results()
    checked_rows = verify(data)
    print(format_table(data, checked_rows))
    print(f"verified: {len(checked_rows)} GPU rows within 1 percent")
