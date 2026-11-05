"""Verification analysis for Level 3 Deep profiler benchmarks.

Loads results_profiler.json, the measured profiler benchmark
from the fleet GPU node (BlackBox, NVIDIA GeForce RTX 5050
Laptop GPU), and recomputes every throughput from the stored
size and median_ms:

- matmul: FLOPs = 2 * size^3, GFLOP/s = FLOPs / seconds / 1e9.
- softmax rows (native op and unfused manual): bytes moved =
  2 * rows * cols * 4 (one read, one write, float32), GB/s =
  bytes / seconds / 1e9. The unfused sequence moves more traffic
  in reality; the model is kept identical so the two routes are
  comparable on one axis, and the README says so.

A stored value verifies when the recomputed value matches it
within 1 percent. Every row must carry correct = true and a
route label from the allowed set. There must be no row labelled
custom: no custom CUDA kernel ran this cycle.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results_profiler.json")
TOLERANCE = 0.01
ALLOWED_ROUTES = {"native op", "unfused manual (native ops)",
                  "compiled", "custom"}


def matmul_gflops(size, median_ms):
    return 2.0 * size ** 3 / (median_ms * 1e-3) / 1e9


def softmax_gb_per_s(rows, cols, median_ms):
    return 2.0 * rows * cols * 4 / (median_ms * 1e-3) / 1e9


def recompute(row):
    kernel = row["kernel"]
    if kernel == "matmul_torch":
        return "gflops", matmul_gflops(row["size"], row["median_ms"]), row["gflops"]
    if kernel in ("softmax_rows_torch", "softmax_rows_unfused_manual",
                  "softmax_rows_compiled"):
        return ("gb_per_s",
                softmax_gb_per_s(row["rows"], row["cols"], row["median_ms"]),
                row["gb_per_s"])
    raise ValueError(f"unknown kernel row: {kernel}")


def within_tolerance(recomputed, stored, tolerance=TOLERANCE):
    if stored == 0:
        return recomputed == 0
    return abs(recomputed - stored) / abs(stored) <= tolerance


def load_results(path=RESULTS):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def verify(data, tolerance=TOLERANCE):
    """Assert every benchmark row recomputes within tolerance,
    is correct, and carries an allowed route label. Asserts no
    row is labelled custom. Returns one checked dict per row."""
    checked = []
    for row in data["benchmarks"]:
        assert row["correct"] is True, f"row not correct: {row}"
        assert row["route"] in ALLOWED_ROUTES, f"bad route: {row}"
        assert row["route"] != "custom", (
            f"custom kernel row present but no custom kernel ran: {row}")
        kind, recomputed, stored = recompute(row)
        assert within_tolerance(recomputed, stored, tolerance), (
            f"{row['kernel']}: recomputed {recomputed:.2f} {kind} vs "
            f"stored {stored} differs by more than {tolerance:.0%}")
        checked.append({
            "kernel": row["kernel"],
            "route": row["route"],
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
        "PyTorch routes only, CUDA events, median. Not custom kernels.",
        f"{'kernel':<30} {'route':<28} {'median_ms':>10} "
        f"{'stored':>10} {'recomputed':>11}",
    ]
    for row, chk in zip(data["benchmarks"], checked):
        lines.append(
            f"{row['kernel']:<30} {row['route']:<28} "
            f"{row['median_ms']:>10} {chk['stored']:>10} "
            f"{chk['recomputed']:>11}")
    return "\n".join(lines)


if __name__ == "__main__":
    data = load_results()
    checked_rows = verify(data)
    print(format_table(data, checked_rows))
    print(f"verified: {len(checked_rows)} profiler rows within 1 percent")
