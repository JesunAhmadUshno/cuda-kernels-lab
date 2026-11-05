"""Level 3 Deep profiler benchmark, run on the fleet GPU node.

Run on BlackBox (NVIDIA GeForce RTX 5050 Laptop GPU, torch
2.11.0+cu128). This script is NOT run on the Muse VM: it needs
a CUDA device. Its output, results_profiler.json, is committed
and verified on the VM by analysis_deep.py.

Method:
- Timing: CUDA events, median of repeated runs after warmup,
  correctness checked against a reference before timing.
- Profiler: torch.profiler (CPU and CUDA activities) per case,
  recording the kernel rows, their counts, and total CUDA time.
- Routes, labelled exactly in every row:
  * "native op": a single PyTorch native op (torch.matmul,
    torch.softmax). torch.softmax is a fused single-kernel op.
  * "unfused manual (native ops)": softmax built from separate
    PyTorch ops (amax, sub, exp, sum, div), several kernels.
  * "compiled": torch.compile of the softmax, only when it
    actually runs on this node. The attempt result, including
    the exact failure text when it fails, is recorded in the
    output header either way. No compiled row is emitted when
    the route does not run.
- No custom CUDA kernel is timed or claimed: the node has no
  CUDA toolkit (see results header fields).
"""
import json
import os
import shutil
import statistics

import torch
from torch.profiler import ProfilerActivity, profile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_profiler.json")


def cuda_median_ms(fn, warmup=5, reps=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(stop))
    return round(statistics.median(times), 4)


def profile_case(fn, warmup=3, reps=10):
    """torch.profiler case. Records CUDA kernel rows when CUPTI
    is available on the node, plus the aten op counts from the
    CPU activity, which are recorded either way. On this node
    CUPTI intermittently fails to initialise (recorded in the
    output header of the first Deep run), so the CUDA-event
    stage table below is the timing profile of record and this
    table carries the op structure."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU,
                             ProfilerActivity.CUDA]) as prof:
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
    rows = []
    total_cuda_us = 0.0
    total_count = 0
    op_rows = []
    for evt in prof.key_averages():
        cuda_us = getattr(evt, "device_time_total", None)
        if cuda_us is None:
            cuda_us = getattr(evt, "cuda_time_total", 0.0)
        if cuda_us and cuda_us > 0:
            rows.append({
                "name": evt.key,
                "count": evt.count,
                "cuda_time_us": round(float(cuda_us), 1),
            })
            total_cuda_us += float(cuda_us)
            total_count += evt.count
        if evt.key.startswith("aten::"):
            op_rows.append({
                "name": evt.key,
                "count": evt.count,
                "cpu_time_us": round(float(evt.cpu_time_total), 1),
            })
    rows.sort(key=lambda r: -r["cuda_time_us"])
    op_rows.sort(key=lambda r: -r["count"])
    return {
        "reps": reps,
        "cuda_activity_available": len(rows) > 0,
        "kernel_count_total": total_count,
        "distinct_kernels": len(rows),
        "total_cuda_time_us": round(total_cuda_us, 1),
        "top_kernels": rows[:6],
        "aten_ops": op_rows[:8],
    }


def stage_profile_softmax(x):
    """CUDA-event timing per stage of the unfused softmax, plus
    the fused single op, on the same input. This is the profile
    table of record when CUPTI is unavailable."""
    stages = []

    def time_stage(name, fn):
        ms = cuda_median_ms(fn, warmup=5, reps=30)
        stages.append({"stage": name, "median_ms": ms})

    time_stage("amax (row max)", lambda: x.amax(dim=-1, keepdim=True))
    m = x.amax(dim=-1, keepdim=True)
    time_stage("exp(x - max)", lambda: torch.exp(x - m))
    e = torch.exp(x - m)
    time_stage("sum (row sum)", lambda: e.sum(dim=-1, keepdim=True))
    s = e.sum(dim=-1, keepdim=True)
    time_stage("div (normalise)", lambda: e / s)

    def unfused(t):
        mm = t.amax(dim=-1, keepdim=True)
        ee = torch.exp(t - mm)
        ss = ee.sum(dim=-1, keepdim=True)
        return ee / ss

    time_stage("unfused full sequence", lambda: unfused(x))
    time_stage("native fused softmax (single op)",
               lambda: torch.softmax(x, dim=-1))
    return stages


def main():
    if not torch.cuda.is_available():
        print("CUDA not available: this script must run on the GPU node")
        return 1
    try:
        import torch.utils.cpp_extension as cpp_ext
        cuda_home = cpp_ext.CUDA_HOME
    except Exception as exc:  # noqa: BLE001
        cuda_home = f"unavailable: {exc}"
    device = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)

    # torch.compile attempt, recorded either way.
    compile_info = {"attempted": True, "success": False, "error": None}
    compiled_softmax = None
    try:
        candidate = torch.compile(
            lambda t: torch.softmax(t, dim=-1))
        probe = torch.randn(64, 64, device="cuda")
        out = candidate(probe)
        torch.cuda.synchronize()
        ref = torch.softmax(probe, dim=-1)
        if not torch.allclose(out, ref, atol=1e-6):
            raise RuntimeError("compiled softmax mismatch vs native")
        compiled_softmax = candidate
        compile_info["success"] = True
    except Exception as exc:  # noqa: BLE001
        compile_info["error"] = f"{type(exc).__name__}: {str(exc)[:600]}"

    benchmarks = []
    profiler_rows = []

    # ---- matmul, native op, three sizes
    for size in (512, 1024, 2048):
        a = torch.randn(size, size, device="cuda")
        b = torch.randn(size, size, device="cuda")
        ref = a @ b
        got = a @ b
        correct = bool(torch.allclose(got, ref, atol=1e-3, rtol=1e-3))
        ms = cuda_median_ms(lambda: a @ b, reps=20)
        gflops = round(2.0 * size ** 3 / (ms * 1e-3) / 1e9, 2)
        benchmarks.append({
            "kernel": "matmul_torch", "route": "native op",
            "size": size, "median_ms": ms, "gflops": gflops,
            "correct": correct,
        })
        if size == 1024:
            prof = profile_case(lambda: a @ b)
            profiler_rows.append({
                "case": "matmul_1024", "route": "native op", **prof})

    # ---- softmax: native fused op vs unfused manual sequence
    def softmax_unfused(t):
        m = t.amax(dim=-1, keepdim=True)
        e = torch.exp(t - m)
        s = e.sum(dim=-1, keepdim=True)
        return e / s

    for rows_n, cols in ((4096, 1024), (2048, 2048), (8192, 512)):
        x = torch.randn(rows_n, cols, device="cuda") * 2.0
        ref = torch.softmax(x, dim=-1)

        got_native = torch.softmax(x, dim=-1)
        correct_native = bool(
            (got_native - ref).abs().max().item() < 1e-6)
        ms_native = cuda_median_ms(lambda: torch.softmax(x, dim=-1))
        benchmarks.append({
            "kernel": "softmax_rows_torch", "route": "native op",
            "rows": rows_n, "cols": cols, "median_ms": ms_native,
            "gb_per_s": round(2.0 * rows_n * cols * 4 /
                              (ms_native * 1e-3) / 1e9, 2),
            "correct": correct_native,
        })

        got_unfused = softmax_unfused(x)
        correct_unfused = bool(
            (got_unfused - ref).abs().max().item() < 1e-5)
        ms_unfused = cuda_median_ms(lambda: softmax_unfused(x))
        benchmarks.append({
            "kernel": "softmax_rows_unfused_manual",
            "route": "unfused manual (native ops)",
            "rows": rows_n, "cols": cols, "median_ms": ms_unfused,
            "gb_per_s": round(2.0 * rows_n * cols * 4 /
                              (ms_unfused * 1e-3) / 1e9, 2),
            "correct": correct_unfused,
        })

        if compiled_softmax is not None:
            got_c = compiled_softmax(x)
            correct_c = bool((got_c - ref).abs().max().item() < 1e-5)
            ms_c = cuda_median_ms(lambda: compiled_softmax(x))
            benchmarks.append({
                "kernel": "softmax_rows_compiled",
                "route": "compiled",
                "rows": rows_n, "cols": cols, "median_ms": ms_c,
                "gb_per_s": round(2.0 * rows_n * cols * 4 /
                                  (ms_c * 1e-3) / 1e9, 2),
                "correct": correct_c,
            })

        if (rows_n, cols) == (4096, 1024):
            profiler_rows.append({
                "case": "softmax_4096x1024", "route": "native op",
                **profile_case(lambda: torch.softmax(x, dim=-1))})
            profiler_rows.append({
                "case": "softmax_4096x1024",
                "route": "unfused manual (native ops)",
                **profile_case(lambda: softmax_unfused(x))})
            if compiled_softmax is not None:
                profiler_rows.append({
                    "case": "softmax_4096x1024", "route": "compiled",
                    **profile_case(lambda: compiled_softmax(x))})

    stage_rows = stage_profile_softmax(
        torch.randn(4096, 1024, device="cuda") * 2.0)

    data = {
        "stage_profile_softmax_4096x1024": stage_rows,
        "device": device,
        "compute_capability": f"{cap[0]}.{cap[1]}",
        "torch_version": torch.__version__,
        "cuda_home": cuda_home,
        "nvcc_on_path": shutil.which("nvcc") is not None,
        "compile_softmax": compile_info,
        "note": ("PyTorch routes only: native op, unfused manual "
                 "(native ops), and compiled when it runs. No custom "
                 "CUDA kernel rows: no CUDA toolkit on this node."),
        "benchmarks": benchmarks,
        "profiler": profiler_rows,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    print(json.dumps(data, indent=2))
    print(f"results_profiler.json written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
