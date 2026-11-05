# Level 2, Build: cuda-kernels-lab

Status: DONE, 2026-11-05. Spec: BUILD-SPEC.md in this folder.

## Build scope

From SPEC.md, the Build level plus the Added track: tests on the
standalone C++ tool, and one debugging session written up from a
real crash. Two named substitutions, both recorded in
BUILD-SPEC.md:

- Test framework: a self-contained assertion runner
  (cpp/tests.cpp) instead of GoogleTest or Catch2. Neither was
  fetched, and the repo carries no vendored dependency.
- Debugging: a sanitizer-led session instead of gdb or LLDB.
  gdb is not installed on the build machine. AddressSanitizer and
  UndefinedBehaviorSanitizer found, located, and verified the fix
  for a real heap out-of-bounds write.

## What is here

- cpp/tool.cpp: the Learn bench tool evolved. Vector add, tiled
  matmul with counted tile loads and guarded edges, and stable
  softmax, with the same math as Learn (max subtraction, guarded
  zero padding). RAII Buffer, templates, STL. It self-checks and
  prints PASS lines.
- cpp/tests.cpp: 16 assertion tests including tool.cpp behind a
  no-main guard, so the tests run the shipped functions.
- cpp/buggy.cpp: the tiled matmul variant with the bounds guard
  removed, kept as the debugging session exhibit.
- DEBUG-SESSION.md: the full debugging write-up.
- cuda/kernels.cu and cuda/build_blackbox.bat: the custom CUDA
  harness. See the status section below before reading anything
  into it: it has not been compiled or run.
- analysis.py: recomputes every throughput in
  results_torch_bench.json from size and median_ms and asserts
  agreement within 1 percent and every correct flag true.
- run_build.py: prints the benchmark table, compiles and runs the
  C++ tests when g++ is available (named skip message when not),
  runs the fixed tool under ASan and UBSan, and writes
  results_build.json.

## C++ test results

Compiled with g++ -std=c++17 -O2 (g++ 13.3.0 on the build VM),
2026-11-05:

- Tool self-check: vector_add PASS, matmul_tiled_2x2 PASS,
  matmul_tiled_3x3_tile2_edge PASS, softmax_stable_large PASS,
  tool ALL PASS.
- Test runner: 16 passed, 0 failed. Coverage: hand-computed
  vector add and matmul values (2x2 and 3x3, naive and tiled),
  tile load counts (4x4 tile 2: 64 vs 128 naive; 8x8 tile 2: 512
  vs 1024; 16x16 tile 4: 2048 vs 8192), edge sizes that are not
  multiples of the tile (3x3 tile 2, 5x5 tile 2, 5x5 tile 4, each
  matched against naive), and softmax (row sums to 1, stable form
  on inputs 1000 and above, uniform row, per-row sums, shift
  invariance).
- Python suite: 30 tests pass in total across Learn and Build
  (18 new Build tests in tests/test_build_kernels.py).

## Debugging session summary

A tiled matmul variant missing its bounds guard, run at n = 3
with tile 2, writes one double past its 72-byte heap result
buffer. AddressSanitizer reports it as a heap-buffer-overflow
WRITE at buggy.cpp line 78, 0 bytes after the region allocated at
line 45. The root cause is the missing (gi < n && gj < n) guard
at the edge tile; the fix restores the guard in the form tool.cpp
and kernels.cu carry, and the fixed tool runs clean under ASan
and UBSan, including the edge sizes. Full report, root cause,
and evidence: DEBUG-SESSION.md.

## GPU benchmarks: PyTorch native ops

Measured on the fleet GPU node (BlackBox). Device: NVIDIA
GeForce RTX 5050 Laptop GPU, compute capability 12.0, torch
2.11.0+cu128. Method: CUDA events timing, median of repeated
runs after warmup, correctness checked against a reference
before timing. Source file: results_torch_bench.json, stored
before this Build cycle and verified here by recomputation
(analysis.py, all rows within 1 percent, every correct flag
true).

These are PyTorch native ops (torch add, torch matmul, torch
softmax). They are not results from the custom kernels in
cuda/kernels.cu, and no custom kernel timing is claimed.

| Kernel (PyTorch native op) | Size | Median | Throughput | Correct |
| --- | --- | --- | --- | --- |
| vector_add_torch | n = 4194304 | 0.132 ms | 381.3 GB/s | true |
| matmul_torch | 512 by 512 | 0.0761 ms | 3529.07 GFLOP/s | true |
| matmul_torch | 1024 by 1024 | 0.3175 ms | 6764.67 GFLOP/s | true |
| softmax_rows_torch | 4096 rows, 1024 cols | 0.0395 ms | 848.71 GB/s | true |

Throughput formulas: vector add moves 3 * n * 4 bytes, matmul
performs 2 * size^3 FLOPs, softmax moves 2 * rows * cols * 4
bytes, each divided by the median time.

## Custom CUDA kernels status

Named blocker, stated plainly:

- cuda/kernels.cu is written: vector_add_kernel,
  matmul_naive_kernel, matmul_tiled_kernel, softmax_rows_kernel,
  plus a canary-buffer debug demo in its harness main. It is
  UNCOMPILED and has never run.
- Why: the fleet GPU node (BlackBox, RTX 5050) has no CUDA
  toolkit in PATH or CUDA_PATH this cycle, and torch reports
  CUDA_HOME as None there, so nvcc is not available to build it.
- A Numba route was also tried and failed on compute capability
  12.0 support.
- Consequence: there are no custom kernel timings, no custom
  kernel correctness results, and no canary demo result. The GPU
  table above is PyTorch native ops only.
- First Deep item: get a CUDA toolkit on the GPU node, compile
  kernels.cu, run its harness checks, and only then publish
  custom kernel numbers next to the PyTorch table.

## Grade

8,200 of 10,000. Deductions: custom CUDA kernels written but not
compiled or run, toolkit missing on the GPU node (minus 1,200,
the largest; this is why no custom kernel number exists and Deep
opens with the compile); debugging session sanitizer-led instead
of gdb, because gdb is not installed on the build machine
(minus 200); C++ tests use a self-contained runner instead of
GoogleTest or Catch2, a named deviation with no vendored
dependency in the repo (minus 200); GPU benchmark rows are
PyTorch native ops on one device with stored medians rounded to
4 significant figures, verified by recomputation only, not
re-measured this cycle (minus 200).
