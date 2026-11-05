# Level 2 Build Spec: cuda-kernels-lab (P04)

Status: written before the Build code, 2026-11-05.

## Scope from SPEC.md

Build closes when the system runs end to end, tests pass, and one
benchmark or result table with method is in the README. The Added
track names: tests on the standalone C++ tool (SPEC text says
GoogleTest or Catch2), and one debugging session written up from a
real crash (SPEC text says gdb or LLDB).

## Decisions taken in this Build, with reasons

1. Test framework: a self-contained assertion runner in
   cpp/tests.cpp, no external framework. Catch2 and GoogleTest were
   not fetched and are not vendored in this repo. Adding either
   would introduce a vendored dependency or a network fetch for a
   test harness that needs about a dozen assertions. The runner
   prints one PASS line per test plus a total, and exits non-zero
   on any failure, so CI treats it exactly like a framework run.
   This is a named deviation from the SPEC wording, not a silent
   one: the SPEC outcome (tests on the tool, run in CI) is met.
2. Debugging session: sanitizer-led, not gdb-led. gdb is not
   installed on the build machine (verified: no gdb binary in
   PATH, 2026-11-05). The session uses AddressSanitizer plus
   UndefinedBehaviorSanitizer on a variant with a genuine heap
   out-of-bounds write, captures the real report, fixes the root
   cause, and re-runs clean. That meets the SPEC outcome (one
   debugging session from a real crash, written up).
3. GPU numbers: only the pre-existing measured file
   results_torch_bench.json is used. Those rows are PyTorch native
   ops on the fleet GPU node (BlackBox, NVIDIA GeForce RTX 5050
   Laptop GPU, compute capability 12.0, torch 2.11.0+cu128, CUDA
   events, median). They are labelled native ops everywhere, never
   as custom kernel results.
4. Custom CUDA kernels: cuda/kernels.cu is a written harness
   (vector_add_kernel, matmul_naive_kernel, matmul_tiled_kernel,
   softmax_rows_kernel, plus a canary debug demo) that has NOT
   been compiled or run. The fleet GPU node has no CUDA toolkit in
   PATH or CUDA_PATH this cycle and torch reports CUDA_HOME None;
   a Numba route failed on compute capability 12.0 support.
   Compiling and running kernels.cu is the first Deep item. This
   blocker is named in every README that mentions the harness.

## Deliverables

- cpp/tool.cpp: the Learn bench tool evolved. Same math as Learn:
  vector add, tiled matmul (tile load counts, edge guards), stable
  softmax (max subtraction, per row). RAII Buffer, templates, STL.
  Its own main prints self-check PASS lines.
- cpp/tests.cpp: assertion test runner including tool.cpp behind a
  no-main guard. Covers hand-computed correctness, tile load
  counts, edge sizes that are not multiples of the tile, and
  softmax row sums. Compiles with g++ -std=c++17 -O2.
- cpp/buggy.cpp: tiled matmul variant whose tile loop is missing
  the bounds guard, writing past a heap allocation at an edge size.
- DEBUG-SESSION.md: symptom, real AddressSanitizer report excerpt
  with stack frames and the SUMMARY line, root cause, fix, and the
  fixed run passing ASan and UBSan clean.
- analysis.py: recomputes GB/s and GFLOP/s from the stored size
  and median_ms values in results_torch_bench.json and asserts
  agreement within 1 percent and every correct flag true.
- run_build.py: prints the benchmark table, compiles and runs the
  C++ tests when g++ is available (named skip message when not),
  runs the fixed tool under sanitizers, and writes
  results_build.json with the C++ tests passed count, the ASan
  session status, and the GPU rows verified count.
- tests/test_build_kernels.py: at least 12 Python unit tests.
- README updates: this level README, the root README (Build DONE
  with headline numbers and the named kernel blocker), and the
  DOSSIER.md Present section.

## Acceptance criteria

- python3 -m unittest discover -s tests passes in full.
- The C++ test runner compiles and every test prints PASS.
- tool.cpp built with -fsanitize=address,undefined runs clean.
- buggy.cpp built with sanitizers produces a real heap-buffer-overflow
  WRITE report whose SUMMARY line is quoted verbatim in
  DEBUG-SESSION.md.
- Every stored GPU throughput recomputes within 1 percent.
- Em dash count is 0 in every touched file. No invented numbers:
  the only GPU numbers are the stored measured rows.
