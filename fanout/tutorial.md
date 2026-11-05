# Tutorial: P04 cuda-kernels-lab, course module

Module: P04-cuda-kernels-lab
By Jesun Ahmad Ushno
Date: 2026-11-05
License for this tutorial: CC BY 4.0. Code in the repo: Apache-2.0.

## What you will do

You will clone the cuda-kernels-lab repo, run all three levels,
run the Python test suite for each level, build and run the C++
tools exactly as CI does, and read your own results against the
recorded v1.0 numbers. Every command below exists in the repo as
shipped at v1.0 and was run by the author of this module on
2026-11-05 to verify it. The Python levels need Python 3.11 or
3.12 and use the standard library only. The C++ steps need g++
with C++17 support. No GPU, no CUDA toolkit, and no credential
are needed for any command in this module.

Read this before you start: every GPU benchmark number recorded
in this repo is a PyTorch native op, labelled as such. The
custom CUDA kernels are written but were NOT compiled and did
NOT run, because the fleet GPU node has no CUDA toolkit (no
nvcc on that node). This module does not compile them either.
It verifies the stored GPU rows by recomputation, which is what
the level runners do.

## Step 0: Clone and enter the repo

```text
git clone https://github.com/JesunAhmadUshno/cuda-kernels-lab.git
cd cuda-kernels-lab
```

## Step 1: Level 1, Learn, CPU reference kernels

Read level-1-learn/README.md first. It explains vector add as
the one thread per element pattern, naive against tiled matmul
as a memory traffic argument, and stable softmax (subtract the
row maximum first, so exp cannot overflow).

Run the Learn runner:

```text
python level-1-learn/run_learn.py
```

Run the Learn tests:

```text
python -m unittest discover -s tests -p "test_learn*.py" -v
```

Build and run the C++ bench tool:

```text
g++ -std=c++17 -O2 -Wall -Wextra level-1-learn/cpp/bench_tool.cpp -o /tmp/bench_tool
/tmp/bench_tool
```

Read your results:

- Expected runner lines: 5 correctness checks, all PASS, then
  the load table: n=4 tile=2 naive_loads=128 tiled_loads=64,
  n=8 tile=2 naive_loads=1024 tiled_loads=512, n=16 tile=4
  naive_loads=8192 tiled_loads=2048, with identical=True on
  every row, and all_pass: True.
- Expected tests: 12 tests, all passing.
- Expected C++ output: vector_add PASS, matmul_tiled_2x2 PASS,
  bench_tool ALL PASS.
- The runner writes level-1-learn/results_learn.json. It holds
  counts only, no wall time, by design.

## Step 2: Level 2, Build, C++ tool, debugging, GPU rows

Read level-2-build/README.md, level-2-build/BUILD-SPEC.md, and
level-2-build/DEBUG-SESSION.md. The Build tool carries vector
add, tiled matmul with guarded edges, and stable softmax. Its
16 tests use a self-contained assertion runner. The debugging
session is sanitizer-led: a variant missing its bounds guard
produced a real AddressSanitizer heap-buffer-overflow WRITE,
and the fixed tool runs clean.

Run the Build runner:

```text
python level-2-build/run_build.py
```

Run the Build tests:

```text
python -m unittest discover -s tests -p "test_build*.py" -v
```

Build and run the C++ Build tests, then the fixed tool under
sanitizers:

```text
g++ -std=c++17 -O2 -Wall -Wextra level-2-build/cpp/tests.cpp -o /tmp/cpp_tests
/tmp/cpp_tests
g++ -std=c++17 -fsanitize=address,undefined -g level-2-build/cpp/tool.cpp -o /tmp/tool_asan
/tmp/tool_asan
```

Read your results:

- Expected runner table: 4 GPU rows verified by recomputation
  within 1 percent, every correct flag true. The rows are
  PyTorch native ops measured on the fleet GPU node (NVIDIA
  GeForce RTX 5050 Laptop GPU, torch 2.11.0+cu128): vector add
  381.3 GB/s, matmul 3529.07 GFLOP/s at 512 and 6764.67
  GFLOP/s at 1024, softmax rows 848.71 GB/s at 4096 by 1024.
  Your machine does not re-measure them; the runner recomputes
  the throughputs from the stored sizes and medians.
- Expected C++ tests: 16 passed, 0 failed, tests ALL PASS.
- Expected sanitizer run: tool ALL PASS, with no sanitizer
  report (the runner also reports the fixed tool as CLEAN).
- Expected Python tests: 18 Build tests, all passing.
- The runner writes level-2-build/results_build.json.

## Step 3: Level 3, Deep, profiler study and C++ Deep track

Read level-3-deep/README.md, level-3-deep/DEEP-SPEC.md, and
level-3-deep/cuda/ANALYSIS.md. This is a compressed Deep. The
profiler study runs on PyTorch routes only, every row labelled
by route. The custom fused softmax kernel is a written
component only, with its blocker analysis in ANALYSIS.md.

Run the Deep runner:

```text
python level-3-deep/run_deep.py
```

Run the Deep tests:

```text
python -m unittest discover -s tests -p "test_deep*.py" -v
```

Build and run the C++ Deep tests, then the Deep tool under
sanitizers:

```text
g++ -std=c++17 -O2 -Wall -Wextra level-3-deep/cpp/tests_deep.cpp -o /tmp/tests_deep -pthread
/tmp/tests_deep
g++ -std=c++17 -fsanitize=address,undefined -g level-3-deep/cpp/deep_tool.cpp -o /tmp/deep_asan -pthread
/tmp/deep_asan
```

Read your results:

- Expected runner table: 9 profiler rows verified by
  recomputation. Matmul (PyTorch native op): 5422.94 GFLOP/s
  at 512, 7642.29 GFLOP/s at 1024, 5970.42 GFLOP/s at 2048.
  Softmax at 4096 by 1024: native fused op 604.58 GB/s
  (0.0555 ms) against 155.13 GB/s (0.2163 ms) for the unfused
  manual sequence, a factor of 3.9. The other two sizes hold
  the same pattern (562.05 against 156.29, and 563.94 against
  155.06).
- Expected C++ Deep tests: 15 passed, 0 failed. The runner
  also re-runs the 16 Build C++ tests as a regression check.
- Expected sanitizer runs: Deep tool and Build tool both
  CLEAN under AddressSanitizer and UBSan.
- Expected Deep tool counts, exact on every run: atomic
  totals of exactly 1000000 under relaxed and under seq_cst
  ordering (4 threads by 250000), handoff 200 of 200 under
  acquire-release, and move counts of exactly 50 copies and
  50 moves for a 200000-double buffer, checksum 150.0. The
  wall times next to those counts are your machine's measured
  values and will differ from the recorded run (recorded:
  copies 83.5614 ms, moves 3.52111 ms). Counts are the claim;
  times vary.
- Expected Python tests: 16 Deep tests, all passing.
- The runner writes level-3-deep/results_deep.json with your
  run's C++ wall times. The profiler rows inside it stay the
  committed BlackBox rows; only the C++ timings are local.

## Step 4: Run the full test suite

```text
python -m unittest discover -s tests -v
python tests/test_smoke.py
```

Expected: 46 tests in total (12 Learn, 18 Build, 16 Deep),
all passing, and the scaffold smoke test PASS.

## Step 5: Check your understanding

1. Tiling leaves multiplies and adds unchanged. What does it
   change, and by how much at 16 by 16 with tile 4? (Answer:
   it changes memory traffic. Tiled loads are 2048 element
   slots against 8192 naive loads, a 4x reduction, because
   each loaded element is reused across a tile row or column.)
2. At 4096 by 1024 the fused native softmax is 3.9 times
   faster than the unfused manual sequence. Which two stages
   dominate the unfused cost in the stage profile, and why?
   (Answer: exp at 0.1464 ms and div at 0.0908 ms. Each stage
   materialises a full intermediate in global memory. The
   fused op writes no intermediate to global memory.)
3. A classmate says the 7642.29 GFLOP/s matmul number proves
   the project's own CUDA kernel is fast. What is wrong with
   that sentence? (Answer: the number is a PyTorch native op
   measured on the fleet GPU node. The custom CUDA kernels in
   this repo were written but never compiled or run, because
   the node has no CUDA toolkit, so no custom kernel speed is
   known or claimed.)

## Limits to keep with the numbers

- Every GPU row is a PyTorch native op (or a manual sequence
  of PyTorch native ops for the unfused rows). No custom
  kernel result exists in v1.0.
- The custom CUDA kernels need a CUDA toolkit to compile.
  Installing one on the fleet GPU node is v1.1 backlog item 1
  in ROADMAP.md at the repo root.
- GPU numbers are one laptop GPU in committed runs, with
  named clock variance across runs. Ratios inside one run are
  the stable finding.
- C++ wall times in the Deep tool vary by machine. The exact
  counts and totals are the portable claim.
