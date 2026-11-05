# cuda-kernels-lab

CUDA kernels from vector add to a fused attention kernel, plus a standalone C++ benchmark tool.

Status: Level 1 Learn DONE and Level 2 Build DONE, 2026-11-05.
Build headline numbers: 16 C++ tests pass, 30 Python tests pass,
the debugging session captured a real AddressSanitizer
heap-buffer-overflow WRITE and the fixed tool runs clean under
ASan and UBSan, and 4 stored GPU benchmark rows verify by
recomputation within 1 percent. The GPU rows are PyTorch native
ops on the NVIDIA GeForce RTX 5050 Laptop GPU (median, CUDA
events): vector add 381.3 GB/s at n = 4194304, matmul 6764.67
GFLOP/s at 1024, softmax rows 848.71 GB/s at 4096 by 1024.
Named kernel blocker: the custom CUDA kernels in
level-2-build/cuda/kernels.cu are written but UNCOMPILED and
have never run, because the fleet GPU node has no CUDA toolkit
in PATH or CUDA_PATH this cycle (torch reports CUDA_HOME None)
and a Numba route failed on compute capability 12.0 support.
Compiling and running kernels.cu is the first Deep item. Level
3 is scaffold only.
Project: P04. Fleet node: BlackBox (GPU). Wave: Wave 2.

## Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.

## Level map

- Level 1, Learn: the core idea, built small and clean, with a teaching path.
- Level 2, Build: a real working system other people can run.
- Level 3, Deep: benchmarks, optimization, evaluation, hard cases.

## Benchmarks

CPU reference counts only, Level 1 Learn, 2026-11-05. Method:
exact operation and load counts from the loop structure in
level-1-learn/kernels.py, run by level-1-learn/run_learn.py. No
wall time and no GPU number is claimed. Full table and teaching
notes: level-1-learn/README.md.

| Test | Result |
| --- | --- |
| Correctness checks (run_learn.py) | 5 of 5 PASS |
| Unit tests (tests/test_learn_kernels.py) | 12 pass |
| Matmul loads, 4x4 tile 2 | naive 128, tiled 64 |
| Matmul loads, 8x8 tile 2 | naive 1024, tiled 512 |
| Matmul loads, 16x16 tile 4 | naive 8192, tiled 2048 |
| C++ bench tool self check (g++ -std=c++17) | vector_add PASS, matmul_tiled_2x2 PASS, ALL PASS |
| Build C++ tests (level-2-build/cpp/tests.cpp) | 16 pass, 0 failed |
| Build Python tests (tests/test_build_kernels.py) | 18 pass (30 total with Learn) |
| Build ASan session | heap-buffer-overflow WRITE captured at buggy.cpp:78, fixed tool clean under ASan and UBSan |
| GPU rows verified (PyTorch native ops, RTX 5050) | 4 of 4 within 1 percent, all correct true |

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
