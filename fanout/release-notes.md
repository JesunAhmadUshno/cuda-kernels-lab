# Release notes draft: cuda-kernels-lab

Status: DRAFT notes for releases that already exist. Dated 2026-11-05.
These notes summarise the existing GitHub releases. No new release is
created by this file.

Releases verified with gh release list on 2026-11-05:
v0.1, v0.5, and v1.0, at the URLs below. Every GPU number in these
notes is a PyTorch native op, labelled as such. The custom CUDA
kernels are written but were NOT compiled and did NOT run at any
release, because the fleet GPU node has no CUDA toolkit.

## v0.1, Learn: CPU reference kernels

URL: https://github.com/JesunAhmadUshno/cuda-kernels-lab/releases/tag/v0.1
Title on GitHub: v0.1 Level 1 Learn: CPU reference kernels
Published: 2026-11-05

- CPU reference kernels in Python (standard library only):
  vector add, naive and tiled matmul with counted tile loads,
  naive and stable softmax, each returning exact operation or
  load counts from its loop structure.
- Correctness: 5 of 5 checks PASS in run_learn.py, including
  the naive softmax overflow case kept as the failure exhibit
  (the stable form subtracts the row maximum first).
- Load table: tiled 64 against naive 128 at 4 by 4 tile 2,
  512 against 1024 at 8 by 8 tile 2, 2048 against 8192 at 16
  by 16 tile 4, results identical in both forms.
- Standalone C++17 bench tool (RAII Buffer, templated
  KernelRunner, STL): self check PASS, built with
  g++ -std=c++17.
- Tests: 12 unit tests in tests/test_learn_kernels.py.
- Limits carried in the release: CPU reference edition, no
  GPU ran and no GPU number is claimed (the GPU fleet node
  was unreachable in that cycle); counts only, no wall time.

## v0.5, Build: C++ tool tests, debugging session, GPU rows

URL: https://github.com/JesunAhmadUshno/cuda-kernels-lab/releases/tag/v0.5
Title on GitHub: v0.5 Level 2 Build
Published: 2026-11-05

- The Learn C++ tool evolved: vector add, tiled matmul with
  counted tile loads and guarded edges, and stable softmax,
  with a 16-test self-contained assertion runner (a named
  deviation from PROPOSAL.md: no GoogleTest or Catch2 fetched or
  vendored).
- Debugging session (level-2-build/DEBUG-SESSION.md): a tiled
  matmul variant missing its bounds guard produced a real
  AddressSanitizer heap-buffer-overflow WRITE at buggy.cpp
  line 78 at an edge size; root cause named, fix restores the
  guard, fixed tool clean under AddressSanitizer and UBSan.
  The session is sanitizer-led, not gdb-led, because gdb is
  not installed on the build machine (named deviation).
- GPU benchmarks, PyTorch native ops only, fleet GPU node
  (NVIDIA GeForce RTX 5050 Laptop GPU, compute capability
  12.0, torch 2.11.0+cu128, CUDA events, median), verified by
  recomputation within 1 percent: vector add 381.3 GB/s at
  n = 4194304 (0.132 ms), matmul 3529.07 GFLOP/s at 512
  (0.0761 ms) and 6764.67 GFLOP/s at 1024 (0.3175 ms),
  softmax rows 848.71 GB/s at 4096 by 1024 (0.0395 ms), all
  correct true.
- Custom CUDA harness level-2-build/cuda/kernels.cu written
  (vector add, naive and tiled matmul, row softmax, canary
  demo) but UNCOMPILED and never run: no CUDA toolkit on the
  GPU node, torch reports CUDA_HOME None, and a Numba route
  failed on compute capability 12.0 support.
- Tests: 18 Build Python tests, 30 in total with Learn; 16
  C++ tests.

## v1.0, Deep: profiler study, atomics and move benchmarks

URL: https://github.com/JesunAhmadUshno/cuda-kernels-lab/releases/tag/v1.0
Title on GitHub: v1.0: Learn, Build, compressed Deep
Published: 2026-11-05

- Compressed Deep under the December Compression Protocol:
  the single hardest capability (profiler and fused-kernel
  work, plus the SPEC C++ Deep track), with the toolkit
  substitution named in PROPOSAL.md (Level 3 Deep spec), never silent. The rest
  of the v2 Deep scope is the v1.1 backlog in REPORT.md.
- Profiler study on PyTorch routes only (same device and
  torch version, CUDA events median, torch.profiler op
  counts), 9 rows verified by recomputation within 1
  percent, every correct flag true: matmul native op
  5422.94 GFLOP/s at 512 (0.0495 ms), 7642.29 at 1024
  (0.281 ms), 5970.42 at 2048 (2.8775 ms); softmax native
  fused op 604.58 GB/s at 4096 by 1024 (0.0555 ms) against
  155.13 GB/s (0.2163 ms) for the unfused manual sequence at
  the same size, a factor of 3.9, with the same pattern at
  2048 by 2048 (562.05 against 156.29) and 8192 by 512
  (563.94 against 155.06).
- Stage profile at 4096 by 1024: amax 0.0218 ms, exp
  0.1464 ms, sum 0.0222 ms, div 0.0908 ms, unfused full
  sequence 0.2147 ms, native fused softmax 0.0572 ms.
  Profiler op counts per 10 reps: one fused op per call for
  the native route, five ops per call for the unfused route.
- C++ Deep track: 15 new assertion tests (31 C++ in total),
  Deep tool and Build tool clean under AddressSanitizer and
  UBSan, atomic counter exactly 1000000 under relaxed
  (25.8349 ms) and seq_cst (20.6221 ms) ordering with 4
  threads by 250000, producer-consumer handoff 200 of 200
  under acquire-release (15.4334 ms), and a move benchmark
  with exact counts (50 copies in 83.5614 ms, 50 moves in
  3.52111 ms, 200000-double buffer, checksum 150.0).
- Custom fused softmax kernel level-3-deep/cuda/
  fused_softmax.cu written only, with its blocker analysis
  and expected profile argument in cuda/ANALYSIS.md. It was
  NOT compiled and did NOT run. torch.compile was attempted
  and failed on the node (TritonMissing), and CUPTI
  CUDA-activity rows were unavailable in the committed runs.
- Tests: 16 Deep Python tests, 46 in total with Learn and
  Build.
