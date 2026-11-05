# cuda-kernels-lab

CUDA kernels from vector add to a fused attention kernel, plus a standalone C++ benchmark tool.

Status: Level 1 Learn DONE, Level 2 Build DONE, and Level 3
Deep DONE, 2026-11-05. Release v1.0.
Deep headline numbers: a profiler study on the fleet GPU node
(NVIDIA GeForce RTX 5050 Laptop GPU, torch 2.11.0+cu128, CUDA
events, median) with 9 measured rows verified by recomputation
within 1 percent: matmul (native op) 7642.29 GFLOP/s at 1024,
softmax native fused op 604.58 GB/s at 4096 by 1024 against
155.13 GB/s for the unfused manual sequence at the same size
(a factor of 3.9), with a stage profile and torch.profiler op
counts. The C++ Deep track: 15 new C++ tests (31 C++ total
with Build), both tools clean under ASan and UBSan, an
atomics counter reaching exactly 1000000 under relaxed and
seq_cst ordering, a producer-consumer handoff at 200 of 200
under acquire-release, and a move benchmark with exact counts
(50 copies in 83.5614 ms, 50 moves in 3.52111 ms for a
200000-double buffer). 46 Python tests pass.
Named kernel blocker, unchanged: the custom CUDA kernels
(level-2-build/cuda/kernels.cu and the Deep fused kernel in
level-3-deep/cuda/fused_softmax.cu) are written but UNCOMPILED
and have never run, because the fleet GPU node has no CUDA
toolkit (nvcc not on PATH, torch reports CUDA_HOME None).
Compiling and running them is v1.1 backlog item 1 in
ROADMAP.md. Level grades, of 10,000, deductions named in each
level README: Learn 8,300, Build 8,200, Deep 7,900.
Project: P04. Fleet node: BlackBox (GPU). Wave: Wave 2.

## Start here

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.
5. Read ROADMAP.md for the v1.1 backlog beyond v1.0.

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
| Deep C++ tests (level-3-deep/cpp/tests_deep.cpp) | 15 pass, 0 failed (31 C++ total) |
| Deep Python tests (tests/test_deep_kernels.py) | 16 pass (46 total) |
| Deep sanitizers | Deep tool and Build tool CLEAN under ASan and UBSan |
| Deep profiler rows verified (PyTorch routes, RTX 5050) | 9 of 9 within 1 percent, all correct true, no custom rows |
| Deep atomics | relaxed and seq_cst totals both exactly 1000000; handoff 200 of 200 ordered |
| Deep move bench | 50 copies 83.5614 ms, 50 moves 3.52111 ms, counts exact |

## Fan-out

Written fan-out set for v1.0, dated 2026-11-05, in fanout/:
blog post (fanout/blog-post.md), step-by-step tutorial
(fanout/tutorial.md), website page copy (fanout/website-page.md),
LinkedIn post draft, not posted (fanout/linkedin-post.md), paper
candidate brief for the fused against unfused profiler note,
nothing submitted (fanout/paper-candidate.md), and draft release
notes for v0.1, v0.5, and v1.0 (fanout/release-notes.md). Channel
outcomes and per-piece publicity grades are logged in
PUBLICITY.md at the repo root.

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## License

Code: Apache-2.0 (see LICENSE). Tutorials: CC BY 4.0. Ruled in D-V3-11, 2026-11-04.
