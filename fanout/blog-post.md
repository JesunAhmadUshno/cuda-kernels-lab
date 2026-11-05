# Down To The Metal, Measured Honestly: Building cuda-kernels-lab in the Open

By Jesun Ahmad Ushno
Date: 2026-11-05
Status: DRAFT, written fan-out for P04 cuda-kernels-lab v1.0. Not posted.

## The problem

A GPU kernel is fast or slow for reasons you can count: how much
global memory traffic it moves, how many kernel launches it pays
for, and whether it writes intermediates back to global memory.
Vendor samples show the fast version. They rarely show the count
that explains why it is fast, the unfused version it beats, or the
blocker that stopped the custom kernel from running at all.

This project, P04 in Roadmap v3, builds the kernel stack in the
open, one level at a time: CPU reference kernels with exact load
counts, a standalone C++ tool with tests and a real sanitizer
debugging session, GPU benchmarks measured on the fleet GPU node,
and a profiler study that separates a fused op from an unfused
sequence stage by stage. One honesty rule runs through every
level and through this post: every GPU benchmark row in this repo
is a PyTorch native op, labelled as such everywhere. The custom
CUDA kernels are written but were NOT compiled and did NOT run,
because the fleet GPU node has no CUDA toolkit (no nvcc on that
node). No custom kernel timing or correctness result is claimed
anywhere in this post or in this repo.

Repository: https://github.com/JesunAhmadUshno/cuda-kernels-lab

## Past, present, future (the dossier, condensed)

Past: vendor CUDA samples and PyTorch kernels already exist. The
prior art and the problem history are recorded in DOSSIER.md and
NOVELTY.md in the repo. The novelty claim there stays marked
PROPOSED. This post makes no novelty claim beyond the measured
work below.

Present: v1.0 closes three levels on 2026-11-05. Learn built the
kernel structures on the CPU with exact operation and load counts,
so the tiling argument can be checked by eye. Build evolved the
standalone C++ tool, tested it, debugged a real heap overflow
with sanitizers, and recorded GPU benchmarks for PyTorch native
ops on the fleet GPU node. Compressed Deep ran a profiler study
on PyTorch routes only, evolved the C++ tool with an atomics and
memory ordering example and a move semantics benchmark, and wrote
the custom fused softmax kernel with its compile blocker analysis.

Future: the rest of the v2 Deep scope is named in ROADMAP.md in
the repo as the v1.1 backlog. Item 1 is compiling and running the
custom CUDA kernels once a CUDA toolkit exists on the fleet GPU
node. It also includes the fused attention variant, the
GoogleTest or Catch2 migration, CUPTI and compiled route
profiler rows when the node supports them, a second device, and
a weakly ordered platform run. None of that is done, and none of
it is claimed here.

## Main results claim

At v1.0: Learn passed 12 Python tests with 5 of 5 correctness checks on CPU reference kernels, with tiled matmul loading 64 element slots against 128 naive loads at 4 by 4 with tile 2, 512 against 1024 at 8 by 8 with tile 2, and 2048 against 8192 at 16 by 16 with tile 4, and no GPU number claimed at that level; Build passed 18 Python tests (30 in total) and 16 C++ tests, with PyTorch native op benchmarks measured on an NVIDIA GeForce RTX 5050 Laptop GPU with torch 2.11.0+cu128 using CUDA-event timing (median) of vector add at 381.3 GB/s, matmul at 6764.67 GFLOP/s at 1024, and softmax rows at 848.71 GB/s at 4096 by 1024, all 4 stored rows recomputing within 1 percent; compressed Deep passed 16 Python tests (46 in total) and 15 new C++ tests (31 C++ in total), with a profiler study on the same device and torch version using CUDA-event timing (median) and torch.profiler op counts in which matmul as a PyTorch native op reached 5422.94 GFLOP/s at 512, 7642.29 GFLOP/s at 1024, and 5970.42 GFLOP/s at 2048, and the softmax PyTorch native fused op reached 604.58 GB/s at 4096 by 1024 against 155.13 GB/s for an unfused manual sequence of separate PyTorch ops at the same size (median 0.0555 ms against 0.2163 ms, a factor of 3.9), all 9 profiler rows correctness-checked and recomputing within 1 percent, the C++ Deep tool running clean under AddressSanitizer and UBSan, an atomic counter reaching exactly 1000000 under both relaxed and sequentially consistent ordering (4 threads by 250000), a producer-consumer handoff passing 200 of 200 trials under acquire-release ordering, and a move-semantics benchmark recording exact counts of 50 copies in 83.5614 ms and 50 moves in 3.52111 ms for a 200000-double buffer; the custom CUDA kernels, including the fused softmax kernel in level-3-deep/cuda/fused_softmax.cu, were written but not compiled and not run, because the fleet GPU node has no CUDA toolkit this cycle; level grades are Learn 8300 of 10000, Build 8200 of 10000, and Deep 7900 of 10000, with the uncompiled custom kernel as the largest Deep deduction.

Every number in that claim comes from a run recorded in the
level READMEs and results files in the repo. Read the limits
section below before quoting any of it. The Build GPU rows and
the Deep profiler rows come from different runs on the same
device, so their matmul and softmax values differ; each table
in this post keeps its own run separate and labelled.

## Results by level

### Level 1, Learn (release v0.1)

CPU reference edition. BlackBox, the GPU fleet node, was
unreachable in that cycle, so no CUDA GPU ran and no GPU timing
is claimed at this level. The Python kernels (vector add, naive
and tiled matmul, naive and stable softmax) return exact
operation and load counts from their loop structure.

| Size | Tile | Naive element loads | Tiled tile loads | Results identical |
| --- | --- | --- | --- | --- |
| 4 | 2 | 128 | 64 | yes |
| 8 | 2 | 1024 | 512 | yes |
| 16 | 4 | 8192 | 2048 | yes |

Tiling does not change the arithmetic. Multiplies and adds are
identical in both forms. It changes the memory traffic: at 16
by 16 with tile 4 the tiled form loads 2048 element slots
against 8192 naive loads, a 4x reduction, because each loaded
element is reused across a whole tile row or column. That reuse
is the entire reason tiled kernels are faster on a GPU, and the
count version of it is what this level proves. The C++ bench
tool self check passes (vector add and tiled matmul), built
with g++ -std=c++17. Tests: 12. Grade: 8300 of 10000.

### Level 2, Build (release v0.5)

The standalone C++ tool now carries vector add, tiled matmul
with guarded edges, and stable softmax, with a 16-test
self-contained assertion runner. The debugging session is real:
a tiled matmul variant missing its bounds guard wrote past its
heap buffer at an edge size, AddressSanitizer reported a
heap-buffer-overflow WRITE at buggy.cpp line 78, and the fixed
tool runs clean under AddressSanitizer and UBSan. The full
write-up is level-2-build/DEBUG-SESSION.md.

GPU benchmarks below are PyTorch native ops only, measured on
the fleet GPU node (NVIDIA GeForce RTX 5050 Laptop GPU,
compute capability 12.0, torch 2.11.0+cu128, CUDA events,
median). Source: level-2-build/results_torch_bench.json,
verified by recomputation within 1 percent.

| Kernel (PyTorch native op) | Size | Median | Throughput | Correct |
| --- | --- | --- | --- | --- |
| vector_add_torch | n = 4194304 | 0.132 ms | 381.3 GB/s | true |
| matmul_torch | 512 by 512 | 0.0761 ms | 3529.07 GFLOP/s | true |
| matmul_torch | 1024 by 1024 | 0.3175 ms | 6764.67 GFLOP/s | true |
| softmax_rows_torch | 4096 rows, 1024 cols | 0.0395 ms | 848.71 GB/s | true |

Tests: 18 Build Python tests, 30 in total; 16 C++ tests.
Grade: 8200 of 10000, largest deduction minus 1200 for the
custom CUDA kernels written but not compiled or run.

### Level 3, Deep, compressed (release v1.0)

Profiler study on PyTorch routes only, same device and torch
version, CUDA events (median), verified by recomputation
within 1 percent (9 of 9 rows, every correct flag true).

Matmul, PyTorch native op:

| Kernel | Route | Size | Median | Throughput | Correct |
| --- | --- | --- | --- | --- | --- |
| matmul_torch | native op | 512 | 0.0495 ms | 5422.94 GFLOP/s | true |
| matmul_torch | native op | 1024 | 0.281 ms | 7642.29 GFLOP/s | true |
| matmul_torch | native op | 2048 | 2.8775 ms | 5970.42 GFLOP/s | true |

Softmax, PyTorch native fused op against an unfused manual
sequence of separate PyTorch ops (amax, sub, exp, sum, div).
Both routes use one traffic model (2 times rows times cols
times 4 bytes) so they compare on one axis.

| Kernel | Route | Size | Median | Throughput | Correct |
| --- | --- | --- | --- | --- | --- |
| softmax_rows_torch | native op | 4096 by 1024 | 0.0555 ms | 604.58 GB/s | true |
| softmax_rows_unfused_manual | unfused manual (native ops) | 4096 by 1024 | 0.2163 ms | 155.13 GB/s | true |
| softmax_rows_torch | native op | 2048 by 2048 | 0.0597 ms | 562.05 GB/s | true |
| softmax_rows_unfused_manual | unfused manual (native ops) | 2048 by 2048 | 0.2147 ms | 156.29 GB/s | true |
| softmax_rows_torch | native op | 8192 by 512 | 0.0595 ms | 563.94 GB/s | true |
| softmax_rows_unfused_manual | unfused manual (native ops) | 8192 by 512 | 0.2164 ms | 155.06 GB/s | true |

Stage profile at 4096 by 1024 (CUDA events): amax 0.0218 ms,
exp 0.1464 ms, sum 0.0222 ms, div 0.0908 ms, unfused full
sequence 0.2147 ms, native fused softmax 0.0572 ms. The exp
and div stages dominate the unfused cost because each one
materialises a full intermediate in global memory. The fused
op writes no intermediate to global memory. Profiler op
counts per 10 reps confirm the shape: the native softmax is
one fused op per call (aten::softmax 10, aten::_softmax 10),
the unfused sequence is five ops per call (aten::amax,
aten::sub, aten::exp, aten::sum, aten::div, each 10).

C++ Deep track (g++ -std=c++17, Muse VM): both tools clean
under AddressSanitizer and UBSan. Atomic counter, 4 threads
by 250000: relaxed total exactly 1000000 in 25.8349 ms,
seq_cst total exactly 1000000 in 20.6221 ms; no ordering
winner is claimed, because the wall times overlap across runs
on x86. Producer-consumer handoff: 200 of 200 under
acquire-release in 15.4334 ms, and the relaxed control also
200 of 200 in 16.495 ms, which is expected on x86 and is
named as no guarantee. Move benchmark, LargeBuffer with
200000 doubles, 50 reps: 50 copies in 83.5614 ms, 50 moves
in 3.52111 ms, counts exact, checksum 150.0, about 24 times
faster for moves on this run.

Tests: 16 Deep Python tests, 46 in total; 15 Deep C++ tests,
31 C++ in total. Grade: 7900 of 10000, largest deduction
minus 1000 for the custom kernel uncompiled.

## Honest limits

- Every GPU number in this repo is a PyTorch native op (or,
  for the unfused rows, a manual sequence of PyTorch native
  ops). No GPU number comes from a custom kernel written in
  this repo.
- The custom CUDA kernels (level-2-build/cuda/kernels.cu and
  level-3-deep/cuda/fused_softmax.cu) are written but NOT
  compiled and have never run. The fleet GPU node has no CUDA
  toolkit: nvcc is not on PATH and torch reports CUDA_HOME as
  None there. A Numba route also failed on compute capability
  12.0 support at Build.
- torch.compile was attempted and failed on that node
  (TritonMissing in the committed profiler run), so no
  compiled rows exist.
- CUPTI CUDA-activity rows were unavailable in the committed
  profiler runs, so the profiler evidence is torch.profiler
  CPU-activity op counts plus CUDA-event timing tables,
  recorded as such.
- All GPU numbers are one laptop GPU (RTX 5050) and the Deep
  README names clock variance across runs this cycle. Only
  the committed runs are quoted, and the fused against
  unfused ratio inside one run is the stable finding.
- Learn is a CPU reference edition with exact counts and no
  wall time, by design. Counts prove the traffic argument,
  not GPU speed.
- The C++ tests use a self-contained assertion runner, not
  GoogleTest or Catch2, and the Build debugging session is
  sanitizer-led, not gdb-led, because gdb is not installed on
  the build machine. Both are named deviations from SPEC.md.
- The atomics example runs on x86, whose strong memory model
  masks ordering differences. The relaxed handoff passing
  there is not evidence it is correct on a weakly ordered
  platform.
- Compressed Deep is not the full v2 Deep scope. The
  remainder is the named v1.1 backlog in ROADMAP.md.

## Where to go next

- Tutorial: fanout/tutorial.md in the repo, a step-by-step
  module that runs all three levels, the C++ tools, and the
  full test suite.
- Releases: v0.1 (Learn), v0.5 (Build), v1.0 (Learn, Build,
  compressed Deep) on the GitHub repo linked above.
- Book chapter: "Down To The Metal" per SPEC.md, drafted as
  part of the fan-out set.
- License: code Apache-2.0, tutorials CC BY 4.0.
