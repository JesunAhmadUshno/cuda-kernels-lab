# Paper candidate brief: fused against unfused softmax profiler note

Project: P04 cuda-kernels-lab
Status: CANDIDATE, brief written 2026-11-05
Submission note: submission is Jesun's tap. Nothing is submitted by
this brief, and no venue has been contacted. Do NOT edit the
publications tracker from this file; the coordinator updates it.

## Working title

Fused against unfused softmax on one laptop GPU: a profiler study
of PyTorch native routes, with exact-count kernel teaching results
and a C++ atomics and move benchmark

## Abstract draft

Based only on the measured v1.0 results recorded in the repo.
Every GPU number below is a PyTorch native op and is stated as
such. No custom CUDA kernel was compiled or run in this work.

GPU kernel performance is usually reported as a single headline
number with no route label and no unfused baseline. This work
presents an open kernel lab and measures each route on the same
device with the same timing method. At Learn level, CPU
reference kernels with exact load counts showed tiled matmul
loading 64 element slots against 128 naive loads at 4 by 4 with
tile 2, 512 against 1024 at 8 by 8 with tile 2, and 2048
against 8192 at 16 by 16 with tile 4, with 5 of 5 correctness
checks and 12 unit tests. At Build level, PyTorch native op
benchmarks on an NVIDIA GeForce RTX 5050 Laptop GPU (compute
capability 12.0, torch 2.11.0+cu128, CUDA events, median)
recorded vector add at 381.3 GB/s, matmul at 3529.07 GFLOP/s
at 512 and 6764.67 GFLOP/s at 1024, and softmax rows at 848.71
GB/s at 4096 by 1024, all 4 rows recomputing within 1 percent,
alongside a C++ tool with 16 tests and a real AddressSanitizer
heap-buffer-overflow finding and fix. At Deep level, a profiler
study on the same device recorded matmul as a PyTorch native
op at 5422.94 GFLOP/s at 512, 7642.29 GFLOP/s at 1024, and
5970.42 GFLOP/s at 2048, and the softmax PyTorch native fused
op at 604.58 GB/s against 155.13 GB/s for an unfused manual
sequence of separate PyTorch ops at 4096 by 1024 (median
0.0555 ms against 0.2163 ms, a factor of 3.9), with the same
pattern at 2048 by 2048 (562.05 against 156.29 GB/s) and 8192
by 512 (563.94 against 155.06 GB/s), all 9 rows recomputing
within 1 percent. A stage profile at 4096 by 1024 attributes
the unfused cost mainly to the exp stage (0.1464 ms) and the
div stage (0.0908 ms), each materialising a full intermediate
in global memory, and torch.profiler op counts show one fused
op per call against five ops per call for the unfused
sequence. The C++ Deep track recorded an atomic counter exact
at 1000000 under relaxed and sequentially consistent ordering
(4 threads by 250000), a producer-consumer handoff at 200 of
200 under acquire-release ordering, and a move benchmark with
exact counts of 50 copies in 83.5614 ms and 50 moves in
3.52111 ms for a 200000-double buffer, with 46 Python tests
and 31 C++ tests passing in total. The custom fused softmax
kernel written for this study could not be compiled, because
the GPU node has no CUDA toolkit; the paper claims the route
labelled profiler comparison and the count results, not any
custom kernel performance.

## Method outline

1. Exact counts at Learn: operation and load counts computed
   from the loop structure in level-1-learn/kernels.py, run by
   level-1-learn/run_learn.py, deterministic, no wall time.
2. Build GPU rows: stored in level-2-build/
   results_torch_bench.json from the fleet GPU node and
   verified by level-2-build/analysis.py, which recomputes
   every throughput from size and median_ms and asserts
   agreement within 1 percent and every correct flag true.
   Throughput formulas: vector add moves 3 times n times 4
   bytes, matmul performs 2 times size cubed FLOPs, softmax
   moves 2 times rows times cols times 4 bytes.
3. Deep profiler rows: produced by level-3-deep/
   profiler_bench.py on the fleet GPU node (CUDA events,
   median after warmup, correctness checked before timing)
   and verified by level-3-deep/analysis_deep.py on the same
   recomputation rule, plus route label checks and a check
   that no row is labelled custom.
4. Routes: native op means a single PyTorch native op
   (torch.softmax is a fused single-kernel op). Unfused manual
   means softmax built from separate PyTorch ops (amax, sub,
   exp, sum, div). Compiled (torch.compile) was attempted and
   failed on the node, so no compiled rows exist. Custom rows
   do not exist, because no custom kernel ran.
5. Stage profile: each unfused stage timed on its own with
   CUDA events at 4096 by 1024, then the full sequence, then
   the fused op, on the same input. Stage times do not add to
   the sequence time and are reported as attribution, not as
   a sum.
6. Profiler op counts: torch.profiler CPU activity per 10
   reps. CUDA-activity rows were unavailable in the committed
   runs (CUPTI failed to initialise), and the paper must say
   so.
7. C++ track: g++ -std=c++17 tools with a self-contained
   assertion runner (16 Build tests, 15 Deep tests), sanitizer
   runs under AddressSanitizer and UBSan, the atomics and
   handoff example, and the LargeBuffer move benchmark with
   counts read from class counters and asserted in tests.

## Results table

Measured 2026-11-05, from the level READMEs in the repo.
GPU rows are PyTorch routes only.

| Level | Set | Headline results | Tests |
| --- | --- | --- | --- |
| Learn | CPU counts, 3 sizes | Tiled loads 64, 512, 2048 against naive 128, 1024, 8192 | 12 Python; C++ bench tool PASS |
| Build | PyTorch native ops, 4 rows | Vector add 381.3 GB/s; matmul 6764.67 GFLOP/s at 1024; softmax 848.71 GB/s at 4096 by 1024 | 18 Python (30 total); 16 C++ |
| Deep, matmul | PyTorch native op, 3 sizes | 5422.94 at 512, 7642.29 at 1024, 5970.42 at 2048 GFLOP/s | Part of 9 verified rows |
| Deep, softmax | Native fused against unfused manual, 3 sizes | 604.58 against 155.13 GB/s at 4096 by 1024 (factor 3.9); 562.05 against 156.29; 563.94 against 155.06 | Part of 9 verified rows |
| Deep, C++ | Atomics, handoff, move | Totals exactly 1000000 both orderings; handoff 200 of 200 ordered; 50 copies 83.5614 ms, 50 moves 3.52111 ms | 16 Python (46 total); 15 C++ (31 total) |

Named non-result: the custom fused softmax kernel in
level-3-deep/cuda/fused_softmax.cu. Its expected profile
argument (one launch per call, predicted median near 0.083 ms
at 4096 by 1024 from its traffic model) is recorded in
level-3-deep/cuda/ANALYSIS.md as a prediction. The paper must
present it as future work to be replaced by a measured row in
v1.1, never as a result.

## Venue class note

Per PROPOSAL.md, the paper angle for P04 is the fused kernel
benchmark note (own kernel against PyTorch, profiler output).
Honest adjustment for v1.0: the own-kernel side of that angle
does not exist yet, because the kernel did not run, so the
v1.0 candidate is the route-labelled profiler note above. The
full SPEC angle becomes available only after v1.1 backlog
item 1 (custom kernel compile and run). Venue class is a
workshop or conference in GPU computing, systems, or ML
systems, or a journal of the AJSE class, per the publications
pipeline tracker rules. One venue at a time. No venue is
named as submitted, accepted, or contacted in this brief.
Fees, if any, arise only after acceptance and remain Jesun's
call under the free-only rule.

## Limits the paper must carry

- Every GPU number is a PyTorch native op or a manual
  sequence of PyTorch native ops. No custom kernel timing,
  correctness, or occupancy result exists.
- One device only (RTX 5050 Laptop GPU), committed runs only,
  with named clock variance across runs. Build and Deep rows
  come from different runs and must not be merged into one
  table without their run labels.
- torch.compile rows do not exist (the attempt failed on the
  node) and CUPTI CUDA-activity rows were unavailable in the
  committed runs.
- The atomics example runs on x86, which masks ordering
  differences. The relaxed handoff passing there is not
  evidence of correctness on a weakly ordered platform.
- C++ tests use a self-contained runner, not GoogleTest or
  Catch2. That deviation stays named until the v1.1 migration.
- Compressed Deep only. The v1.1 backlog (custom kernel
  compile and run, fused attention variant, framework
  migration, CUPTI and compiled rows, second device, weakly
  ordered run) is future work, named in REPORT.md (v1.1 backlog) in the
  repo.

## Submission gate

Submission is Jesun's tap and nothing is submitted. This
brief is the candidate only. The tracker row is updated by
the coordinator from the delivery report for this fan-out
cycle, not by this file.
