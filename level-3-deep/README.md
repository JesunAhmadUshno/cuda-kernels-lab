# Level 3, Deep: cuda-kernels-lab

Status: DONE, 2026-11-05. Spec: DEEP-SPEC.md in this folder.
Compressed Deep: the single hardest capability (profiler and
fused-kernel work, plus the SPEC C++ Deep track), with the
toolkit substitution named in DEEP-SPEC.md, never silent. The
rest of the v2 Deep scope is the v1.1 backlog in the repo-root
ROADMAP.md.

## What is here

- DEEP-SPEC.md: scope, decisions, acceptance criteria.
- profiler_bench.py: the GPU profiler and timing script, run on
  the fleet GPU node (BlackBox) this cycle. Its output is
  results_profiler.json, committed as produced.
- analysis_deep.py: recomputes every throughput in
  results_profiler.json from size and median_ms and asserts
  agreement within 1 percent, the route label is in the allowed
  set, and no row is labelled custom.
- cpp/deep_tool.cpp: the evolved C++ tool. It includes the Build
  tool functions and adds the atomics and memory-ordering
  example and the move-semantics benchmark (below).
- cpp/tests_deep.cpp: 15 assertion tests for the Deep tool.
- cuda/fused_softmax.cu and cuda/ANALYSIS.md: the custom fused
  softmax kernel, written only, with the compile-blocker
  analysis and its expected profile argument.
- run_deep.py: verifies the profiler rows, runs the Deep and
  Build C++ test runners, runs both tools under ASan and UBSan,
  runs the Deep tool and records its measured values into
  results_deep.json.

## Profiler study: PyTorch routes on BlackBox

Measured on the fleet GPU node (BlackBox). Device: NVIDIA
GeForce RTX 5050 Laptop GPU, compute capability 12.0, torch
2.11.0+cu128. Method: CUDA events, median of repeated runs
after warmup, correctness checked against a reference before
timing. Source file: results_profiler.json, verified here by
recomputation (analysis_deep.py, all 9 rows within 1 percent,
every correct flag true).

Route labels are exact. Native op means a single PyTorch
native op (torch.softmax is a fused single-kernel op).
Unfused manual means softmax built from separate PyTorch ops
(amax, sub, exp, sum, div). Compiled means torch.compile; it
was attempted and failed on this node (see Failure modes), so
there are no compiled rows. There are no custom rows: no
custom CUDA kernel ran.

### Matmul, native op

| Kernel | Route | Size | Median | Throughput | Correct |
| --- | --- | --- | --- | --- | --- |
| matmul_torch | native op | 512 | 0.0495 ms | 5422.94 GFLOP/s | true |
| matmul_torch | native op | 1024 | 0.281 ms | 7642.29 GFLOP/s | true |
| matmul_torch | native op | 2048 | 2.8775 ms | 5970.42 GFLOP/s | true |

### Softmax, native fused op vs unfused manual sequence

Throughput uses one traffic model for both routes (2 times
rows times cols times 4 bytes) so the routes compare on one
axis; the unfused sequence moves more traffic in reality,
which is part of why it is slower.

| Kernel | Route | Size | Median | Throughput | Correct |
| --- | --- | --- | --- | --- | --- |
| softmax_rows_torch | native op | 4096 by 1024 | 0.0555 ms | 604.58 GB/s | true |
| softmax_rows_unfused_manual | unfused manual (native ops) | 4096 by 1024 | 0.2163 ms | 155.13 GB/s | true |
| softmax_rows_torch | native op | 2048 by 2048 | 0.0597 ms | 562.05 GB/s | true |
| softmax_rows_unfused_manual | unfused manual (native ops) | 2048 by 2048 | 0.2147 ms | 156.29 GB/s | true |
| softmax_rows_torch | native op | 8192 by 512 | 0.0595 ms | 563.94 GB/s | true |
| softmax_rows_unfused_manual | unfused manual (native ops) | 8192 by 512 | 0.2164 ms | 155.06 GB/s | true |

Headline: at 4096 by 1024 the fused native op median is
0.0555 ms against 0.2163 ms for the unfused sequence, a factor
of 3.9. The pattern holds at all three sizes (unfused medians
cluster at 0.215 ms regardless of shape at this element count,
the fused medians at 0.056 to 0.060 ms).

### Stage profile, softmax 4096 by 1024 (CUDA events)

Each unfused stage timed on its own, then the full sequence,
then the fused op, same input:

| Stage | Median |
| --- | --- |
| amax (row max) | 0.0218 ms |
| exp(x - max) | 0.1464 ms |
| sum (row sum) | 0.0222 ms |
| div (normalise) | 0.0908 ms |
| unfused full sequence | 0.2147 ms |
| native fused softmax (single op) | 0.0572 ms |

The exp and div stages dominate the unfused cost: each one
materialises a full 16 MiB intermediate in global memory. The
fused op writes no intermediate to global memory. Stage times
do not add to the sequence time (each stage timing includes
its own launch and materialisation), they show where the
sequence spends its time.

### Profiler op counts (torch.profiler, CPU activity)

Per 10 repetitions, from results_profiler.json:

- matmul 1024, native op: aten::matmul 10, aten::mm 10.
  One op pair per call.
- softmax 4096 by 1024, native op: aten::softmax 10,
  aten::_softmax 10. One fused op per call.
- softmax 4096 by 1024, unfused manual: aten::amax 10,
  aten::sub 10, aten::exp 10, aten::sum 10, aten::div 10.
  Five ops per call, each a separate kernel launch.

CUDA-activity rows from torch.profiler were not available in
the committed runs (CUPTI failed to initialise, see Failure
modes), so the profiler evidence is the op counts here plus
the CUDA-event tables above, exactly as recorded in the file.

## C++ Deep track (Muse VM, g++ -std=c++17)

Recorded run: results_deep.json, produced by run_deep.py on
2026-11-05 (g++ 13.3.0). Timings are this run's measured
values; counts and totals are exact by construction and are
asserted in the tests.

### Sanitizers

- Deep tool under ASan and UBSan: CLEAN (exit 0, no report).
- Build tool under ASan and UBSan: CLEAN (exit 0, no report).
- The Deep test runner (15 tests) also ran clean under ASan
  and UBSan this cycle.
- No new sanitizer findings appeared in the evolved tool; the
  Build debugging session findings stay fixed. A
  ThreadSanitizer run of the Deep tool under
  setarch x86_64 -R also passed clean this cycle (see Failure
  modes for why TSan is a manual check here, not a gate).

### Atomics and memory ordering, measured

Atomic counter, 4 threads by 250000 increments:

| Ordering | Final total | Expected | Wall time |
| --- | --- | --- | --- |
| relaxed | 1000000 | 1000000 | 25.8349 ms |
| seq_cst | 1000000 | 1000000 | 20.6221 ms |

Both orderings reach the exact total: atomicity of the
read-modify-write guarantees the count under either ordering.
The wall times overlap across repeated runs on this x86 VM
and no ordering winner is claimed; ordering cost is platform
dependent and this platform does not separate them.

Producer-consumer handoff (payload through a plain int, flag
as an atomic, 200 trials):

| Form | Successes | Wall time |
| --- | --- | --- |
| acquire-release (release store, acquire load) | 200 of 200 | 15.4334 ms |
| relaxed control (relaxed store and load) | 200 of 200 | 16.495 ms |

The ordered form is the one with a C++ guarantee: the release
store and acquire load order the payload write before the
payload read. The relaxed control also delivered every
payload on this machine, which is expected on x86 (a strong
memory model masks the missing ordering); the test asserts
the ordered form, and only completion for the control. On a
weakly ordered platform the relaxed form has no such promise.

### Move semantics benchmark, measured

LargeBuffer, 200000 doubles (1.6 MB) per instance, 50 reps:

| Path | Constructor counts | Wall time |
| --- | --- | --- |
| copy | 50 copies, 0 moves | 83.5614 ms |
| move | 50 moves, 0 copies | 3.52111 ms |

Counts are exact (read from the class counters) and asserted
in the tests, including that a moved-from buffer is empty and
its contents arrive intact (checksum 150.0 over both paths).
Moving transfers the vector instead of copying 1.6 MB per
rep, and the measured wall time shows it: about 24 times
faster on this run.

### Tests

- C++ Deep tests: 15 passed, 0 failed (includes the 3 Build
  functions exercised inside the Deep tool, atomics totals,
  handoff, and move counts).
- C++ Build tests (regression): 16 passed, 0 failed.
- Python suite: 46 tests pass in total (16 new Deep tests in
  tests/test_deep_kernels.py).

## Custom fused softmax kernel status

Named blocker, stated plainly, unchanged from Build:

- cuda/fused_softmax.cu is written: softmax_rows_fused_kernel,
  one block per row, shared-memory exp cache, stable max
  subtraction. It is UNCOMPILED and has never run.
- Why: the fleet GPU node has no CUDA toolkit (nvcc not on
  PATH, cuda_home null in the profiler header), and installing
  one is the owner call, not authorized this cycle.
- cuda/ANALYSIS.md records the full blocker evidence and the
  expected profile argument (one launch per call, predicted
  median near 0.083 ms at 4096 by 1024 from its traffic model),
  labelled as a prediction to be replaced by a measured row in
  v1.1.

## Failure modes

- No CUDA toolkit on the GPU node. Custom kernels cannot
  compile or run; every GPU number in this level is a PyTorch
  route. Mitigation is owner-side (install the toolkit), then
  v1.1 backlog item 1 runs the kernel.
- torch.compile does not run on this Windows node, by two
  recorded routes: the committed profiler run failed with
  TritonMissing (no working Triton installation), and an
  earlier probe in this cycle failed later in Inductor with
  the error that it needs a C compiler (cl) on the path. So
  there is no compiled route to compare; the fused side of the
  comparison is the native fused op, labelled as such.
- CUPTI is unreliable on this node. One early probe in this
  cycle returned full CUDA kernel rows from torch.profiler;
  the committed runs failed CUPTI initialisation with
  CUPTI_ERROR_INVALID_DEVICE and recorded
  cuda_activity_available false with zero kernel rows, not
  invented ones. The timing evidence therefore rests on CUDA
  events, and the profiler evidence on CPU-activity op counts.
- GPU clock variance between runs. Repeated profiler runs this
  cycle spread (softmax native median 0.0403 ms in the first
  run, 0.0555 ms in the committed run; matmul GFLOP/s at 2048
  spread across runs). Laptop GPU clocks move with thermals
  and power state. Only the committed run is quoted in this
  README, and the ratios inside one run (fused vs unfused,
  same session) are the stable finding.
- x86 masks ordering bugs. The relaxed handoff control passes
  on this platform; that is not evidence it is correct. The
  ordered form is the shipped pattern, and the tests assert
  only what the platform can show.
- ThreadSanitizer on the build VM aborts with a memory
  mapping FATAL unless ASLR is disabled for the process. It
  was run manually under setarch x86_64 -R and passed clean;
  it is not wired into run_deep.py or CI, because a gate that
  fails for environment reasons teaches people to ignore it.
- Sanitizers do not cover the GPU side at all. Kernel-side
  memory errors in the custom kernel remain untested until it
  compiles (the Build canary demo pattern is its planned
  check, in cuda/ANALYSIS.md).
- LargeBuffer moved-from state is empty by design here
  (data_.clear() after the move). Code that reads a moved-from
  buffer expecting contents will get an empty buffer, loudly
  in size checks, silently in element loops that tolerate
  empty ranges.

## Grade

7,900 of 10,000. Deductions: custom CUDA kernel written but
not compiled or run, toolkit missing on the GPU node (minus
1,000, the largest; no custom kernel number exists because of
it, and the fused comparison is against the native op instead
of the project's own kernel); torch.compile route failed on
the node by two recorded errors, so no compiled row exists
(minus 300); CUPTI CUDA-activity profiler rows unavailable in
the committed runs, profiler evidence rests on op counts plus
CUDA-event tables (minus 250); C++ tests still use the
self-contained runner instead of GoogleTest or Catch2, the
named Build deviation carried into Deep (minus 200); GPU
numbers are one laptop GPU in one committed run, with named
clock variance across runs this cycle (minus 200); the
atomics example runs on x86, whose strong memory model masks
ordering differences, so the ordering cost and the relaxed
hazard are argued, not demonstrated (minus 150).
