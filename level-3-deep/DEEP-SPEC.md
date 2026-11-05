# Level 3 Deep Spec: cuda-kernels-lab (P04)

Status: written before the Deep code, 2026-11-05.
Compressed Deep under the December Compression Protocol: the
single hardest capability in SPEC.md, built, tested, benchmarked,
with failure modes written. The rest of the v2 Deep scope is the
v1.1 backlog in the repo-root ROADMAP.md.

## Scope from SPEC.md

Deep closes when the Deep scope is done, failure modes are
written, and the dossier is updated to ship state. From SPEC.md:

- v1 core: Deep writes a fused softmax or attention kernel and
  profiles it against PyTorch.
- Added track: AddressSanitizer and UBSan pass over the tool with
  findings fixed; one atomics and memory-ordering example,
  measured; move semantics shown in one benchmark note.

## Decisions taken in this Deep, with reasons

1. Toolkit substitution, named, never silent. The fleet GPU node
   (BlackBox, NVIDIA GeForce RTX 5050 Laptop GPU, compute
   capability 12.0) has no CUDA toolkit: nvcc is not in PATH,
   CUDA_PATH is empty, and torch reports CUDA_HOME as None.
   Verified again this cycle by the coordinator and in the
   profiler run header. Custom .cu kernels therefore cannot be
   compiled or run on the current node. Installing a toolkit is
   the owner call and is not authorized in this cycle.
   Substitution: the profiler study runs with torch.profiler and
   CUDA-event timing on PyTorch routes that actually run on the
   node. Every row is labelled exactly as one of: native op,
   compiled, or custom. No row is labelled custom in this Deep.
   The custom fused softmax kernel is a written component only
   (see item 4), with the deduction named in the grade.
2. Fused route. The fused comparison is, in preference order:
   (a) torch.compile fused softmax if it runs on the Windows
   node, (b) the best fused route that does run, named exactly.
   The unfused comparison is a manual softmax built from separate
   PyTorch ops (max, sub, exp, sum, div), which launches several
   kernels per row batch. The profiler table records kernel
   counts and CUDA time from torch.profiler where available,
   plus CUDA-event medians for throughput.
3. C++ Deep track runs on the Muse VM with g++ -std=c++17.
   It evolves the Build tool: the Deep binary includes the Build
   tool functions, adds an atomics and memory-ordering example
   and a move-semantics benchmark, and the whole Deep binary plus
   the Build tool and tests run under AddressSanitizer and
   UBSan. The test framework stays the self-contained assertion
   runner from Build (no vendored GoogleTest or Catch2); that
   migration stays in the v1.1 backlog, as in Build.
4. Custom fused softmax CUDA kernel. level-3-deep/cuda/
   fused_softmax.cu carries the fused kernel source (one block
   per row, shared-memory reductions, stable max subtraction),
   extended from the Build softmax_rows_kernel. It is NOT
   compiled and has NOT run. Its README analysis states why it
   cannot compile on the current node and what profile it is
   expected to show once a toolkit exists (single kernel launch
   per call, no global-memory round trip for max and sum).

## Deliverables

- cpp/deep_tool.cpp: evolved C++ tool. Atomics example
  (atomic counter under relaxed and seq_cst ordering, plus a
  producer-consumer flag handoff under acquire-release and a
  relaxed control), move-semantics benchmark (a large buffer
  type copied vs moved, with copy and move counts and timings),
  and a self-check main.
- cpp/tests_deep.cpp: assertion test runner including
  deep_tool.cpp behind a no-main guard. Extends the C++ test
  count beyond the Build 16.
- cuda/fused_softmax.cu and cuda/ANALYSIS.md: the written custom
  kernel component and its compile-blocker analysis.
- profiler_bench.py: the BlackBox profiler script (torch.profiler
  plus CUDA events), run on BlackBox this cycle.
- results_profiler.json: measured profiler and timing rows from
  BlackBox, committed exactly as produced (plus device header).
- analysis_deep.py: recomputes GB/s and GFLOP/s from the stored
  profiler rows and asserts agreement within 1 percent, in the
  Build analysis.py pattern.
- run_deep.py: verifies the profiler rows, compiles and runs the
  C++ Deep tests, runs the Deep tool and the Build tool under
  ASan and UBSan, runs the atomics and move benchmarks, and
  writes results_deep.json.
- tests/test_deep_kernels.py: Python unit tests for the Deep
  analysis logic, the profiler file, the C++ Deep runner, the
  sanitizer outcomes, and the .cu source guards.
- README.md in this folder: results tables, failure modes,
  grade with itemized deductions.
- Repo-root ROADMAP.md, root README.md, DOSSIER.md updated to
  ship state.

## Acceptance criteria

- python3 -m unittest discover -s tests passes in full.
- The C++ Deep test runner compiles with -std=c++17 and every
  test prints PASS.
- The Deep tool and the Build tool run clean under
  -fsanitize=address,undefined (exit 0, no sanitizer output).
- Atomics example: relaxed and seq_cst counters both reach the
  exact expected total; the producer-consumer handoff passes
  every trial; timings for both orderings are measured and
  recorded, not asserted to a winner.
- Move benchmark: copy and move counts are exact for the stated
  workload and both timings are measured and recorded.
- Every stored profiler throughput recomputes within 1 percent.
  Every profiler row carries an exact route label (native op,
  compiled, or custom) and correct = true where a reference
  check ran.
- No custom kernel timing, correctness, or compile claim.
- Em dash count is 0 in every touched file.
