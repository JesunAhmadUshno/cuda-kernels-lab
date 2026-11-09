# PROPOSAL: cuda-kernels-lab (P04)

Status: NOT LOCKED. This repo is not pre-registered. The scaffold spec
below was committed on 2026-11-04 (commit 0556666), before any code or
result, but it states scope and acceptance criteria only: no
hypotheses, no pass lines and no analysis plan. It is therefore not a
pre-registration, and no lock tag is placed on it. Under the portfolio
standard (METHOD section 7) this repo gets a confirmatory re-run under
a newly written PROPOSAL.md, locked by tag before the re-run; that
re-run is not part of the standard v2 migration.

Everything below the "Candidate hypotheses" section is moved verbatim
from the files it replaces: SPEC.md, NOVELTY.md, DOSSIER.md,
level-2-build/BUILD-SPEC.md and level-3-deep/DEEP-SPEC.md. The moved
text keeps the old file names as written; the evidence map in
REPORT.md gives each old file's new home.

## Question

Can a hand-written CUDA kernel stack, from vector add to a fused
softmax kernel, be built and profiled against PyTorch on one named
GPU, with every number traceable to a recorded run?

## Candidate hypotheses for the confirmatory re-run (draft, not locked)

Drawn from predictions already written in this repo. Pass lines,
budgets and the number of registered trials are open and belong to the
Owner before the lock.

- H1, fused kernel correctness and speed: level-3-deep/cuda/fused_softmax.cu
  matches the PyTorch reference on every registered shape, and its
  median at 4096 by 1024 is near the 0.083 ms predicted in
  level-3-deep/cuda/ANALYSIS.md (tolerance: open).
- H2, fused against unfused: the custom fused kernel beats the unfused
  five-op PyTorch sequence at every registered shape (margin: open).
- H3, tiling: the tiled matmul in level-2-build/cuda/kernels.cu beats
  the naive kernel at the load-table sizes and larger, consistent with
  the load counts in level-1-learn/results_learn.json (margin: open).

Field-profile requirements the re-run must carry (ML and AI research):
baselines a reviewer expects (PyTorch native op, unfused sequence,
torch.compile where it runs), every timed repetition stored with an
interval across sessions, and device, driver, toolkit and torch
versions recorded with each result.

## Scope (scaffold spec, SPEC.md, 2026-11-04)

Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 2.
Hiring evidence: 4 of 5 (CUDA in 12/56) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

### Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

cuda-kernels-lab: EXPAND

Subject: GPU programming.
v1 core [confirmed, v1]: CUDA kernels from vector add to tiled matrix multiply; Deep writes a fused softmax or attention kernel and profiles it against PyTorch.
Added track, the standalone C++ tool (register items 71 to 76, 126, 127, 135):
- Learn: grow the kernel host code into one small standalone C++ tool (a kernel launch and benchmark runner): smart pointers and RAII, templates and the STL, CMake build.
- Build: GoogleTest or Catch2 tests on the tool; one gdb or LLDB debugging session written up from a real crash.
- Deep: AddressSanitizer and UBSan pass over the tool with findings fixed; one atomics and memory-ordering example, measured; move semantics shown in one benchmark note.
Fan-out: Book chapter "Down To The Metal". Paper angle: the fused kernel benchmark note (own kernel against PyTorch, profiler output). Course module: P04-cuda-kernels-lab.

### Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.

## Prior work and novelty (NOVELTY.md, draft)

Status: DRAFT, PROPOSED. Not verified. This text does not move into any public post, paper, or profile until the Jev claim_check gate passes on the exact text.

### Proposed claim

Against vendor CUDA samples and PyTorch kernels: this project aims to differ in the way its SPEC.md section text defines (a from-zero or self-hosted build, measured in public, on a named home fleet, with the failure modes written down). The exact difference is stated at the project's wave, after the prior art is re-read.

### Prior art table

| Prior art | What it does | What this project does differently | Measured difference |
| --- | --- | --- | --- |
| vendor CUDA samples and PyTorch kernels | See DOSSIER.md, Past | PROPOSED, see the claim above | NO MEASUREMENT YET |

### Honest size

If the novelty proves small, this file will say so and name the real value instead (a cleaner reproduction, a better benchmark, a first public dataset). No novelty is invented to fill this section.

## Dossier: past, present, future (DOSSIER.md)

Written under the documentation and publishing standard (past, present, future). Updated at ship time.

### Past

Prior art for this project is named in NOVELTY.md (vendor CUDA samples and PyTorch kernels). The history of the problem, the key papers and systems, and what was tried before are recorded in the Roadmap v2 section text in SPEC.md and will be expanded with dated sources at the project's wave, before Build opens.

### Present

Current demand evidence: hiring value 4 of 5 (CUDA in 12/56) [confirmed, fleet and scoring pass 2026-11-04; for P23 to P29, the decision pack 2026-11-04]. The Round 3 research (2026-11-04) found the market centre at post-training, evals, environments, and inference depth [confirmed as the research finding; the report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md]. Tools and versions are pinned at Build time, not in this v1.

Update, Level 2 Build DONE 2026-11-05: the standalone C++ tool now carries vector add, tiled matmul, and stable softmax with a 16-test self-contained assertion runner (no external framework; Catch2 and GoogleTest were not fetched and nothing is vendored). A real debugging session is on file at level-2-build/DEBUG-SESSION.md: a tiled matmul variant missing its bounds guard produced a real AddressSanitizer heap-buffer-overflow WRITE at an edge size, and the fixed tool runs clean under ASan and UBSan (session is sanitizer-led; gdb is not installed on the build machine). GPU benchmarks on record are PyTorch native ops measured on the fleet GPU node (NVIDIA GeForce RTX 5050 Laptop GPU, compute capability 12.0, torch 2.11.0+cu128, CUDA events, median): vector add 381.3 GB/s, matmul 6764.67 GFLOP/s at 1024, softmax rows 848.71 GB/s; all 4 stored rows recompute within 1 percent. The custom CUDA harness level-2-build/cuda/kernels.cu is written but uncompiled and has never run: the GPU node has no CUDA toolkit in PATH or CUDA_PATH this cycle, torch reports CUDA_HOME None, and a Numba route failed on compute capability 12.0 support. Compiling and running it is the first Deep item.

Update, Level 3 Deep DONE 2026-11-05 (compressed Deep, release v1.0): the profiler study ran on the fleet GPU node with torch.profiler op counts and CUDA-event timing over PyTorch routes only, every row labelled: matmul (native op) at 512, 1024, and 2048 (7642.29 GFLOP/s at 1024), softmax native fused op against an unfused manual sequence at three sizes (604.58 GB/s against 155.13 GB/s at 4096 by 1024, a factor of 3.9, with a per-stage profile: exp and div stages dominate the unfused cost), 9 rows verified by recomputation within 1 percent. torch.compile was attempted and failed on the Windows node by two recorded routes (TritonMissing in the committed run; Inductor needing a C compiler in an earlier probe), and CUPTI CUDA-activity rows were unavailable in the committed runs (CUPTI_ERROR_INVALID_DEVICE), so profiler evidence is op counts plus CUDA-event tables, recorded as such. The C++ Deep track evolved the tool with an atomics and memory-ordering example (counter exact at 1000000 under relaxed and seq_cst; producer-consumer handoff 200 of 200 under acquire-release, with the relaxed control also passing on x86 and named as no guarantee) and a move-semantics benchmark (LargeBuffer, 200000 doubles: 50 copies in 83.5614 ms, 50 moves in 3.52111 ms, counts exact). 15 new C++ tests pass (31 C++ total), 46 Python tests pass, and both tools run clean under ASan and UBSan with no new findings. The custom fused softmax kernel (level-3-deep/cuda/fused_softmax.cu) is written only, with its blocker analysis and expected profile argument in cuda/ANALYSIS.md: the node still has no CUDA toolkit, so it was not compiled or run, and compiling and running the custom kernels is v1.1 backlog item 1 in ROADMAP.md.

### Future

The v1.1 backlog is in ROADMAP.md at the repo root: custom kernel compile and run first, then the fused attention variant, the GoogleTest or Catch2 migration, CUPTI and compiled-route profiler rows when the node supports them, a second device, and a weakly ordered platform run for the atomics example. Any forecast about this field is [unverified] unless a dated primary source is added here at the wave update.

## Level 2 Build spec (level-2-build/BUILD-SPEC.md)

Status: written before the Build code, 2026-11-05.

### Scope from SPEC.md

Build closes when the system runs end to end, tests pass, and one
benchmark or result table with method is in the README. The Added
track names: tests on the standalone C++ tool (SPEC text says
GoogleTest or Catch2), and one debugging session written up from a
real crash (SPEC text says gdb or LLDB).

### Decisions taken in this Build, with reasons

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

### Deliverables

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

### Acceptance criteria

- python3 -m unittest discover -s tests passes in full.
- The C++ test runner compiles and every test prints PASS.
- tool.cpp built with -fsanitize=address,undefined runs clean.
- buggy.cpp built with sanitizers produces a real heap-buffer-overflow
  WRITE report whose SUMMARY line is quoted verbatim in
  DEBUG-SESSION.md.
- Every stored GPU throughput recomputes within 1 percent.
- Em dash count is 0 in every touched file. No invented numbers:
  the only GPU numbers are the stored measured rows.

## Level 3 Deep spec (level-3-deep/DEEP-SPEC.md)

Status: written before the Deep code, 2026-11-05.
Compressed Deep under the December Compression Protocol: the
single hardest capability in SPEC.md, built, tested, benchmarked,
with failure modes written. The rest of the v2 Deep scope is the
v1.1 backlog in the repo-root ROADMAP.md.

### Scope from SPEC.md

Deep closes when the Deep scope is done, failure modes are
written, and the dossier is updated to ship state. From SPEC.md:

- v1 core: Deep writes a fused softmax or attention kernel and
  profiles it against PyTorch.
- Added track: AddressSanitizer and UBSan pass over the tool with
  findings fixed; one atomics and memory-ordering example,
  measured; move semantics shown in one benchmark note.

### Decisions taken in this Deep, with reasons

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

### Deliverables

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

### Acceptance criteria

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
