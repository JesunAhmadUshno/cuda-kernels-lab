# REPORT: cuda-kernels-lab (P04)

Status: interim report, 2026-11-08. The question is not answered yet:
the project's own CUDA kernels have never been compiled or run. This
report records what did run, where its evidence is, and what has no
evidence in the repo. Sections after "Evidence map" are moved verbatim
from the files they replace (the old README body, docs/research-review.md
and ROADMAP.md) and keep the old file names as written; the evidence
map gives each old file's new home.

## Background

The question (PROPOSAL.md) asks whether a hand-written CUDA kernel
stack, from vector add to a fused softmax kernel, can be built and
profiled against PyTorch on one named GPU. The work ran in three
levels on 2026-11-05: CPU reference kernels with exact load counts
(level-1-learn), a standalone C++ tool with tests, a sanitizer
debugging session and PyTorch GPU baselines (level-2-build), and a
profiler study of PyTorch routes plus C++ atomics and move benchmarks
(level-3-deep). Each level folder has its own write-up with method,
tables and failure modes.

## Method

- CPU side: exact operation and load counts from the loop structure
  (level-1-learn/kernels.py), checked against an independent reference
  by level-1-learn/run_learn.py, which writes
  level-1-learn/results_learn.json deterministically. CI regenerates
  that file and the Build summary (level-2-build/results_build.json)
  and fails if either differs from the committed copy.
- GPU side: level-2-build and level-3-deep/profiler_bench.py on the
  fleet GPU node (NVIDIA GeForce RTX 5050 Laptop GPU, compute capability
  12.0, torch 2.11.0+cu128), CUDA events, a correctness check against
  the reference before a time counts. Deep: 5 warm-up runs, then the
  median of 30 timed runs (20 for matmul). Build: median of repeated
  runs after warm-up (level-2-build/README.md). Only the median is stored; the individual timed runs
  were not kept, so no interval exists for any GPU number.
- Throughputs are recomputed from the stored medians by
  level-2-build/analysis.py and level-3-deep/analysis_deep.py (within
  1 percent), in CI and in the site build.
- C++ side: g++ -std=c++17 on an x86 build VM; tests are a
  self-contained assertion runner (docs/decisions/0001); ASan and UBSan
  runs in CI.

## Hypotheses scored

No hypothesis was registered before the results, so none is scored as
confirmed or refuted. The one written prediction, ANALYSIS.md's median
near 0.083 ms for fused_softmax.cu at 4096 by 1024, is UNTESTED: the
kernel has never run. The draft hypotheses for the confirmatory re-run
are in PROPOSAL.md.

## Results that did run

| Result | Value | Evidence file | Regenerate or check |
| --- | --- | --- | --- |
| PyTorch native fused softmax against the five-op unfused sequence, 4096 by 1024 | 0.0555 ms against 0.2163 ms, ratio 3.9 (604.58 against 155.13 GB/s); one median per route, one session, no interval | level-3-deep/results_profiler.json | python3 level-3-deep/analysis_deep.py |
| Same comparison at 2048 by 2048 and 8192 by 512 | ratio 3.6 at both (0.0597 against 0.2147 ms; 0.0595 against 0.2164 ms) | level-3-deep/results_profiler.json | python3 level-3-deep/analysis_deep.py |
| PyTorch matmul (native op) | 5422.94, 7642.29, 5970.42 GFLOP/s at 512, 1024, 2048 | level-3-deep/results_profiler.json | python3 level-3-deep/analysis_deep.py |
| Build PyTorch rows | vector add 381.3 GB/s, matmul 6764.67 GFLOP/s at 1024, softmax 848.71 GB/s | level-2-build/results_torch_bench.json | python3 level-2-build/analysis.py |
| Tiled against naive load counts | 2,048 against 8,192 at 16 by 16, tile 4 (and two smaller sizes) | level-1-learn/results_learn.json | python3 level-1-learn/run_learn.py (byte-identical) |
| C++ tests and sanitizers | 16 Build + 15 Deep tests pass; both tools clean under ASan and UBSan | level-3-deep/results_deep.json, CI log | python3 level-3-deep/run_deep.py |
| Atomics and move benchmark (recorded run) | counter exactly 1000000 under relaxed and seq_cst; handoff 200 of 200; 50 copies 83.5614 ms, 50 moves 3.52111 ms | level-3-deep/results_deep.json (committed copy) | counts are exact on every run; times change with the machine, so CI does not compare them |

The same native op measured in two sessions differs by up to 54
percent (matmul at 512: 3529.07 GFLOP/s in Build, 5422.94 in Deep), so
no single-session GPU number above should be read as a device constant.
Only the within-session ratios are treated as findings.

## Stated in the write-ups, no evidence file in the repo

These statements come from the level write-ups. They are recorded
observations, not results, because no log or results file for them is
committed:

- The earlier profiler run's softmax native median of 0.0403 ms, and
  the run-to-run spread of matmul at 2048 (level-3-deep/README.md).
- One early torch.profiler probe that returned CUDA kernel rows before
  CUPTI failed (level-3-deep/README.md).
- The earlier torch.compile probe that failed in Inductor for want of a
  C compiler (only the TritonMissing failure is in
  results_profiler.json).
- ThreadSanitizer passing clean when run by hand under setarch -R.
- The CMake build of the Learn tool verified locally with g++ 13.3.0.
- The Numba route failing on compute capability 12.0.

## Misses

- The project's own kernels (level-2-build/cuda/kernels.cu,
  level-3-deep/cuda/fused_softmax.cu) were never compiled or run: no
  CUDA toolkit on the GPU node. This is the central miss; it is why the
  verdict is Draft.
- No torch.compile row, no CUPTI kernel rows, one device, no interval.
- C++ tests use a home-grown runner instead of GoogleTest or Catch2;
  the debugging session is sanitizer-led, not gdb.

## Threats to validity

Each level write-up ends with its failure modes (level-1-learn,
level-2-build and level-3-deep README.md, and
level-3-deep/cuda/ANALYSIS.md). The main ones: laptop GPU clocks move
with thermals and power state; x86 masks memory-ordering bugs; models
(traffic counts, the fused-kernel prediction) are arithmetic, not
measurements.

## Conclusion

The repo is a careful record of baselines and of a C++ tool, with
every number recomputed. It does not yet answer its question. The next
step is a CUDA toolkit on the GPU node (Owner call), a locked
PROPOSAL.md for the confirmatory re-run, and then the custom kernel
rows.

## Evidence map

| Old file | New home |
| --- | --- |
| SPEC.md, NOVELTY.md, DOSSIER.md | PROPOSAL.md |
| level-2-build/BUILD-SPEC.md, level-3-deep/DEEP-SPEC.md | PROPOSAL.md |
| README.md body (status, benchmarks, fan-out, novelty, reading order) | REPORT.md, below |
| docs/research-review.md | REPORT.md, below |
| ROADMAP.md | REPORT.md, below (Future work) |

## Results on record (moved from the README, 2026-11-05)

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
ROADMAP.md.
Project: P04. Fleet node: BlackBox (GPU). Wave: Wave 2.

Live dashboard: https://jesunahmadushno.github.io/cuda-kernels-lab/

The dashboard (source in site/) reads every number from the
level-*/results*.json files and the level write-ups at build time.
The tiling, matmul and softmax calculators run in the browser; the
GPU and C++ numbers are labelled replays of the recorded runs; the
fused-kernel figure is labelled a prediction. A research-rubric
review of the repo is in docs/research-review.md.

### Level map

- Level 1, Learn: the core idea, built small and clean, with a teaching path.
- Level 2, Build: a real working system other people can run.
- Level 3, Deep: benchmarks, optimization, evaluation, hard cases.

### Benchmarks

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

## Research review (docs/research-review.md, 2026-11-07)

Date: 2026-11-07. Scope: the whole repo at v1.0 plus the site/
dashboard added on branch site/live-dashboard. Every number quoted here is in a committed
results file or is recomputed from one by site/src/lab.check.mjs.

Rule used: every gap that can be closed without a GPU, a toolkit
install, paid services or long runs is closed in this change. Every
remaining gap names the exact compute or data it needs.

### Summary

| Criterion | Finding |
| --- | --- |
| Research question | Clear in SPEC.md (own fused kernel against PyTorch), but the data answers a narrower question (PyTorch fused op against PyTorch unfused ops). |
| Hypotheses | ANALYSIS.md states one falsifiable prediction (about 0.083 ms for fused_softmax.cu at 4096 by 1024); no other hypothesis is written before measurement. |
| Method | CUDA events, warmup, median, correctness gate before timing, exact route labels. Single device, single session per number. |
| Baselines | Native fused op and unfused op sequence are honest baselines; the project's own kernel arm and the torch.compile arm are missing. |
| Statistics and uncertainty | One median per row, no dispersion, raw samples not stored. Cross-session drift is now quantified (below) but not bounded. |
| Reproducibility | Results files committed as produced, every throughput recomputed in Python and now in the site build, CI runs all CPU-side tests. GPU runs need the named node. |
| Threats to validity | Each level README names its failure modes; the dashboard now carries a limitations window. |
| Related work | DOSSIER.md covers the field informally; NOVELTY.md is a draft. A short positioning section is added below. |
| Limitations | Stated plainly in README, level write-ups and on the page; custom-kernel status is never blurred. |

Overall: a careful engineering record with honest labels, short of
a research result because the central comparison (own kernel
against PyTorch) has no data yet.

### Research question

SPEC.md: Deep "writes a fused softmax or attention kernel and
profiles it against PyTorch". What the data supports today: on one
RTX 5050 Laptop GPU, PyTorch's native fused softmax ran 3.9 times
faster than the same math as five separate PyTorch ops (0.0555 ms
against 0.2163 ms at 4096 by 1024, results_profiler.json). The
dashboard headline and status board say this, and say the lab's
own kernels are written but not compiled.

### Gaps closed in this change (no GPU needed)

1. Site cannot drift from the repo. site/src/lab.check.mjs runs on
   `npm test` and before every `npm run build` (so the Pages
   workflow cannot publish a drifted page). It asserts all three
   load_table rows of level-1-learn/results_learn.json from both
   the closed forms and a loop-for-loop JS port of
   kernels.matmul_tiled, the run_learn.py softmax and matmul hand
   cases, and that all 13 recorded GPU throughputs (9 Deep, 4
   Build) recompute within 1 percent, mirroring analysis.py and
   analysis_deep.py.
2. Edge-padding caveat (Learn README deduction). An in-bounds tiled
   load count, M·K·ceil(N/t) + K·N·ceil(M/t), is computed next to
   the padded count and checked against a brute-force count of the
   guarded loads in the port. Example at 18 by 14 by 16, tile 4:
   2,560 padded slots against 2,272 in-bounds loads. Computed, not
   measured.
3. Per-route traffic model. The shared 2x axis understates the
   unfused route's traffic. Counting it op by op (amax 1, sub 2,
   exp 2, sum 1, div 2 passes = 8) gives 620.5 GB/s effective
   traffic for the unfused route at 4096 by 1024, close to the
   fused op's 604.58 GB/s. Reading: the unfused route is not slow
   per byte; it moves about 4 times the bytes, consistent with the
   3.9 times gap. Labelled MODEL on the page.
4. Fused-kernel prediction made inspectable. The ANALYSIS.md
   arithmetic (3 passes at the native op's 604.58 GB/s, about
   0.083 ms) is reproduced by the build check and exposed as a
   calculator, badged PREDICTION, with the 1024-column shared-cache
   limit warned.
5. Run-to-run variance quantified. Same native op, Build session
   against Deep session: matmul 512 +53.7 percent, matmul 1024
   +13.0 percent, softmax 4096 by 1024 −28.8 percent. Shown as a
   "same op, two sessions" panel so no single-session throughput is
   read as a device constant.
6. Provenance naming. The page states that the C++ track ran on the
   x86 build VM (called "Muse VM" in the write-ups) and the GPU rows
   on the fleet GPU node (BlackBox), so the two names are no longer
   confusing.
7. Related-work positioning (below).

### Remaining gaps and the exact compute or data each needs

| Gap | What closes it |
| --- | --- |
| No custom kernel has run (kernels.cu, fused_softmax.cu) | A CUDA toolkit with sm_120 support on the GPU node (owner call), then compile both files, run their harness checks, and add rows labelled custom to results_profiler.json. ROADMAP item 1. |
| Fused attention kernel not written | Writing it is free; measuring it needs the same toolkit and GPU time. ROADMAP item 2. |
| No dispersion, no raw samples | Re-run profiler_bench.py on the GPU node storing every timed rep, at least 5 separate sessions, then report median with IQR and a bootstrap interval across sessions. A few minutes of GPU time per session. |
| Single device, no clock or power log | A second GPU (ROADMAP item 6) and nvidia-smi clock, power and temperature logging during each run (ROADMAP item 8). |
| No hardware roofline | The device's peak DRAM bandwidth and FP32 peak, from the vendor datasheet for this exact laptop SKU or from a measured device-to-device copy on the GPU node. The repo records neither, so no efficiency figure is shown. |
| CUPTI kernel rows missing | A reliable CUPTI initialisation on the node and a profiler re-run. ROADMAP item 4. |
| torch.compile arm missing | A working Triton (or a C compiler for Inductor) on the GPU node. ROADMAP item 5. |
| Tiling effect not measured | Compiled kernels.cu on a GPU, naive against tiled at the load_table sizes and larger. |
| Ordering argued, not observed | The atomics and handoff run on a weakly ordered (ARM) machine. ROADMAP item 7. |
| Home-grown C++ test runner, TSan not a gate | GoogleTest or Catch2 fetched in CI (a dependency fetch, ROADMAP item 3); a CI runner where TSan works without disabling ASLR. |
| NOVELTY.md still a draft | The claim_check gate the repo names, a paid review step outside this change. |

### Related work

- Milakov and Gimelshein, "Online normalizer calculation for
  softmax", arXiv:1805.02867, 2018. Computes the max and the
  normaliser in one pass. fused_softmax.cu reads the input twice;
  an online-normaliser variant is the natural next step toward the
  2-pass minimum.
- Dao, Fu, Ermon, Rudra and Ré, "FlashAttention: Fast and
  Memory-Efficient Exact Attention with IO-Awareness", NeurIPS
  2022, arXiv:2205.14135. Fuses score, softmax and value product
  with tiling to avoid writing intermediates to global memory; the
  fused attention kernel in ROADMAP item 2 belongs to this line.
- Williams, Waterman and Patterson, "Roofline: an insightful visual
  performance model for multicore architectures", Communications of
  the ACM 52(4), 2009. The model this repo lacks a device peak for.
- Volkov and Demmel, "Benchmarking GPUs to tune dense linear
  algebra", SC 2008. Classic account of tiling and shared-memory
  reuse in GPU matmul, the subject of Level 1.

Positioning: this repo makes no novelty claim. Its value is a fully
labelled teaching and measurement record in which every number is
recomputed and nothing unrun is reported as a result.

### Engineering notes

- Tests: 46 Python tests and 31 C++ tests pass locally; the site
  check is part of the site build.
- Site dependencies are exact-pinned (react 18.3.1, react-dom
  18.3.1, vite 6.4.5, @vitejs/plugin-react 4.3.4). A full
  `npm audit` in `site/` (dev dependencies included) reports 0
  vulnerabilities, and the published site is static files only.
- No secrets, no external scripts; fonts load from Google Fonts.

## Future work (ROADMAP.md, v1.1 backlog)

Written at Level 3 Deep ship time, 2026-11-05. v1.0 ships
Learn, Build, and compressed Deep. Everything below is the
remaining v2 Deep scope, named as the v1.1 backlog under the
December Compression Protocol. Nothing below is done, and
nothing below is claimed anywhere in this repo as done.

## v1.1 backlog, in order

1. Compile and run the custom CUDA kernels. Get a CUDA
   toolkit on the fleet GPU node (owner call), compile
   level-2-build/cuda/kernels.cu and
   level-3-deep/cuda/fused_softmax.cu, run their harness
   checks (including the canary demo for the Build harness),
   and publish measured custom rows next to the PyTorch
   tables, every row labelled custom. The fused kernel's
   expected profile argument in
   level-3-deep/cuda/ANALYSIS.md is replaced by its measured
   row at that point.
2. Fused attention kernel. Deep v2 scope named a fused
   softmax or attention kernel; v1.0 wrote the fused softmax.
   The attention variant (fused score, softmax, and value
   product in one kernel) is v1.1 work.
3. GoogleTest or Catch2 migration. Build and Deep C++ tests
   use a self-contained assertion runner (a named deviation
   from SPEC.md). Migrating to a real framework, with the
   dependency vendored or fetched in CI, closes that
   deviation.
4. CUPTI profiler rows. When the node's CUPTI initialisation
   is reliable, re-run profiler_bench.py so
   results_profiler.json carries CUDA-activity kernel rows
   (names, counts, CUDA times) next to the op counts.
5. torch.compile route. When a working Triton or a C
   compiler exists on the GPU node, add the compiled softmax
   rows to the profiler study.
6. Second device. Every GPU number in v1.0 is one laptop GPU
   (RTX 5050). A second device, even a free-tier cloud GPU,
   would separate device facts from method facts.
7. Weakly ordered platform run. The atomics and handoff
   example on an ARM machine, where relaxed ordering hazards
   are observable rather than argued.
8. Larger and longer benchmarks. More sizes, longer timing
   windows, and clock logging (nvidia-smi during the run) to
   bound the variance named in the Deep failure modes.

## Standing rules for v1.1 work

- No custom kernel number is published before the kernel
  compiles and runs on a named device with a named toolkit
  version.
- Every new measured row lands in a results file first and is
  verified by recomputation before any README quotes it.

## v1.1 backlog amendment 2026-11-06: GPU first-principles spine

Decision (Jesun's order, 2026-11-06): the BEFORE-GPU-ENGINEERING first-principles curriculum is adopted two ways at once. It becomes the study spine for this repo's v1.1 work, and it gets its own study repo, gpu-first-principles-lab (https://github.com/JesunAhmadUshno/gpu-first-principles-lab), where each concept is restated in original words, verified against a primary source, implemented, measured on the RTX 5050, and closed with a published verdict.

Gate rule: a v1.1 item below opens only when the gpu-first-principles-lab unit that feeds it has a measured verdict. Reading alone opens nothing. Each item lands as a pull request carrying before and after numbers measured on the named device, under the standing rules above.

Items:

- (a) Coalescing upgrade, fed directly by unit U11 (global memory and coalescing, deep), with micro-study M1 from U02 as background. The v1.0 copy and access-pattern kernels are revisited against the U11 verdicts and re-measured.
- (b) Occupancy tuning pass, fed by verdicts from U03 (occupancy sweep, micro-study M2) and U10 (register physics). Block sizes and launch bounds in the v1.0 kernels are tuned against the measured occupancy behaviour, not against the maximum-occupancy heuristic.
- (c) Tiled GEMM progression, fed directly by U12 and U13 (micro-studies M3b and M3c). The matmul line advances naive to tiled to tuned, with the tuned step reported honestly against the library call, exactly as M3c measures it.
- (d) Register-spilling case study, fed by U10. A kernel variant that spills is induced or identified, the cost is measured, and the fix is landed with the before and after rows side by side.

Cross-reference: the study map, the frozen protocols (M1, M2, M3a to M3c), and every feeding verdict live in gpu-first-principles-lab. This repo stays the applied record; that repo stays the study record. Nothing is built twice.

## Fan-out

Written fan-out set for v1.0, dated 2026-11-05, in fanout/:
blog post (fanout/blog-post.md), step-by-step tutorial
(fanout/tutorial.md), website page copy (fanout/website-page.md),
LinkedIn post draft, not posted (fanout/linkedin-post.md), paper
candidate brief for the fused against unfused profiler note,
nothing submitted (fanout/paper-candidate.md), and draft release
notes for v0.1, v0.5, and v1.0 (fanout/release-notes.md). Channel
outcomes are logged in PUBLICITY.md at the repo root.

## Novelty

PROPOSED. See NOVELTY.md. The novelty text is a draft until the Jev claim_check gate passes on it.

## Reading order (moved from the README)

1. Read SPEC.md for scope and acceptance criteria.
2. Read DOSSIER.md for the past, present, and future of this problem (v1).
3. Read NOVELTY.md for the proposed novelty (DRAFT, pending the Jev claim_check gate).
4. Start at level-1-learn/README.md, then level-2-build, then level-3-deep.
5. Read ROADMAP.md for the v1.1 backlog beyond v1.0.
