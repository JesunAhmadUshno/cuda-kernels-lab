# Research review: cuda-kernels-lab

Date: 2026-11-07. Scope: the whole repo at v1.0 plus the site/
dashboard added on branch site/live-dashboard. Graded against a
doctoral research rubric. Scale per criterion: 1 (absent) to 5
(publication ready). Every number quoted here is in a committed
results file or is recomputed from one by site/src/lab.check.mjs.

Rule used: every gap that can be closed without a GPU, a toolkit
install, paid services or long runs is closed in this change. Every
remaining gap names the exact compute or data it needs.

## Summary

| Criterion | Grade | One-line reason |
| --- | --- | --- |
| Research question | 3 | Clear in SPEC.md (own fused kernel against PyTorch), but the data answers a narrower question (PyTorch fused op against PyTorch unfused ops). |
| Hypotheses | 2 | ANALYSIS.md states one falsifiable prediction (about 0.083 ms for fused_softmax.cu at 4096 by 1024); no other hypothesis is written before measurement. |
| Method | 3 | CUDA events, warmup, median, correctness gate before timing, exact route labels. Single device, single session per number. |
| Baselines | 3 | Native fused op and unfused op sequence are honest baselines; the project's own kernel arm and the torch.compile arm are missing. |
| Statistics and uncertainty | 2 | One median per row, no dispersion, raw samples not stored. Cross-session drift is now quantified (below) but not bounded. |
| Reproducibility | 4 | Results files committed as produced, every throughput recomputed in Python and now in the site build, CI runs all CPU-side tests. GPU runs need the named node. |
| Threats to validity | 4 | Each level README names its failure modes; the dashboard now carries a limitations window. |
| Related work | 2 | DOSSIER.md covers the field informally; NOVELTY.md is a draft. A short positioning section is added below. |
| Limitations | 4 | Stated plainly in README, level write-ups and on the page; custom-kernel status is never blurred. |

Overall: a careful engineering record with honest labels, short of
a research result because the central comparison (own kernel
against PyTorch) has no data yet.

## Research question

SPEC.md: Deep "writes a fused softmax or attention kernel and
profiles it against PyTorch". What the data supports today: on one
RTX 5050 Laptop GPU, PyTorch's native fused softmax ran 3.9 times
faster than the same math as five separate PyTorch ops (0.0555 ms
against 0.2163 ms at 4096 by 1024, results_profiler.json). The
dashboard headline and status board say this, and say the lab's
own kernels are written but not compiled.

## Gaps closed in this change (no GPU needed)

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

## Remaining gaps and the exact compute or data each needs

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

## Related work

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

## Engineering notes

- Tests: 46 Python tests and 31 C++ tests pass locally; the site
  check is part of the site build.
- Site dependencies are exact-pinned (react 18.3.1, react-dom
  18.3.1, vite 6.4.5, @vitejs/plugin-react 4.3.4). A full
  `npm audit` in `site/` (dev dependencies included) reports 0
  vulnerabilities, and the published site is static files only.
- No secrets, no external scripts; fonts load from Google Fonts.
