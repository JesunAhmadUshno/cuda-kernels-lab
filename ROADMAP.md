# ROADMAP: cuda-kernels-lab (P04), v1.1 backlog

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
- Grades are re-issued per level with itemized deductions;
  v1.0 grades are not edited retroactively.
