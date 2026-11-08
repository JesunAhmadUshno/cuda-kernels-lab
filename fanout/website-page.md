# Website project page copy: cuda-kernels-lab

Site: jesunahmadushno.com
Status: COPY ONLY, draft for the project page. Dated 2026-11-05.
Nothing on any website was changed by this file.

## Headline

cuda-kernels-lab: GPU kernels, C++ tooling, and a profiler study, measured in the open

## Summary

cuda-kernels-lab is an open GPU programming project built by
Jesun Ahmad Ushno as project P04 in Roadmap v3. It builds the
kernel stack in three levels: CPU reference kernels with exact
operation and load counts and a standalone C++ bench tool,
a C++ tool with tests and a real sanitizer debugging session
plus GPU benchmarks on the fleet GPU node, and a profiler
study that separates a fused softmax op from an unfused
sequence, with C++ atomics, memory ordering, and move
semantics benchmarks. One rule is stated on every page of the
repo: every GPU benchmark row is a PyTorch native op, labelled
as such. The custom CUDA kernels are written but were NOT
compiled and did NOT run, because the fleet GPU node has no
CUDA toolkit.

Status: v1.0, 2026-11-05. Learn, Build, and compressed Deep
are done. Total Python tests: 46, all passing at v1.0. C++
tests: 31 in total (16 Build, 15 Deep).

## Results

All GPU results below are PyTorch native ops measured on the
fleet GPU node (NVIDIA GeForce RTX 5050 Laptop GPU, compute
capability 12.0, torch 2.11.0+cu128, CUDA events, median),
recorded in the level READMEs in the GitHub repo and verified
by recomputation within 1 percent. The Build rows and the
Deep profiler rows come from different runs on the same
device and are kept separate.

| Level | Set | Headline result | Notes |
| --- | --- | --- | --- |
| Learn | CPU reference kernels, exact counts | Tiled loads 64 against 128 naive at 4 by 4 tile 2; 512 against 1024 at 8 by 8 tile 2; 2048 against 8192 at 16 by 16 tile 4 | 5 of 5 correctness checks; 12 tests; no GPU number claimed at this level |
| Build | PyTorch native ops on the fleet GPU node | Vector add 381.3 GB/s; matmul 6764.67 GFLOP/s at 1024; softmax rows 848.71 GB/s at 4096 by 1024 | 4 rows verified; 18 tests (30 total); 16 C++ tests; real ASan heap overflow found and fixed |
| Deep (compressed) | Profiler study, PyTorch routes, 9 rows | Matmul native op 7642.29 GFLOP/s at 1024; softmax native fused op 604.58 GB/s against 155.13 GB/s unfused at 4096 by 1024 (factor 3.9) | 16 tests (46 total); 15 Deep C++ tests (31 C++ total); atomics exact at 1000000; handoff 200 of 200; moves about 24x faster than copies on the recorded run |

## Honest limits

- Every GPU number is a PyTorch native op. No custom kernel
  number exists in v1.0.
- The custom CUDA kernels (level-2-build/cuda/kernels.cu and
  level-3-deep/cuda/fused_softmax.cu) are written but NOT
  compiled and have never run: no nvcc on the fleet GPU node,
  torch reports CUDA_HOME as None there.
- torch.compile was attempted and failed on that node, and
  CUPTI CUDA-activity rows were unavailable in the committed
  profiler runs. The profiler evidence is op counts plus
  CUDA-event tables.
- GPU numbers are one laptop GPU, with named clock variance
  across runs. Ratios inside one run are the stable finding.
- Compressed Deep is not the full v2 Deep scope. The
  remainder is the v1.1 backlog in the repo REPORT.md,
  item 1 being the custom kernel compile and run.

## Links

- GitHub repo: https://github.com/JesunAhmadUshno/cuda-kernels-lab
- Release v0.1 (Learn): https://github.com/JesunAhmadUshno/cuda-kernels-lab/releases/tag/v0.1
- Release v0.5 (Build): https://github.com/JesunAhmadUshno/cuda-kernels-lab/releases/tag/v0.5
- Release v1.0 (Learn, Build, compressed Deep): https://github.com/JesunAhmadUshno/cuda-kernels-lab/releases/tag/v1.0
- Tutorial: fanout/tutorial.md in the GitHub repo
- Blog post draft: fanout/blog-post.md in the GitHub repo

## License note

Code: Apache-2.0. Tutorials: CC BY 4.0.
