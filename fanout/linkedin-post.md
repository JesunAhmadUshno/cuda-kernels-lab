# LinkedIn post draft: P04 cuda-kernels-lab v1.0

Status: DRAFT NOT POSTED
Date drafted: 2026-11-05
Author: Jesun Ahmad Ushno
Rule for this platform: the link goes in the first comment, not in the
post body. The first comment line is at the end of this file.

## Post body

Why is a fused kernel faster? I wanted the count, not the slogan.
So I built the kernel stack in the open, and labelled every number
by the route that produced it.

cuda-kernels-lab v1.0 is done: Learn, Build, and a compressed Deep.
46 Python tests and 31 C++ tests, all passing.

Learn: CPU reference kernels with exact load counts. Tiled matmul
loads 2048 element slots against 8192 naive loads at 16 by 16 with
tile 4. Same arithmetic, a quarter of the memory traffic. That
reuse is why tiling wins on a GPU, proved here as a count. 12 tests.

Build: a standalone C++ tool with 16 tests, and a debugging session
I did not stage: a tiled matmul variant missing its bounds guard
wrote past its heap buffer, AddressSanitizer caught it, and the
fixed tool runs clean. GPU benchmarks on my fleet GPU node (RTX
5050, PyTorch native ops): vector add 381.3 GB/s, matmul 6764.67
GFLOP/s at 1024, softmax rows 848.71 GB/s.

Deep (compressed): a profiler study on PyTorch routes. Matmul
(native op) reached 7642.29 GFLOP/s at 1024. Softmax at 4096 by
1024: the native fused op hit 604.58 GB/s in 0.0555 ms, against
155.13 GB/s in 0.2163 ms for an unfused sequence of separate ops,
a factor of 3.9. The stage profile shows why: exp and div each
write a full intermediate to global memory, and the fused op
writes none. Plus the C++ track: an atomic counter exact at
1000000 under two orderings, a handoff at 200 of 200, and moves
about 24 times faster than copies on the recorded run.

The honest limit, in the post and not in a footnote: every GPU
number above is a PyTorch native op. My custom CUDA kernels,
including the fused softmax kernel, are written but were NOT
compiled and have NOT run, because the GPU node has no CUDA
toolkit (no nvcc). Compiling and running them is item 1 of the
v1.1 backlog, and no custom kernel speed is claimed until then.
Grades, with deductions named in the repo: Learn 8300, Build
8200, Deep 7900 of 10000.

If you write GPU code: what do you trust more, a profiler table
or a single headline GFLOP/s number?

## First comment (post this as the first comment, per the standing rule)

Repo, releases v0.1, v0.5, and v1.0, and the full tutorial: https://github.com/JesunAhmadUshno/cuda-kernels-lab
