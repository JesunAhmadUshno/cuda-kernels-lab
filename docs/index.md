# cuda-kernels-lab

CUDA kernels from vector add to a fused attention kernel, plus a standalone C++ benchmark tool. Built in three levels with measured results at each level.

- Learn: CPU reference kernels (vector add, naive and tiled matmul, stable softmax) with exact operation and load counts, plus a C++ bench tool.
- Build: C++ tool tests and a sanitizer debugging session, plus GPU benchmarks on the fleet GPU node.
- Deep (compressed): a profiler study on the fleet GPU node, C++ atomics and move benchmarks, and a written fused softmax kernel.

GPU benchmark rows below are PyTorch native ops, labelled as such. They are not custom kernel results.

- Build, PyTorch native ops: vector add 381.3 GB/s, and matmul 6,764.67 GFLOP/s at 1024.
- Deep profiler, PyTorch native op: matmul 7,642.29 GFLOP/s at 1024.
- Deep profiler, softmax at 4096 by 1024: native fused op 604.58 GB/s against 155.13 GB/s for the unfused manual sequence.

Custom CUDA kernels status, stated plainly: the custom CUDA kernels (level-2-build/cuda/kernels.cu and level-3-deep/cuda/fused_softmax.cu) are written but NOT compiled or run, because nvcc is absent on the fleet GPU node (no CUDA toolkit, torch reports CUDA_HOME None). No custom kernel timing or correctness result is claimed. Compiling and running them is v1.1 backlog item 1 in REPORT.md (v1.1 backlog).

Every number above is already published in this repo (README.md and the level write-ups) and reproduced or verified by the runners and analysis scripts in this repo, checked in CI. The full write-ups are in the level folders: level-1-learn, level-2-build, level-3-deep.

Releases: https://github.com/JesunAhmadUshno/cuda-kernels-lab/releases
