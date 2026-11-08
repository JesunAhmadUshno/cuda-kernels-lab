# Custom fused softmax kernel: analysis (Level 3 Deep)

Status: the kernel in fused_softmax.cu is a written component
only. It has NOT been compiled and has NOT run. This file is the
written analysis the Deep spec requires for it: why it cannot
compile on the current node, and what profile it is expected to
show when it can.

## Why it cannot compile on the current node

Verified on the fleet GPU node (BlackBox) on 2026-11-05, in the
Build cycle and again in this Deep cycle (the profiler run
header in ../results_profiler.json carries the live fields):

- nvcc is not on PATH on the node (nvcc_on_path: false in the
  profiler output; the Build cycle probe found no CUDA toolkit
  in PATH or CUDA_PATH).
- torch reports CUDA_HOME as None (cuda_home: null in the
  profiler output), so even the PyTorch extension route has no
  toolkit to invoke.
- The Numba route was tried at Build and failed: NVVM reported
  no supported compute capability for the 12.0 device.
- Installing a CUDA toolkit on the node is the owner call. It
  was not authorized for this cycle, so no install was attempted.

Consequence, stated plainly: there are no custom kernel timings,
no custom kernel correctness results, and no occupancy numbers
for fused_softmax.cu anywhere in this repo.

## What the kernel does differently

The Build softmax_rows_kernel (level-2-build/cuda/kernels.cu)
re-reads the global input three times: once for the max, once
for the exp sum, and once to write the normalised output. The
Deep fused kernel caches exp(x - row_max) in shared memory on
the second pass and normalises from that cache, so the global
input is read twice and never a third time. Shared memory per
block is 5 KiB (4 KiB exp cache at 1024 columns, 1 KiB reduction
scratch), under the 48 KiB default limit.

## Expected profile argument

This is an argument about the shape of the profile, written
before any measurement, to be tested when the kernel runs
(v1.1 backlog item 1). It is not a result.

- Launch shape: exactly one kernel launch per softmax call,
  one block per row, 256 threads per block. In a profiler table
  it should appear as a single kernel row per call, like the
  native fused op (one aten::_softmax op per call in the Deep
  profiler op counts), and unlike the unfused manual sequence
  (five aten ops per call in the same counts, several kernel
  launches, and a measured median 3.9 times the fused op at
  4096 by 1024 on this node).
- Memory traffic: the theoretical minimum for softmax is one
  read and one write of the matrix (2 times rows times cols
  times 4 bytes), which the native fused op approaches at
  604.58 GB/s measured. This kernel reads the input twice, so
  its honest traffic model is 3 times rows times cols times 4
  bytes; at the native op's achieved bandwidth that predicts a
  median near 0.083 ms at 4096 by 1024, between the native fused
  op (0.0555 ms) and the unfused sequence (0.2163 ms). The v1.1
  run must replace this paragraph with the measured row, in the
  results_profiler.json shape, with route labelled custom.
- Correctness bar when it runs: max abs difference against a
  CPU stable softmax reference under 1e-5, the same bar the
  Build harness set for softmax_rows_kernel, plus an edge case
  at a row length that is not a multiple of the block size.

## Failure modes specific to this kernel

- Rows over 1024 columns overflow the shared exp cache. The
  kernel is specified for cols up to 1024 only; the harness
  must reject larger rows instead of launching.
- The reduction assumes blockDim.x is a power of two (256).
  Launching with any other block size silently corrupts the
  max and sum reductions.
- expf and fmaxf are device intrinsics with float precision;
  the 1e-5 correctness bar is set for float32, and a float64
  variant would need its own bar.
