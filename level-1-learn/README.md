# Level 1, Learn: cuda-kernels-lab

Status: DONE, 2026-11-05. CPU reference edition. BlackBox, the GPU
fleet node, was unreachable this cycle, so no CUDA GPU ran and no GPU
timing or CUDA result is claimed anywhere in this level. The GPU runs
are Build and Deep scope and are named NEEDS-BLACKBOX-ON in the batch
ledger. What is built here teaches the same kernel structure on the
CPU, and correctness is the claim, not speed.

Scope for this level, from PROPOSAL.md: CUDA kernels from vector add to
tiled matmul, grown into one small standalone C++ tool (kernel launch
and benchmark runner): smart pointers and RAII, templates and the
STL, CMake build.

## What is here

- kernels.py (stdlib only): vector_add, matmul_naive, matmul_tiled
  (tile size parameter, tile loads counted), softmax_naive and
  softmax_stable (max subtraction). Each returns its result plus exact
  operation or load counts computed from the loop structure.
- run_learn.py: correctness against a plain independent reference on
  labelled small cases, plus the tiled vs naive load count table
  below. Writes results_learn.json deterministically: counts only,
  no wall time. Run twice, the file is byte identical (verified).
- cpp/bench_tool.cpp plus cpp/CMakeLists.txt: a standalone C++17
  tool with an RAII Buffer class owning memory through unique_ptr
  (smart pointers and RAII), a templated KernelRunner, templated
  vector add and tiled matmul kernels, and STL vectors. Its main
  checks its own results and prints PASS lines with operation counts.
  Compiler path that actually ran in tests and CI: g++ -std=c++17
  direct. The CMake build was also verified locally on 2026-11-05
  (g++ 13.3.0) and produced the same PASS output.

## Teaching map: grid, block, tile on the CPU

- Vector add is the one thread per element pattern. Thread i, or
  loop index i here, computes c[i] = a[i] + b[i]. A CUDA grid splits
  the index range across blocks and threads; the loop here walks the
  same range in order.
- Naive matmul is one thread per output element. Each thread walks a
  full row and a full column, reading global memory at every step:
  2 * M * N * K element loads with no reuse.
- Tiled matmul splits the output into tile by tile blocks. Each
  block cooperatively loads one tile of A and one tile of B into
  shared memory once, then every output in the block reuses them.
  In kernels.py the local a_tile and b_tile lists are the shared
  memory stand in, the block loops are the block indices, and
  tile_loads counts element slots loaded, including zero padded
  slots at the edges (a guarded full tile load).
- Softmax shows a numerical kernel lesson: exp overflows near 709,
  so the naive form raises OverflowError on inputs like 1000. The
  stable form subtracts the maximum first. Subtracting a constant
  from every input leaves softmax unchanged, and the largest term
  becomes exp(0) = 1, so nothing can overflow.

## Results (CPU reference counts only, measured 2026-11-05)

Correctness checks, all PASS (run_learn.py, 5 checks): vector add
small case, 2x2 matmul against hand values, 3x3 tiled at tile 2
against a plain reference, stable softmax on large inputs, and the
naive softmax overflow case. Unit tests: 12 pass, including
hand-computed 2x2 and 3x3 matmul in both naive and tiled form, tile
load counts at two sizes, and the C++ tool compile and self check.

| Size | Tile | Naive element loads | Tiled tile loads | Multiplies | Adds | Results identical |
| --- | --- | --- | --- | --- | --- | --- |
| 4 | 2 | 128 | 64 | 64 | 48 | yes |
| 8 | 2 | 1024 | 512 | 512 | 448 | yes |
| 16 | 4 | 8192 | 2048 | 4096 | 3840 | yes |

C++ tool self check: vector_add PASS ops=3, matmul_tiled_2x2 PASS
ops=8 tile loads, bench_tool ALL PASS.

Reading the table: tiling does not change the arithmetic (multiplies
and adds are identical). It changes the memory traffic. At 16 by 16
with tile 4 the tiled form loads 2,048 element slots against 8,192
naive loads, a 4x reduction, because each loaded element is reused
across a whole tile row or column. That reuse is the entire reason
tiled kernels are faster on a GPU, and the count version of it is
what this level proves.

## Failure modes and limits

- No GPU ran. Any claim about CUDA speed, occupancy, or real shared
  memory behavior is out of scope here and belongs to Build and
  Deep, which need BlackBox on.
- Counts are exact integers from the loop structure, not timings.
  CPU wall time would teach nothing about GPU performance and is
  deliberately not claimed.
- Tile load counts include zero padded edge slots. At sizes not
  divisible by the tile, padding makes the tiled count look worse
  than an in-bounds only count would; the table sizes divide evenly.
- The fixed tile arrays in the C++ tool cap the tile at 8. The
  Python version has no such cap. Lifting the cap is Build scope.
- Softmax naive is kept on purpose as the failure exhibit. It raises
  OverflowError on large inputs; production code uses the stable
  form only.

