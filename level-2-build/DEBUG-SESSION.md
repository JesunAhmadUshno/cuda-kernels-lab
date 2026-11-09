# Debugging session: tiled matmul heap overflow (Level 2 Build)

Date: 2026-11-05. Machine: the Muse build VM, g++ 13.3.0.

Method note: this session is sanitizer-led, and says so plainly.
gdb is not installed on the build machine (verified: no gdb
binary in PATH), so no gdb or LLDB session was possible here. The
crash was found, located, and verified fixed with
AddressSanitizer and UndefinedBehaviorSanitizer instead, which is
the stronger tool for this class of bug: it names the exact write,
the allocation it overran, and both stack traces. The SPEC outcome
(one debugging session written up from a real crash) is met; the
tool substitution is named here, not hidden.

## The code under test

cpp/buggy.cpp is a variant of the tiled matmul in cpp/tool.cpp.
One change was made on purpose: the bounds guard was removed from
the tile loop. The subject case is n = 3 at tile = 2. Three is not
a multiple of two, so the edge tile covers row and column index 3,
which does not exist in a 3 by 3 matrix.

## Symptom

Built plain, the buggy program can appear to work: a heap write
just past an allocation often lands in slack space and the wrong
values are stored where nothing reads them back in this small
demo. Built with sanitizers and run, it aborts before printing
anything:

    g++ -std=c++17 -fsanitize=address,undefined -g cpp/buggy.cpp -o buggy
    ./buggy        # exit code 1, AddressSanitizer abort, no output

## The sanitizer report (real, captured 2026-11-05)

Excerpt, with the address and process id from the actual run. The
middle shadow byte dump is trimmed; the error line, the write
stack, the allocation note, the allocation stack, and the SUMMARY
line are verbatim.

```
=================================================================
==5260==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x5070000001b8 at pc 0x5fc669cfe913 bp 0x7ffcc5ea73c0 sp 0x7ffcc5ea73b0
WRITE of size 8 at 0x5070000001b8 thread T0
    #0 0x5fc669cfe912 in MatmulTiledBuggy /home/hatch/workspace/roadmap-v3/cuda-kernels-lab/level-2-build/cpp/buggy.cpp:78
    #1 0x5fc669cff3a0 in main /home/hatch/workspace/roadmap-v3/cuda-kernels-lab/level-2-build/cpp/buggy.cpp:91
    #2 0x717f1ff1f1c9  (/lib/x86_64-linux-gnu/libc.so.6+0x2a1c9)
    #3 0x717f1ff1f28a in __libc_start_main (/lib/x86_64-linux-gnu/libc.so.6+0x2a28a)

0x5070000001b8 is located 0 bytes after 72-byte region [0x507000000170,0x5070000001b8)
allocated by thread T0 here:
    #0 0x717f20bb36c8 in operator new[](unsigned long) ../../../../src/libsanitizer/asan/asan_new_delete.cpp:98
    #1 0x5fc669cfdcac in MatmulTiledBuggy /home/hatch/workspace/roadmap-v3/cuda-kernels-lab/level-2-build/cpp/buggy.cpp:45
    #2 0x5fc669cff3a0 in main /home/hatch/workspace/roadmap-v3/cuda-kernels-lab/level-2-build/cpp/buggy.cpp:91

SUMMARY: AddressSanitizer: heap-buffer-overflow /home/hatch/workspace/roadmap-v3/cuda-kernels-lab/level-2-build/cpp/buggy.cpp:78 in MatmulTiledBuggy
==5260==ABORTING
```

Reading the report: a WRITE of size 8 (one double) at buggy.cpp
line 78 in MatmulTiledBuggy, landing 0 bytes after a 72-byte
region. 72 bytes is 9 doubles: the 3 by 3 result matrix allocated
at buggy.cpp line 45 with new double[n * n]. The first overrunning
write is therefore c[9], the store for the edge tile position
gi = 3, gj = 0 (index 3 * 3 + 0 = 9).

## Root cause

The tile loop iterates every position of the tile, including
positions outside the matrix at an edge, and the store step

    c[gi * n + gj] = acc[i * tile + j];

runs with no check that gi and gj are inside the n by n result.
Tile loads have the same hazard on the read side, which is why
both the loads and the store in correct tiled kernels are guarded:
a padded position contributes zero to the product and is never
stored. Remove the guard and any size that is not a multiple of
the tile writes past the result buffer. Sizes that divide evenly
never trigger it, which is why the 2x2 and 4x4 cases pass and the
bug hides until an edge size is tested.

## The fix

Restore the guard, in exactly the form the correct code carries:

- cpp/tool.cpp, MatmulTiled store step:
  `if (gi >= n || gj >= n) continue;  // bounds guard (the fix)`
  and the matching guarded loads,
  `(gi < n && gj < n) ? a[gi * n + gj] : T{0}`.
- cuda/kernels.cu, matmul_tiled_kernel loads:
  `(row < n && acol < n) ? a[row * n + acol] : 0.0f` and
  `(brow < n && col < n) ? b[brow * n + col] : 0.0f`, with the
  store under `if (row < n && col < n)`.

buggy.cpp itself is left buggy on purpose, as the session record
and a regression exhibit. The shipped tool is tool.cpp.

## Fixed run evidence

tool.cpp, fixed, built and run under both sanitizers on
2026-11-05:

    g++ -std=c++17 -fsanitize=address,undefined -g cpp/tool.cpp -o tool_asan
    ./tool_asan

Output, exit code 0, no sanitizer report of any kind:

```
vector_add: PASS
matmul_tiled_2x2: PASS
matmul_tiled_3x3_tile2_edge: PASS
softmax_stable_large: PASS row_sum=1
tool: ALL PASS
```

The full C++ test runner (cpp/tests.cpp, 16 tests, including the
3x3 tile 2 and 5x5 tile 2 and tile 4 edge cases) was also built
with -fsanitize=address,undefined and ran clean: 16 passed, 0
failed, exit code 0, no sanitizer output.

## Lesson carried to Deep

Edge sizes are where tile bugs live. Every tiled kernel test in
this repo includes at least one size that is not a multiple of
the tile, and Deep opens with the full tool under ASan and UBSan
plus the atomics and memory ordering example (see PROPOSAL.md).
