# SPEC: cuda-kernels-lab (P04)

Phase skill: spec-driven-development (house engineering pipeline).
Status: Wave 0 scaffold spec, from Roadmap v2 text and the v3 decision pack. Scope and acceptance criteria below are the v2 section text, kept as the source of truth until the project's wave opens its own spec pass.

Fleet node: BlackBox (GPU). Wave: Wave 2.
Hiring evidence: 4 of 5 (CUDA in 12/56) [confirmed, fleet and scoring pass 2026-11-04, or the decision pack for P23 to P29].

## Project section text (Roadmap v2, or the v3 candidate text for P23 to P29)

cuda-kernels-lab: EXPAND

Subject: GPU programming.
v1 core [confirmed, v1]: CUDA kernels from vector add to tiled matrix multiply; Deep writes a fused softmax or attention kernel and profiles it against PyTorch.
Added track, the standalone C++ tool (register items 71 to 76, 126, 127, 135):
- Learn: grow the kernel host code into one small standalone C++ tool (a kernel launch and benchmark runner): smart pointers and RAII, templates and the STL, CMake build.
- Build: GoogleTest or Catch2 tests on the tool; one gdb or LLDB debugging session written up from a real crash.
- Deep: AddressSanitizer and UBSan pass over the tool with findings fixed; one atomics and memory-ordering example, measured; move semantics shown in one benchmark note.
Fan-out: Book chapter "Down To The Metal". Paper angle: the fused kernel benchmark note (own kernel against PyTorch, profiler output). Course module: P04-cuda-kernels-lab.

## Acceptance criteria (scaffold level)

- Learn closes when the level-1-learn code runs, its README teaches the idea, and a fresh-eyes reader can follow it.
- Build closes when the system runs end to end, tests pass, and one benchmark or result table with method is in the README.
- Deep closes when the Deep scope in the section text is done, the failure modes are written, and the dossier is updated to ship state.
- No level closes on a claim. Evidence only: code, tests, numbers, links.
