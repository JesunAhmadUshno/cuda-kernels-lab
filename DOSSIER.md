# DOSSIER: cuda-kernels-lab (P04) v1, 2026-11-04

Written under the documentation and publishing standard (past, present, future). Updated at ship time.

## Past

Prior art for this project is named in NOVELTY.md (vendor CUDA samples and PyTorch kernels). The history of the problem, the key papers and systems, and what was tried before are recorded in the Roadmap v2 section text in SPEC.md and will be expanded with dated sources at the project's wave, before Build opens.

## Present

Current demand evidence: hiring value 4 of 5 (CUDA in 12/56) [confirmed, fleet and scoring pass 2026-11-04; for P23 to P29, the decision pack 2026-11-04]. The Round 3 research (2026-11-04) found the market centre at post-training, evals, environments, and inference depth [confirmed as the research finding; the report is at ~/workspace/research_notes/roadmap-v3-revaluation-research-20261104-0949/report.md]. Tools and versions are pinned at Build time, not in this v1.

Update, Level 2 Build DONE 2026-11-05: the standalone C++ tool now carries vector add, tiled matmul, and stable softmax with a 16-test self-contained assertion runner (no external framework; Catch2 and GoogleTest were not fetched and nothing is vendored). A real debugging session is on file at level-2-build/DEBUG-SESSION.md: a tiled matmul variant missing its bounds guard produced a real AddressSanitizer heap-buffer-overflow WRITE at an edge size, and the fixed tool runs clean under ASan and UBSan (session is sanitizer-led; gdb is not installed on the build machine). GPU benchmarks on record are PyTorch native ops measured on the fleet GPU node (NVIDIA GeForce RTX 5050 Laptop GPU, compute capability 12.0, torch 2.11.0+cu128, CUDA events, median): vector add 381.3 GB/s, matmul 6764.67 GFLOP/s at 1024, softmax rows 848.71 GB/s; all 4 stored rows recompute within 1 percent. The custom CUDA harness level-2-build/cuda/kernels.cu is written but uncompiled and has never run: the GPU node has no CUDA toolkit in PATH or CUDA_PATH this cycle, torch reports CUDA_HOME None, and a Numba route failed on compute capability 12.0 support. Compiling and running it is the first Deep item. Build grade: 8,200 of 10,000, deductions named in level-2-build/README.md.

## Future

Open problems and the project's next steps sit in the Deep level scope in SPEC.md. Any forecast about this field is [unverified] unless a dated primary source is added here at the wave update.
