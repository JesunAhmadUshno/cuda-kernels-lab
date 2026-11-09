# 0002. Publish PyTorch-route numbers only while the custom kernels are unrun

Date: 2026-11-05. Status: accepted (recorded 2026-11-08 from
PROPOSAL.md, Level 3 Deep spec).

## Context

The fleet GPU node has no CUDA toolkit (nvcc absent, CUDA_HOME None),
so kernels.cu and fused_softmax.cu cannot be compiled. Installing a
toolkit is an Owner call that was not made.

## Decision

Every GPU row in the repo is a PyTorch route, labelled exactly
(native op, unfused manual, compiled). No row is labelled custom, and
no custom-kernel number, timing or correctness claim appears anywhere
until the kernel compiles and runs on a named device with a named
toolkit version. The fused-kernel figure in ANALYSIS.md is labelled a
prediction.

## Consequences

The project's central comparison (own kernel against PyTorch) has no
data, which caps the verdict at Draft. Lifting it needs the toolkit and
a locked proposal for the re-run.
