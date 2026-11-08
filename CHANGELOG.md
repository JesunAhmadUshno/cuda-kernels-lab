# Changelog

All notable changes to this project. Format: Keep a Changelog 1.1.
Versions follow Semantic Versioning. The tags v0.1, v0.5 and v1.0 mark
the old Learn, Build and Deep levels; they predate the portfolio
standard v2 and are not a "v1.0 Released" verdict under it.

## [Unreleased]

### Changed
- 2026-11-08: Restructured to the portfolio standard v2. README is the
  abstract (verdict: Draft, not pre-registered). PROPOSAL.md (not
  locked) replaces SPEC.md, NOVELTY.md, DOSSIER.md and the Build and
  Deep specs; REPORT.md replaces the old README body,
  docs/research-review.md and ROADMAP.md. Grades and rubric scores
  removed from every public file and the dashboard; grades are kept
  internal. Decision records added in docs/decisions/.
- 2026-11-08: CI gains read-only token permissions, a layout and
  no-grade test, and a check that results_learn.json and
  results_build.json regenerate unchanged. CodeQL covers Python,
  JavaScript and C++; Dependabot watches site/ npm.

### Added
- 2026-11-07: Live dashboard in site/, built from the results files by
  the Pages workflow, with a known-answer check before every build.
- 2026-11-06: v1.1 backlog amendment: the GPU first-principles spine
  (gpu-first-principles-lab), now in REPORT.md.

## [1.0] - 2026-11-05

### Added
- Level 3 Deep: profiler study of PyTorch routes on the RTX 5050
  (9 rows verified by recomputation), C++ atomics and move benchmarks,
  15 C++ tests, ASan and UBSan clean, fused_softmax.cu written (not
  compiled or run).
- Actions polish pack (CI matrix, CodeQL, release, labeler, greetings,
  stale) and the fan-out drafts in fanout/.

## [0.5] - 2026-11-05

### Added
- Level 2 Build: C++ tool with 16 tests, an AddressSanitizer debugging
  session, 4 PyTorch GPU rows verified, kernels.cu written (not
  compiled or run).

## [0.1] - 2026-11-05

### Added
- Level 1 Learn: CPU reference kernels with exact load counts, stable
  softmax, C++ RAII bench tool.
