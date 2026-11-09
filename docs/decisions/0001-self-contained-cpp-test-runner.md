# 0001. Self-contained C++ test runner instead of GoogleTest or Catch2

Date: 2026-11-05. Status: accepted (recorded 2026-11-08 from
PROPOSAL.md, Level 2 Build spec).

## Context

The scaffold spec named GoogleTest or Catch2 for the C++ tool's tests.
Neither was installed on the build machine, and the repo vendors no
third-party code.

## Decision

The Build and Deep C++ tests use a small assertion runner in
level-2-build/cpp/tests.cpp and level-3-deep/cpp/tests_deep.cpp,
compiled with plain g++ in CI.

## Consequences

No dependency to fetch or pin; tests build anywhere g++ exists. The
deviation from the spec is named, and migrating to a real framework is
item 3 of the v1.1 backlog in REPORT.md.
