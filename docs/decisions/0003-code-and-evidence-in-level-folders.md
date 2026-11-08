# 0003. Keep code and evidence in the level folders under standard v2

Date: 2026-11-08. Status: accepted.

## Context

Standard v2 suggests src/, experiments/ and results/. Here each level
folder holds its code next to the results file that code produced, and
the site, the tests and the CI read those files by path.

## Decision

Keep level-1-learn/, level-2-build/ and level-3-deep/ as domain-named
folders, listed in the README layout table, instead of splitting them.

## Consequences

No path churn and no broken links in the dated write-ups. The README
layout table and REPORT.md's results table carry the mapping from each
number to its evidence file.
