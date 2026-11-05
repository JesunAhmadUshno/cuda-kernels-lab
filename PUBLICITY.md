# PUBLICITY: cuda-kernels-lab (P04)

Publicity log under the documentation and publishing standard,
Section 7. Every finished level produces its written fan-out set,
each piece graded on the 1,000-point publicity rubric (accuracy 300,
research 200, gap fit 200, clarity 150, reach format 150), ship line
900. Drafts are prepared by the loop. Actual social posting follows
the standing platform rules and gates. The loop itself never posts
to social.

Project state: v1.0, Learn, Build, and compressed Deep done
2026-11-05, 46 Python tests passing. Level grades in the level
READMEs: Learn 8,300 of 10,000, Build 8,200 of 10,000, Deep 7,900
of 10,000.

Fan-out set date: 2026-11-05 (Sprint 2 early work). Project state:
v1.0, Learn, Build, and compressed Deep done 2026-11-05, 46 Python
tests passing, 31 C++ tests passing. Every GPU benchmark row in
this project is a PyTorch native op, labelled as such everywhere.
The custom CUDA kernels are written but NOT compiled and NOT run
(no nvcc on the fleet GPU node, BlackBox); that limit is carried
in every fan-out piece, not footnoted away.

## Channel log (2026-11-05)

| Channel | Outcome on 2026-11-05 |
| --- | --- |
| GitHub releases | Live: v0.1 https://github.com/JesunAhmadUshno/cuda-kernels-lab/releases/tag/v0.1 , v0.5 https://github.com/JesunAhmadUshno/cuda-kernels-lab/releases/tag/v0.5 , v1.0 https://github.com/JesunAhmadUshno/cuda-kernels-lab/releases/tag/v1.0 (verified with gh release list, 2026-11-05). Draft release notes for all three are in fanout/release-notes.md. No new release was created for the fan-out. |
| Portfolio index | Held: fan-out links are ready, the index update is a coordinator step outside this cycle. Named reason: held for index refresh. |
| Website page | Draft in fanout/website-page.md. Held: not published to jesunahmadushno.com. |
| Blog | Draft in fanout/blog-post.md. Held: not posted. |
| Tutorial | In repo: fanout/tutorial.md. Every command in it was run and verified on 2026-11-05 (all three level runners, three per-level test discoveries, the C++ bench, Build, and Deep compile and run steps including the sanitizer steps, the full suite at 46 tests, and the smoke test). |
| LinkedIn post | Draft in fanout/linkedin-post.md, marked DRAFT NOT POSTED, link carried as the first-comment line per the LinkedIn rule. Held: not posted, loop never posts. |
| X | Draft held: no X copy is shipped in this set beyond the LinkedIn draft source text. Held under the standing social rules; posting is outside the loop. |
| Facebook | Draft held: no Facebook copy posted. Standing rule applies (no link in body or comments, keyword call to action fulfilled by DM). Held: not posted, loop never posts. |
| Instagram | Draft held: no Instagram copy posted. Standing rule applies (no clickable link in caption, link in bio). Held: not posted, loop never posts. |
| Threads | Draft held: no Threads copy posted. Threads waits for Jesun's tap per post, hash-locked on exact text and media. Held: awaiting Jesun's tap. |
| Video | Held: engine queue, January production for lab projects per the December Compression Protocol. No video was produced in this cycle. |
| Book chapter | Draft angle in fanout/blog-post.md and SPEC.md ("Down To The Metal"). Held: chapter draft follows the finished project set; not a published chapter. |
| Paper candidate | Draft brief in fanout/paper-candidate.md (fused against unfused profiler note). Submission is Jesun's tap. Nothing submitted, no venue contacted. The publications tracker is updated by the coordinator, not by this cycle. |

## Jev gate (2026-11-05)

claim_check on the blog post Main results claim paragraph,
extracted byte-exact programmatically from fanout/blog-post.md.
Model jev-1.13.0, question id supported, noul 0.76, route REVIEW
(clear at 0.85 or above, block at 0.10 or below, review between).
Usage: 2057 input tokens, 20 output tokens. Arithmetic
pre-computed by code before the gate: fused ratio 0.2163 divided
by 0.0555 equals 3.897, stated as 3.9; atomics total 4 times
250000 equals 1000000. Spec and results are saved in the goal
hidden_files as jev-p04-fanout-claim-spec-2026-11-05.json and
jev-p04-fanout-claim-results-2026-11-05.json. REVIEW means the
claim ships as a reviewed draft piece, not as a cleared claim;
no part of the paragraph was blocked.

## Publicity grades (1,000-point rubric, ship line 900)

| Piece | Accuracy /300 | Research /200 | Gap fit /200 | Clarity /150 | Reach format /150 | Total /1000 |
| --- | --- | --- | --- | --- | --- | --- |
| fanout/blog-post.md | 295 | 190 | 195 | 145 | 140 | 965 |
| fanout/tutorial.md | 300 | 185 | 195 | 148 | 142 | 970 |
| fanout/website-page.md | 295 | 180 | 190 | 148 | 145 | 958 |
| fanout/linkedin-post.md | 290 | 175 | 190 | 145 | 140 | 940 |
| fanout/paper-candidate.md | 295 | 195 | 195 | 140 | 130 | 955 |
| fanout/release-notes.md | 298 | 185 | 185 | 145 | 135 | 948 |

All six pieces are at or above the 900 ship line. No piece was
shipped ungraded. Overall fan-out grade: 9,560 of 10,000 (the
mean of the six piece grades, 956 of 1,000, scaled; deductions
below are the per-piece named deductions, not rounded away).

## Named deductions (unclosable in this cycle)

- Blog post, minus 35: research is condensed from DOSSIER.md v1,
  whose prior art section is brief (minus 10 research); every GPU
  result is a PyTorch native op on one laptop GPU, which limits
  how far the results travel, stated in the piece (minus 10
  reach, minus 5 accuracy for reader interpretation risk, minus
  5 gap fit, minus 5 clarity).
- Tutorial, minus 30: research framing is thinner than the blog
  by design, because the module teaches runs rather than the
  field (minus 15 research); the module verifies stored GPU rows
  by recomputation and cannot re-measure them or compile the
  custom kernels (minus 5 gap fit, minus 2 clarity, minus 8
  reach).
- Website page, minus 42: condensed research (minus 20),
  condensed gap framing (minus 10), and the page is copy only,
  not a live page (minus 5 reach, minus 5 accuracy, minus 2
  clarity).
- LinkedIn post, minus 60: short form compresses the research
  and the failure mode detail (minus 25 research, minus 10
  accuracy, minus 10 gap fit, minus 5 clarity, minus 10 reach,
  because the post is a held draft and carries no live link in
  the body by rule).
- Paper candidate, minus 45: this is a brief, not a drafted
  paper (minus 20 reach, minus 10 clarity); venue is classed,
  not named (minus 5 research, minus 5 gap fit, minus 5
  accuracy, because the SPEC own-kernel angle cannot be claimed
  until the kernel runs in v1.1).
- Release notes, minus 52: notes summarise releases that
  already exist, so research and gap framing are summary grade
  (minus 15 research, minus 15 gap fit); format is repo-bound
  (minus 15 reach, minus 2 accuracy, minus 5 clarity).

## Actions polish pack (2026-11-05)

Applied through a pull request on 2026-11-05, following the Actions
polish pack standard. The pack adds: CI on a Python 3.11 and 3.12
matrix with a ruff lint step (all pre-existing P04 CI steps
preserved: scaffold smoke test, Learn tests and runner, C++ bench
tool compile and self check with g++, Build tests and runner,
Build C++ tests compile and run plus the ASan tool run, Deep tests
and runner, Deep C++ tests compile and run plus the ASan deep tool
run with -pthread), CodeQL scanning (Python is the analysed
surface; the C++ is built in CI), Dependabot for pip and GitHub
Actions, a release workflow that attaches a source tarball named
cuda-kernels-lab-<tag>.tar.gz to v* tag releases, Labeler,
Greetings (snake_case inputs for actions/first-interaction@v3:
repo_token, issue_message, pr_message), Stale in comment-only
mode (days-before-close: -1, never closes), and a docs summary
page at docs/index.md for GitHub Pages. Ruff scope is recorded in
ruff.toml (E, F, I, line-length 120). Ruff fixes in this pull
request were style only: 5 initial errors (3 import order, 1
unused import, 1 module import not at top from the sys.path
pattern), fixed by ruff fix output and a noqa E402 marker on the
path-insert import in tests/test_learn_kernels.py, with no change
to behaviour. requirements.txt records the standard-library-only
fact for the levels, tests, and runners (with the torch exception
for the GPU-node profiler script stated in that file) so
Dependabot pip has a manifest to watch. The docs page states
plainly that the GPU rows are PyTorch native ops and that the
custom CUDA kernels are written but not compiled or run, because
nvcc is absent on the fleet GPU node.

| Channel | Outcome on 2026-11-05 |
| --- | --- |
| GitHub Pages | Docs page added at docs/index.md in this pull request. Pages enablement and live build status are verified after merge and recorded in the batch ledger; no live claim is made in this file before that check. |
| PyPI publishing | Held: needs a PyPI token, a new credential. Jesun's tap only. Not set up. |
| Social channels (LinkedIn, X, Facebook, Instagram, Threads) | Held: no posts made. The loop never posts to social; Threads waits for Jesun's tap per post under the standing gates. |

Publicity grade for this polish pack log piece, on the 1,000
point rubric (accuracy 300, research 200, gap fit 200, clarity
150, reach format 150), ship line 900:

| Piece | Accuracy /300 | Research /200 | Gap fit /200 | Clarity /150 | Reach /150 | Total /1000 |
| --- | --- | --- | --- | --- | --- | --- |
| PUBLICITY.md polish pack section (2026-11-05) | 295 | 180 | 195 | 145 | 100 | 915 |

At or above the 900 ship line. Named deductions: research minus
20, because the pack is a standard applied across repos and this
section records its application rather than new research; reach
minus 50, because the docs page is copy in this pull request with
no live Pages claim yet, PyPI publishing is held, and no social
channel carries this piece by the standing gates; accuracy minus
5, gap fit minus 5, and clarity minus 5 for summary compression
of the full pack standard, which lives in the batch files.
