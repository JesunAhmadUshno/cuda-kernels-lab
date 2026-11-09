# PUBLICITY: cuda-kernels-lab (P04)

Publicity log under the documentation and publishing standard,
Section 7. Every finished level produces its written fan-out set. Drafts are prepared by the loop. Actual social posting follows
the standing platform rules and gates. The loop itself never posts
to social.

Project state: v1.0, Learn, Build, and compressed Deep done
2026-11-05, 46 Python tests passing.

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
| Book chapter | Draft angle in fanout/blog-post.md and PROPOSAL.md ("Down To The Metal"). Held: chapter draft follows the finished project set; not a published chapter. |
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

