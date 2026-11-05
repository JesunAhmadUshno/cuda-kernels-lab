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

Fan-out set: NOT STARTED yet. The written fan-out set (blog,
tutorial, website page copy, LinkedIn draft, release notes) is
Sprint 2 exit work for this project. No fan-out piece exists yet,
none is graded here, and no channel outcome is claimed for it.

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
