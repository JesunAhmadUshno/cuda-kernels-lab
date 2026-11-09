"""Portfolio standard checks: the required files exist and no public file carries a grade.

No lock-hash test: this repo is not pre-registered (PROPOSAL.md is not locked).
Add a SHA-256 pin here when the confirmatory proposal is tagged.
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEED = ["README.md", "PROPOSAL.md", "REPORT.md", "CHANGELOG.md", "CITATION.cff", "SECURITY.md", "LICENSE",
        "docs/decisions/0001-self-contained-cpp-test-runner.md",
        "level-1-learn/README.md", "level-2-build/README.md", "level-3-deep/README.md",
        "level-1-learn/results_learn.json", "level-2-build/results_build.json",
        "level-2-build/results_torch_bench.json", "level-3-deep/results_profiler.json",
        "level-3-deep/results_deep.json"]
GRADE = re.compile(r"of 10,?000|/ ?10,?000|/1000\b|Grade:|^#+ .*\bGrades?\b|\bgrades? on record", re.I | re.M)


def public_files():
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    if not out:  # not a git checkout: walk the tree
        out = [os.path.relpath(os.path.join(d, f), ROOT) for d, _, fs in os.walk(ROOT)
               if ".git" not in d and "node_modules" not in d for f in fs]
    return [p for p in out if p.endswith((".md", ".jsx", ".js", ".html", ".cff")) and "node_modules" not in p]


def test_required_files_exist():
    missing = [p for p in NEED if not os.path.exists(os.path.join(ROOT, p))]
    assert not missing, "missing files: " + ", ".join(missing)


def test_grade_pattern_catches_grades():
    for bad in ("Learn 8,300 of 10,000", "Grade: 7900", "## Grade", "| Total /1000 |", "Grades on record"):
        assert GRADE.search(bad), bad
    assert not GRADE.search("atomic counter exactly 1000000")


def test_no_public_grade():
    hits = []
    for p in public_files():
        for i, line in enumerate(open(os.path.join(ROOT, p), encoding="utf-8"), 1):
            if GRADE.search(line):
                hits.append(f"{p}:{i}: {line.strip()}")
    assert not hits, "grades are internal only:\n" + "\n".join(hits)


def test_readme_abstract():
    text = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
    for key in ("**Question.**", "**Headline.**", "**Field.**", "**Job\nfamily.**", "**Verdict.**", "## Lifecycle"):
        assert key.replace("\n", " ") in text.replace("\n", " "), key


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
    print("layout and no-grade check: PASS")
