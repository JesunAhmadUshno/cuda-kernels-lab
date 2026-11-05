"""Level 2 Build runner.

1. Verifies results_torch_bench.json through analysis.py and
   prints the benchmark table (PyTorch native ops on the fleet
   GPU node, never custom kernel results).
2. Compiles and runs the C++ test runner (cpp/tests.cpp) when g++
   is available; skips with a named message when it is not.
3. Builds cpp/tool.cpp with AddressSanitizer and
   UndefinedBehaviorSanitizer and runs it; the fixed tool must
   exit 0 with no sanitizer output. Together with the captured
   report in DEBUG-SESSION.md (which must contain the sanitizer
   SUMMARY line), that sets the ASan session status.
4. Writes results_build.json summarising the run: C++ tests
   passed count, ASan session status, GPU rows verified count.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analysis  # noqa: E402

CPP_DIR = os.path.join(HERE, "cpp")
DEBUG_SESSION = os.path.join(HERE, "DEBUG-SESSION.md")
OUT = os.path.join(HERE, "results_build.json")


def run_cpp_tests():
    """Return the number of C++ tests passed, or None on skip."""
    gpp = shutil.which("g++")
    if gpp is None:
        print("g++ not available: C++ tests SKIPPED (named skip, "
              "not a pass)")
        return None
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "cpp_tests")
        build = subprocess.run(
            [gpp, "-std=c++17", "-O2", "-Wall", "-Wextra",
             os.path.join(CPP_DIR, "tests.cpp"), "-o", exe],
            capture_output=True, text=True)
        if build.returncode != 0:
            print(build.stderr)
            raise SystemExit("C++ test runner failed to compile")
        run = subprocess.run([exe], capture_output=True, text=True)
        print(run.stdout, end="")
        if run.returncode != 0:
            raise SystemExit("C++ tests failed")
        passed = run.stdout.count(": PASS")
        return passed


def run_fixed_tool_sanitizers():
    """True when the fixed tool runs clean under ASan and UBSan."""
    gpp = shutil.which("g++")
    if gpp is None:
        print("g++ not available: sanitizer run SKIPPED (named skip)")
        return False
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "tool_asan")
        build = subprocess.run(
            [gpp, "-std=c++17", "-fsanitize=address,undefined", "-g",
             os.path.join(CPP_DIR, "tool.cpp"), "-o", exe],
            capture_output=True, text=True)
        if build.returncode != 0:
            print(build.stderr)
            raise SystemExit("sanitizer build of tool.cpp failed")
        run = subprocess.run([exe], capture_output=True, text=True)
        clean = (run.returncode == 0 and "Sanitizer" not in run.stderr
                 and "runtime error" not in run.stderr)
        print(f"fixed tool under ASan+UBSan: "
              f"{'CLEAN' if clean else 'DIRTY'} (exit {run.returncode})")
        return clean


def asan_session_status(fixed_clean):
    """Session counts as recorded when the real buggy report is on
    file in DEBUG-SESSION.md and the fixed tool runs clean."""
    try:
        with open(DEBUG_SESSION, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return "not_recorded"
    has_summary = "SUMMARY: AddressSanitizer: heap-buffer-overflow" in text
    if has_summary and fixed_clean:
        return "recorded_and_fixed_clean"
    if has_summary:
        return "recorded_fixed_run_not_clean"
    return "not_recorded"


def main():
    data = analysis.load_results()
    checked = analysis.verify(data)
    print(analysis.format_table(data, checked))
    print(f"GPU rows verified: {len(checked)}")

    cpp_passed = run_cpp_tests()
    fixed_clean = run_fixed_tool_sanitizers()
    status = asan_session_status(fixed_clean)

    summary = {
        "cpp_tests_passed": cpp_passed,
        "asan_session_status": status,
        "gpu_rows_verified": len(checked),
        "gpu_device": data["device"],
        "gpu_note": ("PyTorch native ops only; custom CUDA kernels "
                     "in cuda/kernels.cu are written but not compiled "
                     "or run (no CUDA toolkit on the fleet GPU node)"),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(f"results_build.json written: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
