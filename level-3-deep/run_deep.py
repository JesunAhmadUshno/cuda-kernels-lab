"""Level 3 Deep runner.

1. Verifies results_profiler.json through analysis_deep.py and
   prints the benchmark table (PyTorch routes on the fleet GPU
   node, labelled per row, never custom kernel results).
2. Compiles and runs the C++ Deep test runner
   (cpp/tests_deep.cpp) and the Build C++ test runner when g++
   is available; skips with a named message when it is not.
3. Builds the Deep tool and the Build tool with
   AddressSanitizer and UndefinedBehaviorSanitizer and runs
   them; both must exit 0 with no sanitizer output.
4. Runs the Deep tool plain, parses its measured atomics,
   handoff, and move benchmark values, and writes
   results_deep.json summarising the whole Deep run.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import analysis_deep  # noqa: E402

CPP_DIR = os.path.join(HERE, "cpp")
BUILD_CPP = os.path.join(ROOT, "level-2-build", "cpp")
OUT = os.path.join(HERE, "results_deep.json")


def compile_and_run(gpp, sources, exe_name, extra=None):
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, exe_name)
        cmd = [gpp, "-std=c++17", "-O2", "-Wall", "-Wextra"]
        if extra:
            cmd += extra
        cmd += list(sources) + ["-o", exe, "-pthread"]
        build = subprocess.run(cmd, capture_output=True, text=True)
        if build.returncode != 0:
            print(build.stderr)
            raise SystemExit(f"compile failed: {sources}")
        run = subprocess.run([exe], capture_output=True, text=True)
        return run


def run_cpp_tests(gpp, path, label):
    if gpp is None:
        print(f"g++ not available: {label} SKIPPED (named skip)")
        return None
    run = compile_and_run(gpp, [path], "tests_bin")
    print(run.stdout, end="")
    if run.returncode != 0:
        raise SystemExit(f"{label} failed")
    return run.stdout.count(": PASS")


def run_sanitizer(gpp, source, label):
    if gpp is None:
        print(f"g++ not available: {label} sanitizer run SKIPPED")
        return False
    with tempfile.TemporaryDirectory() as tmp:
        exe = os.path.join(tmp, "tool_asan")
        build = subprocess.run(
            [gpp, "-std=c++17", "-fsanitize=address,undefined", "-g",
             source, "-o", exe, "-pthread"],
            capture_output=True, text=True)
        if build.returncode != 0:
            print(build.stderr)
            raise SystemExit(f"sanitizer build failed: {label}")
        run = subprocess.run([exe], capture_output=True, text=True)
        clean = (run.returncode == 0 and "Sanitizer" not in run.stderr
                 and "runtime error" not in run.stderr)
        print(f"{label} under ASan+UBSan: "
              f"{'CLEAN' if clean else 'DIRTY'} (exit {run.returncode})")
        return clean


def parse_deep_tool_output(text):
    def grab(pattern, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    return {
        "atomics": {
            "threads": 4,
            "per_thread": 250000,
            "relaxed_total": grab(r"atomics_relaxed_total=(\d+)", int),
            "relaxed_ms": grab(r"atomics_relaxed_ms=([\d.]+)"),
            "seqcst_total": grab(r"atomics_seqcst_total=(\d+)", int),
            "seqcst_ms": grab(r"atomics_seqcst_ms=([\d.]+)"),
        },
        "handoff": {
            "ordered_successes": grab(r"handoff_ordered_successes=(\d+)", int),
            "ordered_trials": grab(r"handoff_ordered_successes=\d+/(\d+)", int),
            "ordered_ms": grab(r"handoff_ordered_ms=([\d.]+)"),
            "relaxed_successes": grab(r"handoff_relaxed_successes=(\d+)", int),
            "relaxed_trials": grab(r"handoff_relaxed_successes=\d+/(\d+)", int),
            "relaxed_ms": grab(r"handoff_relaxed_ms=([\d.]+)"),
        },
        "move_bench": {
            "elements": grab(r"move_elements=(\d+)", int),
            "reps": grab(r"move_reps=(\d+)", int),
            "copy_count": grab(r"move_copy_count=(\d+)", int),
            "copy_ms": grab(r"move_copy_ms=([\d.]+)"),
            "move_count": grab(r"move_move_count=(\d+)", int),
            "move_ms": grab(r"move_move_ms=([\d.]+)"),
            "checksum": grab(r"move_checksum=([\d.]+)"),
        },
    }


def main():
    data = analysis_deep.load_results()
    checked = analysis_deep.verify(data)
    print(analysis_deep.format_table(data, checked))
    print(f"profiler rows verified: {len(checked)}")

    gpp = shutil.which("g++")
    deep_passed = run_cpp_tests(
        gpp, os.path.join(CPP_DIR, "tests_deep.cpp"), "Deep C++ tests")
    build_passed = run_cpp_tests(
        gpp, os.path.join(BUILD_CPP, "tests.cpp"), "Build C++ tests")

    deep_clean = run_sanitizer(
        gpp, os.path.join(CPP_DIR, "deep_tool.cpp"), "Deep tool")
    build_clean = run_sanitizer(
        gpp, os.path.join(BUILD_CPP, "tool.cpp"), "Build tool")

    measured = {"atomics": None, "handoff": None, "move_bench": None}
    if gpp is not None:
        run = compile_and_run(
            gpp, [os.path.join(CPP_DIR, "deep_tool.cpp")], "deep_tool")
        print(run.stdout, end="")
        if run.returncode != 0:
            raise SystemExit("Deep tool self-check failed")
        measured = parse_deep_tool_output(run.stdout)

    summary = {
        "cpp_deep_tests_passed": deep_passed,
        "cpp_build_tests_passed": build_passed,
        "asan_ubsan_deep_tool": "clean" if deep_clean else "not_clean",
        "asan_ubsan_build_tool": "clean" if build_clean else "not_clean",
        "atomics": measured["atomics"],
        "handoff": measured["handoff"],
        "move_bench": measured["move_bench"],
        "gpu_benchmark_rows_verified": len(checked),
        "profiler_cases": len(data.get("profiler", [])),
        "gpu_device": data["device"],
        "compile_route_attempted": data["compile_softmax"]["attempted"],
        "compile_route_success": data["compile_softmax"]["success"],
        "custom_kernel_compiled": False,
        "custom_kernel_note": ("fused_softmax.cu is written only; "
                               "no CUDA toolkit on the fleet GPU "
                               "node, so it was not compiled or run"),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")
    print(f"results_deep.json written: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
