"""Unit tests for Level 2 Build (P04 cuda-kernels-lab).

Covers: the analysis recomputation math on synthetic rows, the
results_torch_bench.json schema and correctness flags, recomputation
of the stored GPU rows within 1 percent, the kernels.cu source
(four named kernels plus the bounds guard), the C++ test runner
executing and passing when g++ exists, and the DEBUG-SESSION.md
sanitizer SUMMARY line.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD = os.path.join(ROOT, "level-2-build")
sys.path.insert(0, BUILD)
import analysis  # noqa: E402


def load_bench():
    with open(os.path.join(BUILD, "results_torch_bench.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


class TestRecomputeMath(unittest.TestCase):
    def test_vector_gb_per_s_synthetic(self):
        # 3 * 1,000,000 * 4 bytes in 1 ms = 12 GB/s exactly.
        self.assertAlmostEqual(analysis.vector_gb_per_s(1_000_000, 1.0),
                               12.0, places=9)

    def test_matmul_gflops_synthetic(self):
        # 2 * 100^3 FLOPs in 2 ms = 1 GFLOP/s exactly.
        self.assertAlmostEqual(analysis.matmul_gflops(100, 2.0),
                               1.0, places=9)

    def test_softmax_gb_per_s_synthetic(self):
        # 2 * 1000 * 500 * 4 bytes in 0.5 ms = 8 GB/s exactly.
        self.assertAlmostEqual(analysis.softmax_gb_per_s(1000, 500, 0.5),
                               8.0, places=9)

    def test_within_tolerance_bounds(self):
        self.assertTrue(analysis.within_tolerance(100.5, 100.0))
        self.assertTrue(analysis.within_tolerance(99.5, 100.0))
        self.assertFalse(analysis.within_tolerance(102.0, 100.0))
        self.assertFalse(analysis.within_tolerance(98.0, 100.0))

    def test_verify_rejects_wrong_stored_value(self):
        data = {"benchmarks": [{
            "kernel": "vector_add_torch", "n": 1_000_000,
            "median_ms": 1.0, "gb_per_s": 999.0, "correct": True}]}
        with self.assertRaises(AssertionError):
            analysis.verify(data)

    def test_verify_rejects_incorrect_flag(self):
        data = {"benchmarks": [{
            "kernel": "vector_add_torch", "n": 1_000_000,
            "median_ms": 1.0, "gb_per_s": 12.0, "correct": False}]}
        with self.assertRaises(AssertionError):
            analysis.verify(data)


class TestTorchBenchFile(unittest.TestCase):
    def test_schema_and_device(self):
        data = load_bench()
        self.assertEqual(data["device"],
                         "NVIDIA GeForce RTX 5050 Laptop GPU")
        self.assertEqual(data["compute_capability"], "12.0")
        self.assertIn("torch_version", data)
        self.assertEqual(len(data["benchmarks"]), 4)
        for row in data["benchmarks"]:
            self.assertIn("kernel", row)
            self.assertIn("median_ms", row)
            self.assertIn("correct", row)
            self.assertTrue("gb_per_s" in row or "gflops" in row)

    def test_all_correct_flags_true(self):
        data = load_bench()
        self.assertTrue(all(r["correct"] is True
                            for r in data["benchmarks"]))

    def test_stored_rows_recompute_within_one_percent(self):
        checked = analysis.verify(load_bench())
        self.assertEqual(len(checked), 4)
        for chk in checked:
            self.assertLessEqual(chk["rel_diff"], 0.01)

    def test_vector_add_row_recompute(self):
        row = load_bench()["benchmarks"][0]
        self.assertEqual(row["kernel"], "vector_add_torch")
        self.assertEqual(row["n"], 4194304)
        recomputed = analysis.vector_gb_per_s(row["n"], row["median_ms"])
        self.assertAlmostEqual(recomputed, 381.3, delta=381.3 * 0.01)

    def test_matmul_rows_recompute(self):
        rows = [r for r in load_bench()["benchmarks"]
                if r["kernel"] == "matmul_torch"]
        self.assertEqual([r["size"] for r in rows], [512, 1024])
        for row in rows:
            recomputed = analysis.matmul_gflops(row["size"],
                                                row["median_ms"])
            self.assertAlmostEqual(recomputed, row["gflops"],
                                   delta=row["gflops"] * 0.01)

    def test_softmax_row_recompute(self):
        row = load_bench()["benchmarks"][3]
        self.assertEqual(row["kernel"], "softmax_rows_torch")
        self.assertEqual((row["rows"], row["cols"]), (4096, 1024))
        recomputed = analysis.softmax_gb_per_s(row["rows"], row["cols"],
                                               row["median_ms"])
        self.assertAlmostEqual(recomputed, row["gb_per_s"],
                               delta=row["gb_per_s"] * 0.01)


class TestCudaSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(os.path.join(BUILD, "cuda", "kernels.cu"),
                  encoding="utf-8") as fh:
            cls.src = fh.read()

    def test_four_named_kernels_present(self):
        for name in ("vector_add_kernel", "matmul_naive_kernel",
                     "matmul_tiled_kernel", "softmax_rows_kernel"):
            self.assertIn(name, self.src)

    def test_bounds_guard_present(self):
        # The vector add guard and the tiled load and store guards
        # whose absence causes the DEBUG-SESSION crash.
        self.assertIn("if (i < n) c[i] = a[i] + b[i];", self.src)
        self.assertIn("(row < n && acol < n)", self.src)
        self.assertIn("(brow < n && col < n)", self.src)
        self.assertIn("if (row < n && col < n) c[row * n + col] = sum;",
                      self.src)


class TestCppRunner(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "g++ not available")
    def test_cpp_tests_compile_and_pass(self):
        src = os.path.join(BUILD, "cpp", "tests.cpp")
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "cpp_tests")
            build = subprocess.run(
                ["g++", "-std=c++17", "-O2", src, "-o", exe],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([exe], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("tests: ALL PASS", run.stdout)
            self.assertGreaterEqual(run.stdout.count(": PASS"), 12)

    @unittest.skipUnless(shutil.which("g++"), "g++ not available")
    def test_tool_compiles_and_self_checks(self):
        src = os.path.join(BUILD, "cpp", "tool.cpp")
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "tool")
            build = subprocess.run(
                ["g++", "-std=c++17", "-O2", src, "-o", exe],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([exe], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("tool: ALL PASS", run.stdout)


class TestDebugSession(unittest.TestCase):
    def test_debug_session_contains_summary_line(self):
        with open(os.path.join(BUILD, "DEBUG-SESSION.md"),
                  encoding="utf-8") as fh:
            text = fh.read()
        self.assertIn(
            "SUMMARY: AddressSanitizer: heap-buffer-overflow", text)
        self.assertIn("WRITE of size 8", text)
        self.assertIn("MatmulTiledBuggy", text)

    def test_results_build_summary_schema(self):
        path = os.path.join(BUILD, "results_build.json")
        self.assertTrue(os.path.exists(path),
                        "run_build.py must run before the full suite "
                        "check; results_build.json is a Build output")
        with open(path, encoding="utf-8") as fh:
            summary = json.load(fh)
        self.assertEqual(summary["gpu_rows_verified"], 4)
        self.assertEqual(summary["cpp_tests_passed"], 16)
        self.assertEqual(summary["asan_session_status"],
                         "recorded_and_fixed_clean")


if __name__ == "__main__":
    unittest.main()
