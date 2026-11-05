"""Unit tests for Level 3 Deep (P04 cuda-kernels-lab).

Covers: the Deep analysis recomputation math on synthetic rows,
the results_profiler.json schema, route labels and correctness
flags, recomputation of the stored profiler rows within 1
percent, the stage profile table, the torch.compile failure
being recorded (not hidden), the fused_softmax.cu source being
present with its guard and shared memory cache, and the C++
Deep test runner executing and passing when g++ exists.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEEP = os.path.join(ROOT, "level-3-deep")
sys.path.insert(0, DEEP)
import analysis_deep  # noqa: E402


def load_profiler():
    with open(os.path.join(DEEP, "results_profiler.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


class TestDeepRecomputeMath(unittest.TestCase):
    def test_matmul_gflops_synthetic(self):
        # 2 * 100^3 FLOPs in 2 ms = 1 GFLOP/s exactly.
        self.assertAlmostEqual(analysis_deep.matmul_gflops(100, 2.0),
                               1.0, places=9)

    def test_softmax_gb_per_s_synthetic(self):
        # 2 * 1000 * 500 * 4 bytes in 0.5 ms = 8 GB/s exactly.
        self.assertAlmostEqual(
            analysis_deep.softmax_gb_per_s(1000, 500, 0.5), 8.0, places=9)

    def test_verify_rejects_wrong_stored_value(self):
        data = {"benchmarks": [{
            "kernel": "matmul_torch", "route": "native op", "size": 100,
            "median_ms": 2.0, "gflops": 999.0, "correct": True}]}
        with self.assertRaises(AssertionError):
            analysis_deep.verify(data)

    def test_verify_rejects_custom_route_row(self):
        data = {"benchmarks": [{
            "kernel": "softmax_rows_torch", "route": "custom",
            "rows": 1000, "cols": 500, "median_ms": 0.5,
            "gb_per_s": 8.0, "correct": True}]}
        with self.assertRaises(AssertionError):
            analysis_deep.verify(data)

    def test_verify_rejects_unknown_route(self):
        data = {"benchmarks": [{
            "kernel": "matmul_torch", "route": "magic", "size": 100,
            "median_ms": 2.0, "gflops": 1.0, "correct": True}]}
        with self.assertRaises(AssertionError):
            analysis_deep.verify(data)


class TestProfilerFile(unittest.TestCase):
    def test_schema_and_device(self):
        data = load_profiler()
        self.assertEqual(data["device"],
                         "NVIDIA GeForce RTX 5050 Laptop GPU")
        self.assertEqual(data["compute_capability"], "12.0")
        self.assertIn("torch_version", data)
        self.assertEqual(len(data["benchmarks"]), 9)
        for row in data["benchmarks"]:
            self.assertIn(row["route"], analysis_deep.ALLOWED_ROUTES)
            self.assertNotEqual(row["route"], "custom")
            self.assertIn("median_ms", row)
            self.assertTrue(row["correct"] is True)

    def test_no_toolkit_recorded(self):
        data = load_profiler()
        self.assertIsNone(data["cuda_home"])
        self.assertFalse(data["nvcc_on_path"])

    def test_compile_attempt_recorded(self):
        data = load_profiler()
        comp = data["compile_softmax"]
        self.assertTrue(comp["attempted"])
        if comp["success"]:
            self.assertTrue(any(
                r["route"] == "compiled" for r in data["benchmarks"]))
        else:
            self.assertTrue(comp["error"])
            self.assertFalse(any(
                r["route"] == "compiled" for r in data["benchmarks"]))

    def test_stored_rows_recompute_within_one_percent(self):
        checked = analysis_deep.verify(load_profiler())
        self.assertEqual(len(checked), 9)
        for chk in checked:
            self.assertLessEqual(chk["rel_diff"], 0.01)

    def test_matmul_sizes_present(self):
        rows = [r for r in load_profiler()["benchmarks"]
                if r["kernel"] == "matmul_torch"]
        self.assertEqual([r["size"] for r in rows], [512, 1024, 2048])
        self.assertTrue(all(r["route"] == "native op" for r in rows))

    def test_softmax_routes_and_sizes(self):
        data = load_profiler()
        native = [r for r in data["benchmarks"]
                  if r["kernel"] == "softmax_rows_torch"]
        unfused = [r for r in data["benchmarks"]
                   if r["kernel"] == "softmax_rows_unfused_manual"]
        self.assertEqual(len(native), 3)
        self.assertEqual(len(unfused), 3)
        self.assertTrue(all(
            r["route"] == "unfused manual (native ops)" for r in unfused))
        # The fused native op is faster than the unfused sequence
        # at the headline size in the measured file.
        n0 = [r for r in native if r["rows"] == 4096][0]
        u0 = [r for r in unfused if r["rows"] == 4096][0]
        self.assertLess(n0["median_ms"], u0["median_ms"])
        self.assertGreater(n0["gb_per_s"], u0["gb_per_s"])

    def test_profiler_op_counts(self):
        data = load_profiler()
        cases = {(c["case"], c["route"]) for c in data["profiler"]}
        self.assertIn(("matmul_1024", "native op"), cases)
        self.assertIn(("softmax_4096x1024", "native op"), cases)
        self.assertIn(("softmax_4096x1024",
                       "unfused manual (native ops)"), cases)
        unfused = [c for c in data["profiler"]
                   if c["route"] == "unfused manual (native ops)"][0]
        op_names = {op["name"] for op in unfused["aten_ops"]}
        self.assertTrue({"aten::amax", "aten::exp", "aten::sum",
                         "aten::div"} <= op_names)
        native = [c for c in data["profiler"]
                  if c["case"] == "softmax_4096x1024"
                  and c["route"] == "native op"][0]
        self.assertTrue(any(op["name"] == "aten::_softmax"
                            for op in native["aten_ops"]))

    def test_stage_profile_present(self):
        data = load_profiler()
        stages = data["stage_profile_softmax_4096x1024"]
        names = [s["stage"] for s in stages]
        self.assertIn("native fused softmax (single op)", names)
        self.assertIn("unfused full sequence", names)
        for s in stages:
            self.assertGreater(s["median_ms"], 0)
        fused = [s for s in stages
                 if s["stage"] == "native fused softmax (single op)"][0]
        unfused = [s for s in stages
                   if s["stage"] == "unfused full sequence"][0]
        self.assertLess(fused["median_ms"], unfused["median_ms"])


class TestFusedKernelSource(unittest.TestCase):
    def test_fused_cu_source(self):
        path = os.path.join(DEEP, "cuda", "fused_softmax.cu")
        with open(path, encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("softmax_rows_fused_kernel", src)
        self.assertIn("__shared__", src)
        self.assertIn("vals", src)
        self.assertIn("NOT been compiled", src)
        self.assertIn("__syncthreads", src)


class TestDeepCppRunner(unittest.TestCase):
    def test_cpp_deep_tests_pass_when_gpp_available(self):
        gpp = shutil.which("g++")
        if gpp is None:
            self.skipTest("g++ not available")
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "tests_deep")
            build = subprocess.run(
                [gpp, "-std=c++17", "-O2", "-Wall", "-Wextra",
                 os.path.join(DEEP, "cpp", "tests_deep.cpp"),
                 "-o", exe, "-pthread"],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([exe], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertEqual(run.stdout.count(": PASS"), 15)
            self.assertIn("tests_deep: ALL PASS", run.stdout)

    def test_results_deep_summary_schema(self):
        path = os.path.join(DEEP, "results_deep.json")
        self.assertTrue(os.path.exists(path),
                        "run_deep.py must run before the full suite "
                        "check; results_deep.json is a Deep output")
        with open(path, encoding="utf-8") as fh:
            summary = json.load(fh)
        self.assertEqual(summary["cpp_deep_tests_passed"], 15)
        self.assertEqual(summary["cpp_build_tests_passed"], 16)
        self.assertEqual(summary["asan_ubsan_deep_tool"], "clean")
        self.assertEqual(summary["asan_ubsan_build_tool"], "clean")
        self.assertEqual(summary["gpu_benchmark_rows_verified"], 9)
        self.assertFalse(summary["custom_kernel_compiled"])
        self.assertEqual(summary["atomics"]["relaxed_total"], 1000000)
        self.assertEqual(summary["atomics"]["seqcst_total"], 1000000)
        self.assertEqual(summary["handoff"]["ordered_successes"], 200)
        self.assertEqual(summary["move_bench"]["copy_count"], 50)
        self.assertEqual(summary["move_bench"]["move_count"], 50)


if __name__ == "__main__":
    unittest.main()
