"""Unit tests for Level 1 Learn kernels (P04 cuda-kernels-lab).

Hand-computed values throughout. At least 10 tests: vector add,
2x2 and 3x3 matmul naive and tiled with identical results, tile
load counts, stable softmax, and the C++ tool compile and self check.
"""
import math
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "level-1-learn"))
from kernels import (matmul_naive, matmul_tiled, softmax_naive,
                     softmax_stable, vector_add)


class TestVectorAdd(unittest.TestCase):
    def test_hand_values(self):
        got, counts = vector_add([1, 2, 3], [10, 20, 30])
        self.assertEqual(got, [11, 22, 33])
        self.assertEqual(counts["adds"], 3)
        self.assertEqual(counts["element_loads"], 6)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            vector_add([1], [1, 2])


class TestMatmul(unittest.TestCase):
    A2 = [[1, 2], [3, 4]]
    B2 = [[5, 6], [7, 8]]
    W2 = [[19, 22], [43, 50]]
    A3 = [[2, 0, 1], [1, 3, 2], [0, 1, 1]]
    B3 = [[1, 2, 0], [0, 1, 3], [2, 1, 1]]
    # Hand computed: row0: 2*1+0+1*2=4, 2*2+0+1*1=5, 0+0+1*1=1
    # row1: 1+0+4=5, 2+3+2=7, 0+9+2=11 ; row2: 0+0+2=2, 0+1+1=2, 0+3+1=4
    W3 = [[4, 5, 1], [5, 7, 11], [2, 2, 4]]

    def test_naive_2x2_hand(self):
        got, counts = matmul_naive(self.A2, self.B2)
        self.assertEqual(got, self.W2)
        self.assertEqual(counts["multiplies"], 8)
        self.assertEqual(counts["adds"], 4)
        self.assertEqual(counts["element_loads"], 16)

    def test_tiled_2x2_hand(self):
        got, _ = matmul_tiled(self.A2, self.B2, 2)
        self.assertEqual(got, self.W2)

    def test_naive_3x3_hand(self):
        got, _ = matmul_naive(self.A3, self.B3)
        self.assertEqual(got, self.W3)

    def test_tiled_3x3_tile2_matches_hand_and_naive(self):
        got, _ = matmul_tiled(self.A3, self.B3, 2)
        naive, _ = matmul_naive(self.A3, self.B3)
        self.assertEqual(got, self.W3)
        self.assertEqual(got, naive)

    def test_tile_load_counts_4x4_tile2(self):
        a = [[float(i * 4 + j) for j in range(4)] for i in range(4)]
        _, cn = matmul_naive(a, a)
        _, ct = matmul_tiled(a, a, 2)
        # naive: 2 * 4 * 4 * 4 = 128. tiled: 4 blocks * 2 k tiles *
        # 2 tiles * (2*2) slots = 64.
        self.assertEqual(cn["element_loads"], 128)
        self.assertEqual(ct["tile_loads"], 64)
        self.assertEqual(ct["tiles_loaded"], 16)

    def test_tile_load_counts_8x8_tile2(self):
        a = [[float(i * 8 + j) for j in range(8)] for i in range(8)]
        _, cn = matmul_naive(a, a)
        _, ct = matmul_tiled(a, a, 2)
        # naive: 2 * 8^3 = 1024. tiled: 16 blocks * 4 k tiles * 8 = 512.
        self.assertEqual(cn["element_loads"], 1024)
        self.assertEqual(ct["tile_loads"], 512)


class TestSoftmax(unittest.TestCase):
    def test_stable_sums_to_one(self):
        got, _ = softmax_stable([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(got), 1.0, places=12)
        self.assertTrue(got[2] > got[1] > got[0])

    def test_stable_matches_naive_on_small(self):
        small = [0.5, -1.0, 2.0]
        a, _ = softmax_naive(small)
        b, _ = softmax_stable(small)
        for x, y in zip(a, b):
            self.assertAlmostEqual(x, y, places=12)

    def test_stable_large_naive_overflows(self):
        got, _ = softmax_stable([1000.0, 1001.0, 999.0])
        self.assertAlmostEqual(sum(got), 1.0, places=12)
        self.assertTrue(all(math.isfinite(v) for v in got))
        with self.assertRaises(OverflowError):
            softmax_naive([1000.0, 1001.0])


class TestCppTool(unittest.TestCase):
    def test_cpp_tool_compiles_and_self_checks(self):
        src = os.path.join(ROOT, "level-1-learn", "cpp", "bench_tool.cpp")
        with tempfile.TemporaryDirectory() as tmp:
            exe = os.path.join(tmp, "bench_tool")
            build = subprocess.run(
                ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra",
                 src, "-o", exe],
                capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stderr)
            run = subprocess.run([exe], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("vector_add: PASS", run.stdout)
            self.assertIn("matmul_tiled_2x2: PASS", run.stdout)
            self.assertIn("ALL PASS", run.stdout)


if __name__ == "__main__":
    unittest.main()
