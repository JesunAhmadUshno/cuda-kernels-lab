// tests.cpp - self-contained assertion test runner for tool.cpp.
//
// Framework choice: no external framework. Catch2 and GoogleTest
// were not fetched, and this repo carries no vendored dependency;
// pulling a framework in for these assertions would add a network
// fetch or a vendored copy with no benefit at this scale. This
// runner provides the contract CI needs: one PASS or FAIL line per
// test, a total line, and a non-zero exit code on any failure.
//
// Build: g++ -std=c++17 -O2 tests.cpp -o tests
// It includes tool.cpp behind the TOOL_NO_MAIN guard so the tests
// exercise exactly the functions the tool ships.

#define TOOL_NO_MAIN
#include "tool.cpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

static int g_passed = 0;
static int g_failed = 0;

static void Report(const std::string& name, bool ok) {
  std::cout << name << ": " << (ok ? "PASS" : "FAIL") << "\n";
  if (ok) ++g_passed; else ++g_failed;
}

static bool NearVec(const std::vector<double>& got,
                    const std::vector<double>& want, double tol = 1e-9) {
  if (got.size() != want.size()) return false;
  for (std::size_t i = 0; i < got.size(); ++i) {
    if (std::fabs(got[i] - want[i]) > tol) return false;
  }
  return true;
}

static std::vector<double> SquareSeq(int n, double start) {
  std::vector<double> m(static_cast<std::size_t>(n) * n);
  for (int i = 0; i < n * n; ++i) m[static_cast<std::size_t>(i)] = start + i;
  return m;
}

int main() {
  // 1. Vector add, hand-computed values.
  {
    Buffer<double> a(3), b(3);
    a[0] = 1; a[1] = 2; a[2] = 3;
    b[0] = 10; b[1] = 20; b[2] = 30;
    long long adds = 0;
    auto got = VectorAdd<double>(a, b, adds);
    Report("vector_add_hand_values",
           NearVec(got, {11, 22, 33}) && adds == 3);
  }
  // 2. Vector add with negative and zero values.
  {
    Buffer<double> a(4), b(4);
    a[0] = -5; a[1] = 0; a[2] = 7.5; a[3] = -2.5;
    b[0] = 5; b[1] = -3; b[2] = -7.5; b[3] = 2.5;
    long long adds = 0;
    auto got = VectorAdd<double>(a, b, adds);
    Report("vector_add_signed_and_zero",
           NearVec(got, {0, -3, 0, 0}) && adds == 4);
  }
  // 3. Matmul naive 2x2, hand-computed.
  {
    long long loads = 0;
    auto got = MatmulNaive<double>({1, 2, 3, 4}, {5, 6, 7, 8}, 2, loads);
    Report("matmul_naive_2x2_hand",
           NearVec(got, {19, 22, 43, 50}) && loads == 16);
  }
  // 4. Matmul tiled 2x2, hand-computed, matches naive.
  {
    long long tl = 0, nl = 0;
    auto got = MatmulTiled<double>({1, 2, 3, 4}, {5, 6, 7, 8}, 2, 2, tl);
    auto ref = MatmulNaive<double>({1, 2, 3, 4}, {5, 6, 7, 8}, 2, nl);
    Report("matmul_tiled_2x2_hand", NearVec(got, {19, 22, 43, 50}) &&
                                    NearVec(got, ref));
  }
  // 5. Matmul tiled 3x3 at tile 2, hand-computed (Learn values).
  {
    std::vector<double> a = {2, 0, 1, 1, 3, 2, 0, 1, 1};
    std::vector<double> b = {1, 2, 0, 0, 1, 3, 2, 1, 1};
    long long tl = 0;
    auto got = MatmulTiled<double>(a, b, 3, 2, tl);
    Report("matmul_tiled_3x3_hand",
           NearVec(got, {4, 5, 1, 5, 7, 11, 2, 2, 4}));
  }
  // 6. Tile load count, 4x4 tile 2: blocks 2x2, k tiles 2, so
  //    4 * 2 * 2 * (2*2) = 64, against 128 naive element loads.
  {
    auto m = SquareSeq(4, 0.0);
    long long tl = 0, nl = 0;
    MatmulTiled<double>(m, m, 4, 2, tl);
    MatmulNaive<double>(m, m, 4, nl);
    Report("tile_load_count_4x4_tile2", tl == 64 && nl == 128);
  }
  // 7. Tile load count, 8x8 tile 2: blocks 4x4, k tiles 4, so
  //    16 * 4 * 2 * (2*2) = 512, against 1024 naive loads.
  {
    auto m = SquareSeq(8, 0.0);
    long long tl = 0, nl = 0;
    MatmulTiled<double>(m, m, 8, 2, tl);
    MatmulNaive<double>(m, m, 8, nl);
    Report("tile_load_count_8x8_tile2", tl == 512 && nl == 1024);
  }
  // 8. Tile load count, 16x16 tile 4: blocks 4x4, k tiles 4, so
  //    16 * 4 * 2 * (4*4) = 2048, against 8192 naive loads.
  {
    auto m = SquareSeq(16, 0.0);
    long long tl = 0, nl = 0;
    MatmulTiled<double>(m, m, 16, 4, tl);
    MatmulNaive<double>(m, m, 16, nl);
    Report("tile_load_count_16x16_tile4", tl == 2048 && nl == 8192);
  }
  // 9. Edge size: 3x3 at tile 2 is not a multiple of the tile.
  //    Tiled must equal naive exactly (guarded zero padding).
  {
    std::vector<double> a = {2, 0, 1, 1, 3, 2, 0, 1, 1};
    std::vector<double> b = {1, 2, 0, 0, 1, 3, 2, 1, 1};
    long long tl = 0, nl = 0;
    auto got = MatmulTiled<double>(a, b, 3, 2, tl);
    auto ref = MatmulNaive<double>(a, b, 3, nl);
    // Edge tile loads: blocks 2x2, k tiles 2, 4 * 2 * 2 * 4 = 64.
    Report("edge_3x3_tile2_matches_naive", NearVec(got, ref) && tl == 64);
  }
  // 10. Edge size: 5x5 at tile 4 (5 is not a multiple of 4).
  {
    auto a = SquareSeq(5, 1.0);
    auto b = SquareSeq(5, -2.0);
    long long tl = 0, nl = 0;
    auto got = MatmulTiled<double>(a, b, 5, 4, tl);
    auto ref = MatmulNaive<double>(a, b, 5, nl);
    // blocks 2x2, k tiles 2, 4 * 2 * 2 * 16 = 256.
    Report("edge_5x5_tile4_matches_naive", NearVec(got, ref) && tl == 256);
  }
  // 11. Edge size: 5x5 at tile 2 against naive.
  {
    auto a = SquareSeq(5, 0.5);
    auto b = SquareSeq(5, 3.0);
    long long tl = 0, nl = 0;
    auto got = MatmulTiled<double>(a, b, 5, 2, tl);
    auto ref = MatmulNaive<double>(a, b, 5, nl);
    Report("edge_5x5_tile2_matches_naive", NearVec(got, ref));
  }
  // 12. Softmax row sums to 1 and ordering is preserved.
  {
    auto got = SoftmaxStable<double>({1.0, 2.0, 3.0});
    double s = got[0] + got[1] + got[2];
    Report("softmax_row_sum_to_one",
           std::fabs(s - 1.0) < 1e-12 && got[2] > got[1] && got[1] > got[0]);
  }
  // 13. Stable softmax on large inputs: finite, sums to 1. The
  //     naive form overflows on these values (Learn exhibit).
  {
    auto got = SoftmaxStable<double>({1000.0, 1001.0, 999.0});
    double s = got[0] + got[1] + got[2];
    bool finite = std::isfinite(got[0]) && std::isfinite(got[1]) &&
                  std::isfinite(got[2]);
    Report("softmax_stable_large_inputs",
           finite && std::fabs(s - 1.0) < 1e-12 && got[1] > got[0]);
  }
  // 14. Softmax of a uniform row is uniform.
  {
    auto got = SoftmaxStable<double>({7.0, 7.0, 7.0, 7.0});
    Report("softmax_uniform_row",
           NearVec(got, {0.25, 0.25, 0.25, 0.25}, 1e-12));
  }
  // 15. SoftmaxRows: every row sums to 1.
  {
    auto got = SoftmaxRows<double>({1.0, 2.0, 3.0, -1.0, 0.0, 1.0}, 2, 3);
    double s0 = got[0] + got[1] + got[2];
    double s1 = got[3] + got[4] + got[5];
    Report("softmax_rows_each_row_sums_to_one",
           std::fabs(s0 - 1.0) < 1e-12 && std::fabs(s1 - 1.0) < 1e-12);
  }
  // 16. Softmax shift invariance: adding a constant to a row does
  //     not change the result (the reason max subtraction is safe).
  {
    auto a = SoftmaxStable<double>({0.5, -1.0, 2.0});
    auto b = SoftmaxStable<double>({1000.5, 999.0, 1002.0});
    Report("softmax_shift_invariant", NearVec(a, b, 1e-12));
  }

  std::cout << "total: " << g_passed << " passed, " << g_failed
            << " failed\n";
  std::cout << "tests: " << (g_failed == 0 ? "ALL PASS" : "FAILURES")
            << "\n";
  return g_failed == 0 ? 0 : 1;
}
