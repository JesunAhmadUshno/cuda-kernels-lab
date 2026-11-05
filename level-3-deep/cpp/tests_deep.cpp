// tests_deep.cpp - self-contained assertion test runner for
// the Level 3 Deep C++ tool.
//
// Framework choice matches Build: no external framework, no
// vendored dependency. The runner prints one PASS or FAIL line
// per test, a total line, and exits non-zero on any failure.
//
// Build: g++ -std=c++17 -O2 tests_deep.cpp -o tests_deep
// It includes deep_tool.cpp behind the DEEP_NO_MAIN guard (which
// in turn includes the Build tool behind TOOL_NO_MAIN), so the
// tests exercise exactly the functions the Deep tool ships.

#define DEEP_NO_MAIN
#include "deep_tool.cpp"

#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

static int g_passed = 0;
static int g_failed = 0;

static void Report(const std::string& name, bool ok) {
  std::cout << name << ": " << (ok ? "PASS" : "FAIL") << "\n";
  if (ok) ++g_passed; else ++g_failed;
}

int main() {
  // 1. Build vector add still correct inside the Deep tool.
  {
    Buffer<double> a(3), b(3);
    a[0] = 1; a[1] = 2; a[2] = 3;
    b[0] = 10; b[1] = 20; b[2] = 30;
    long long adds = 0;
    auto got = VectorAdd<double>(a, b, adds);
    Report("deep_includes_build_vector_add",
           got.size() == 3 && got[0] == 11 && got[1] == 22 &&
               got[2] == 33 && adds == 3);
  }
  // 2. Build tiled matmul edge case still correct in Deep.
  {
    std::vector<double> a = {2, 0, 1, 1, 3, 2, 0, 1, 1};
    std::vector<double> b = {1, 2, 0, 0, 1, 3, 2, 1, 1};
    long long tl = 0;
    auto got = MatmulTiled<double>(a, b, 3, 2, tl);
    std::vector<double> want = {4, 5, 1, 5, 7, 11, 2, 2, 4};
    bool ok = got.size() == want.size();
    for (std::size_t i = 0; ok && i < got.size(); ++i) {
      ok = std::fabs(got[i] - want[i]) < 1e-9;
    }
    Report("deep_includes_build_tiled_matmul_edge", ok);
  }
  // 3. Build stable softmax still correct in Deep.
  {
    auto got = SoftmaxStable<double>({1000.0, 1001.0, 999.0});
    double s = got[0] + got[1] + got[2];
    Report("deep_includes_build_softmax_stable",
           std::fabs(s - 1.0) < 1e-12 && got[1] > got[0] && got[0] > got[2]);
  }
  // 4. Atomic counter, relaxed ordering, exact total.
  {
    AtomicRun r = AtomicCounterRun(4, 50000, false);
    Report("atomic_counter_relaxed_exact_total", r.total == 200000 &&
                                                  r.elapsed_ms >= 0.0);
  }
  // 5. Atomic counter, seq_cst ordering, exact total.
  {
    AtomicRun r = AtomicCounterRun(4, 50000, true);
    Report("atomic_counter_seqcst_exact_total", r.total == 200000 &&
                                                r.elapsed_ms >= 0.0);
  }
  // 6. Atomic counter, single thread relaxed, exact total.
  {
    AtomicRun r = AtomicCounterRun(1, 1000, false);
    Report("atomic_counter_single_thread", r.total == 1000);
  }
  // 7. Atomic counter with an uneven split still sums exactly:
  //    3 threads by 33333 = 99999.
  {
    AtomicRun r = AtomicCounterRun(3, 33333, true);
    Report("atomic_counter_uneven_split_exact", r.total == 99999);
  }
  // 8. Producer-consumer handoff, acquire-release, all intact.
  {
    HandoffRun r = ProducerConsumerRun(100, true);
    Report("handoff_acquire_release_all_intact",
           r.successes == r.trials && r.trials == 100);
  }
  // 9. Producer-consumer handoff, relaxed control, runs and
  //    reports its count. On this platform the count is also
  //    expected to be full (x86 has a strong memory model), but
  //    the test asserts only that the run completed and the
  //    counting is consistent, because relaxed ordering gives
  //    no cross-variable guarantee in the C++ model.
  {
    HandoffRun r = ProducerConsumerRun(100, false);
    Report("handoff_relaxed_control_completes",
           r.trials == 100 && r.successes >= 0 && r.successes <= 100);
  }
  // 10. LargeBuffer copy constructor copies and counts one copy.
  {
    LargeBuffer::ResetCounts();
    LargeBuffer src(1000, 2.0);
    LargeBuffer copy = src;
    Report("large_buffer_copy_counts_and_copies",
           LargeBuffer::copy_count == 1 && LargeBuffer::move_count == 0 &&
               copy.size() == 1000 && src.size() == 1000 &&
               std::fabs(copy.checksum() - 2000.0) < 1e-9);
  }
  // 11. LargeBuffer move constructor moves, counts one move, and
  //     leaves the source empty.
  {
    LargeBuffer::ResetCounts();
    LargeBuffer src(1000, 2.0);
    LargeBuffer moved = std::move(src);
    Report("large_buffer_move_counts_and_empties_source",
           LargeBuffer::move_count == 1 && LargeBuffer::copy_count == 0 &&
               moved.size() == 1000 && src.size() == 0 &&
               std::fabs(moved.checksum() - 2000.0) < 1e-9);
  }
  // 12. LargeBuffer copy assignment counts and copies.
  {
    LargeBuffer::ResetCounts();
    LargeBuffer src(500, 3.0);
    LargeBuffer dst(10, 0.0);
    dst = src;
    Report("large_buffer_copy_assignment",
           LargeBuffer::copy_count == 1 && dst.size() == 500 &&
               std::fabs(dst.checksum() - 1500.0) < 1e-9);
  }
  // 13. LargeBuffer move assignment counts and empties source.
  {
    LargeBuffer::ResetCounts();
    LargeBuffer src(500, 3.0);
    LargeBuffer dst(10, 0.0);
    dst = std::move(src);
    Report("large_buffer_move_assignment",
           LargeBuffer::move_count == 1 && dst.size() == 500 &&
               src.size() == 0 &&
               std::fabs(dst.checksum() - 1500.0) < 1e-9);
  }
  // 14. Move bench: exact counts on both paths, small size.
  {
    MoveBench r = MoveBenchRun(10000, 10);
    Report("move_bench_exact_counts_small",
           r.copies == 10 && r.moves == 10 && r.moves_in_copy_path == 0 &&
               r.copies_in_move_path == 0 &&
               std::fabs(r.checksum - 30.0) < 1e-9);
  }
  // 15. Move bench: timings are measured (non-negative) and the
  //     element and rep fields echo the workload.
  {
    MoveBench r = MoveBenchRun(50000, 5);
    Report("move_bench_timing_fields",
           r.elements == 50000 && r.reps == 5 && r.copy_ms >= 0.0 &&
               r.move_ms >= 0.0);
  }

  std::cout << "total: " << g_passed << " passed, " << g_failed
            << " failed\n";
  std::cout << "tests_deep: " << (g_failed == 0 ? "ALL PASS" : "FAILURES")
            << "\n";
  return g_failed == 0 ? 0 : 1;
}
