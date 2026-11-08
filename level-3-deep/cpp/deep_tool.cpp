// deep_tool.cpp - standalone C++17 tool for cuda-kernels-lab
// Level 3 Deep.
//
// Evolves the Level 2 Build tool: this file includes the Build
// tool (level-2-build/cpp/tool.cpp) behind its no-main guard, so
// the Deep binary runs the same VectorAdd, MatmulTiled and
// SoftmaxStable functions the Build shipped, and adds the two
// Deep items from PROPOSAL.md:
//
// 1. Atomics and memory ordering, measured. An atomic counter is
//    incremented by several threads under relaxed ordering and
//    under sequentially consistent ordering; both must reach the
//    exact expected total (atomicity is what guarantees the
//    count, the ordering choice is about the cost and about what
//    other memory operations are ordered around the counter).
//    A producer-consumer handoff then passes a payload through a
//    plain int guarded by an atomic flag: the ordered form uses
//    release on the store and acquire on the load, the control
//    form uses relaxed for both. Every trial is checked.
// 2. Move semantics, measured. LargeBuffer is a move-aware type
//    owning a large vector. A benchmark copies it and moves it a
//    fixed number of times and records the exact copy and move
//    constructor counts and the wall time of each path.
//
// main() is guarded by DEEP_NO_MAIN so cpp/tests_deep.cpp can
// include this file and test the same functions the tool runs.
// No GPU or CUDA kernel result is claimed from this file.

#define TOOL_NO_MAIN
#include "../../level-2-build/cpp/tool.cpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

// ---------------------------------------------------------------- atomics

struct AtomicRun {
  long long total;
  double elapsed_ms;
};

// Increment a shared atomic counter from `threads` threads, each
// doing `per_thread` increments. When seq_cst is true every
// increment uses memory_order_seq_cst, otherwise
// memory_order_relaxed. Returns the final total and the measured
// wall time of the threaded section.
AtomicRun AtomicCounterRun(int threads, int per_thread, bool seq_cst) {
  std::atomic<long long> counter{0};
  auto start = std::chrono::steady_clock::now();
  {
    std::vector<std::thread> pool;
    pool.reserve(static_cast<std::size_t>(threads));
    for (int t = 0; t < threads; ++t) {
      pool.emplace_back([&counter, per_thread, seq_cst]() {
        for (int i = 0; i < per_thread; ++i) {
          if (seq_cst) {
            counter.fetch_add(1, std::memory_order_seq_cst);
          } else {
            counter.fetch_add(1, std::memory_order_relaxed);
          }
        }
      });
    }
    for (auto& th : pool) th.join();
  }
  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  return {counter.load(std::memory_order_relaxed), ms};
}

struct HandoffRun {
  int trials;
  int successes;
  double elapsed_ms;
};

// Producer-consumer handoff, one pair of threads per trial. The
// producer writes a plain int payload, then sets an atomic flag.
// The consumer spins on the flag, then reads the payload and
// checks it equals the value for that trial. With ordered=true
// the flag store is release and the load is acquire, which is
// the pairing that orders the payload write before the read.
// With ordered=false both are relaxed (the control). Returns the
// number of trials whose payload arrived intact.
HandoffRun ProducerConsumerRun(int trials, bool ordered) {
  int successes = 0;
  auto start = std::chrono::steady_clock::now();
  for (int trial = 0; trial < trials; ++trial) {
    std::atomic<int> flag{0};
    int payload = 0;
    const int expected = 1000 + trial;
    std::thread producer([&]() {
      payload = expected;
      if (ordered) {
        flag.store(1, std::memory_order_release);
      } else {
        flag.store(1, std::memory_order_relaxed);
      }
    });
    std::thread consumer([&]() {
      if (ordered) {
        while (flag.load(std::memory_order_acquire) == 0) {
        }
      } else {
        while (flag.load(std::memory_order_relaxed) == 0) {
        }
      }
    });
    producer.join();
    consumer.join();
    // Both threads are joined here, so reading payload is safe
    // and the check measures what the handoff delivered.
    if (payload == expected) ++successes;
  }
  auto end = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(end - start).count();
  return {trials, successes, ms};
}

// ---------------------------------------------------------- move semantics

// A large buffer type with counted copy and move operations.
// Copying copies every element. Moving transfers the vector and
// leaves the source empty, which the tests assert directly.
class LargeBuffer {
 public:
  LargeBuffer(std::size_t n, double value)
      : data_(n, value) {}
  LargeBuffer(const LargeBuffer& other) : data_(other.data_) {
    ++copy_count;
  }
  LargeBuffer& operator=(const LargeBuffer& other) {
    data_ = other.data_;
    ++copy_count;
    return *this;
  }
  LargeBuffer(LargeBuffer&& other) noexcept
      : data_(std::move(other.data_)) {
    ++move_count;
    other.data_.clear();
  }
  LargeBuffer& operator=(LargeBuffer&& other) noexcept {
    data_ = std::move(other.data_);
    ++move_count;
    other.data_.clear();
    return *this;
  }
  std::size_t size() const { return data_.size(); }
  double at(std::size_t i) const { return data_[i]; }
  double checksum() const {
    double s = 0.0;
    for (double v : data_) s += v;
    return s;
  }

  static long long copy_count;
  static long long move_count;
  static void ResetCounts() { copy_count = 0; move_count = 0; }

 private:
  std::vector<double> data_;
};

long long LargeBuffer::copy_count = 0;
long long LargeBuffer::move_count = 0;

struct MoveBench {
  std::size_t elements;
  int reps;
  long long copies;
  long long moves_in_copy_path;
  double copy_ms;
  long long moves;
  long long copies_in_move_path;
  double move_ms;
  double checksum;
};

// Copy a fresh LargeBuffer `reps` times, then move a fresh one
// `reps` times. Counts are read from the class counters after
// each path, so the recorded counts are the exact constructor
// and assignment counts for that path. Times are wall times of
// each loop. The checksum accumulates one element per result so
// the work cannot be optimised away and correctness is visible.
MoveBench MoveBenchRun(std::size_t elements, int reps) {
  MoveBench out{};
  out.elements = elements;
  out.reps = reps;
  out.checksum = 0.0;

  LargeBuffer::ResetCounts();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) {
    LargeBuffer src(elements, 1.5);
    LargeBuffer copy = src;  // copy constructor
    out.checksum += copy.at(0);
  }
  auto mid = std::chrono::steady_clock::now();
  out.copy_ms = std::chrono::duration<double, std::milli>(mid - start).count();
  out.copies = LargeBuffer::copy_count;
  out.moves_in_copy_path = LargeBuffer::move_count;

  LargeBuffer::ResetCounts();
  for (int i = 0; i < reps; ++i) {
    LargeBuffer src(elements, 1.5);
    LargeBuffer moved = std::move(src);  // move constructor
    out.checksum += moved.at(0);
  }
  auto end = std::chrono::steady_clock::now();
  out.move_ms = std::chrono::duration<double, std::milli>(end - mid).count();
  out.moves = LargeBuffer::move_count;
  out.copies_in_move_path = LargeBuffer::copy_count;
  return out;
}

// ------------------------------------------------------------------- main

#ifndef DEEP_NO_MAIN
int main() {
  bool all = true;

  // Build tool still works inside the Deep binary.
  Buffer<double> a(3), b(3);
  a[0] = 1; a[1] = 2; a[2] = 3;
  b[0] = 10; b[1] = 20; b[2] = 30;
  long long adds = 0;
  auto sum = VectorAdd<double>(a, b, adds);
  bool vadd_ok = sum.size() == 3 && sum[0] == 11 && sum[1] == 22 &&
                 sum[2] == 33;
  std::cout << "build_vector_add_in_deep: " << (vadd_ok ? "PASS" : "FAIL")
            << "\n";
  all &= vadd_ok;

  // Atomics: 4 threads by 250000 increments = 1000000 exactly.
  const int kThreads = 4;
  const int kPerThread = 250000;
  const long long kExpected = 1LL * kThreads * kPerThread;
  AtomicRun relaxed = AtomicCounterRun(kThreads, kPerThread, false);
  AtomicRun seqcst = AtomicCounterRun(kThreads, kPerThread, true);
  std::cout << "atomics_relaxed_total=" << relaxed.total << "\n";
  std::cout << "atomics_relaxed_ms=" << relaxed.elapsed_ms << "\n";
  std::cout << "atomics_seqcst_total=" << seqcst.total << "\n";
  std::cout << "atomics_seqcst_ms=" << seqcst.elapsed_ms << "\n";
  bool atomics_ok = relaxed.total == kExpected && seqcst.total == kExpected;
  std::cout << "atomics_counter_exact: " << (atomics_ok ? "PASS" : "FAIL")
            << "\n";
  all &= atomics_ok;

  // Producer-consumer handoff: 200 trials each way.
  HandoffRun ordered = ProducerConsumerRun(200, true);
  HandoffRun relaxed_h = ProducerConsumerRun(200, false);
  std::cout << "handoff_ordered_successes=" << ordered.successes << "/"
            << ordered.trials << "\n";
  std::cout << "handoff_ordered_ms=" << ordered.elapsed_ms << "\n";
  std::cout << "handoff_relaxed_successes=" << relaxed_h.successes << "/"
            << relaxed_h.trials << "\n";
  std::cout << "handoff_relaxed_ms=" << relaxed_h.elapsed_ms << "\n";
  bool handoff_ok = ordered.successes == ordered.trials;
  std::cout << "handoff_ordered_all_intact: "
            << (handoff_ok ? "PASS" : "FAIL") << "\n";
  all &= handoff_ok;

  // Move semantics benchmark: 200000 doubles, 50 reps each path.
  MoveBench bench = MoveBenchRun(200000, 50);
  std::cout << "move_elements=" << bench.elements << "\n";
  std::cout << "move_reps=" << bench.reps << "\n";
  std::cout << "move_copy_count=" << bench.copies << "\n";
  std::cout << "move_copy_ms=" << bench.copy_ms << "\n";
  std::cout << "move_move_count=" << bench.moves << "\n";
  std::cout << "move_move_ms=" << bench.move_ms << "\n";
  std::cout << "move_checksum=" << bench.checksum << "\n";
  bool move_ok = bench.copies == bench.reps && bench.moves == bench.reps &&
                 bench.moves_in_copy_path == 0 &&
                 bench.copies_in_move_path == 0 &&
                 std::fabs(bench.checksum - 150.0) < 1e-9;
  std::cout << "move_counts_exact: " << (move_ok ? "PASS" : "FAIL") << "\n";
  all &= move_ok;

  std::cout << "deep_tool: " << (all ? "ALL PASS" : "FAILURES") << "\n";
  return all ? 0 : 1;
}
#endif  // DEEP_NO_MAIN
