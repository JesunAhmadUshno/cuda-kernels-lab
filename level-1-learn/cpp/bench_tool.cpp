// bench_tool.cpp - standalone C++17 kernel launch and benchmark runner
// for cuda-kernels-lab Level 1 Learn, CPU reference edition.
//
// Teaches the host side of the CUDA pattern: an RAII buffer owning its
// memory with a smart pointer, a templated kernel runner, and STL
// containers. The kernels run on the CPU and mirror the CUDA indexing
// (one element per virtual thread for vector add, tiled blocks for
// matmul). No CUDA GPU ran this cycle, so no GPU timing is claimed.
// The tool checks its own results and prints PASS lines with exact
// operation counts (counts only, no wall time in claimed results).

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// RAII buffer: owns a heap array through unique_ptr, frees on scope
// exit, move only (no accidental copies of owned memory).
template <typename T>
class Buffer {
 public:
  explicit Buffer(std::size_t n) : data_(std::make_unique<T[]>(n)), size_(n) {}
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer(Buffer&&) noexcept = default;
  Buffer& operator=(Buffer&&) noexcept = default;
  T& operator[](std::size_t i) { return data_[i]; }
  const T& operator[](std::size_t i) const { return data_[i]; }
  std::size_t size() const { return size_; }
  T* data() { return data_.get(); }

 private:
  std::unique_ptr<T[]> data_;
  std::size_t size_;
};

// Templated kernel runner: runs a kernel callable over a named case,
// counts the operations the callable reports, and verifies the result
// against an expected vector within a tolerance.
template <typename T>
struct KernelRunner {
  static bool Run(const std::string& name, const std::vector<T>& got,
                  const std::vector<T>& want, long long ops) {
    bool ok = got.size() == want.size();
    for (std::size_t i = 0; ok && i < got.size(); ++i) {
      if (std::fabs(static_cast<double>(got[i] - want[i])) > 1e-9) ok = false;
    }
    std::cout << name << ": " << (ok ? "PASS" : "FAIL") << " ops=" << ops
              << "\n";
    return ok;
  }
};

// Vector add kernel: virtual thread i computes c[i] = a[i] + b[i].
template <typename T>
std::vector<T> VectorAdd(const Buffer<T>& a, const Buffer<T>& b,
                         long long& adds) {
  std::vector<T> c;
  c.reserve(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) c.push_back(a[i] + b[i]);
  adds = static_cast<long long>(a.size());
  return c;
}

// Tiled matmul on square matrices, tile by tile blocks with local
// tile arrays standing in for CUDA shared memory.
template <typename T>
std::vector<T> MatmulTiled(const std::vector<T>& a, const std::vector<T>& b,
                           int n, int tile, long long& tile_loads) {
  std::vector<T> c(static_cast<std::size_t>(n) * n, T{0});
  tile_loads = 0;
  for (int br = 0; br < n; br += tile) {
    for (int bc = 0; bc < n; bc += tile) {
      for (int t = 0; t < n; t += tile) {
        T a_tile[64];
        T b_tile[64];
        for (int i = 0; i < tile; ++i) {
          for (int j = 0; j < tile; ++j) {
            int gi = br + i, gj = t + j;
            a_tile[i * tile + j] =
                (gi < n && gj < n) ? a[static_cast<std::size_t>(gi) * n + gj] : T{0};
            int gi2 = t + i, gj2 = bc + j;
            b_tile[i * tile + j] =
                (gi2 < n && gj2 < n) ? b[static_cast<std::size_t>(gi2) * n + gj2] : T{0};
          }
        }
        tile_loads += 2LL * tile * tile;
        for (int i = 0; i < tile; ++i) {
          for (int j = 0; j < tile; ++j) {
            int gi = br + i, gj = bc + j;
            if (gi >= n || gj >= n) continue;
            T total = c[static_cast<std::size_t>(gi) * n + gj];
            for (int p = 0; p < tile; ++p) {
              total += a_tile[i * tile + p] * b_tile[p * tile + j];
            }
            c[static_cast<std::size_t>(gi) * n + gj] = total;
          }
        }
      }
    }
  }
  return c;
}

int main() {
  bool all = true;

  Buffer<double> a(3), b(3);
  a[0] = 1; a[1] = 2; a[2] = 3;
  b[0] = 10; b[1] = 20; b[2] = 30;
  long long adds = 0;
  auto sum = VectorAdd<double>(a, b, adds);
  all &= KernelRunner<double>::Run("vector_add", sum, {11, 22, 33}, adds);

  // 2x2 hand case: [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]].
  std::vector<double> m1 = {1, 2, 3, 4};
  std::vector<double> m2 = {5, 6, 7, 8};
  long long loads = 0;
  auto prod = MatmulTiled<double>(m1, m2, 2, 2, loads);
  all &= KernelRunner<double>::Run("matmul_tiled_2x2", prod,
                                   {19, 22, 43, 50}, loads);

  std::cout << "bench_tool: " << (all ? "ALL PASS" : "FAILURES") << "\n";
  return all ? 0 : 1;
}
