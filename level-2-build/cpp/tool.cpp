// tool.cpp - standalone C++17 tool for cuda-kernels-lab Level 2 Build.
//
// Evolved from the Level 1 Learn bench tool (level-1-learn/cpp/
// bench_tool.cpp). Same math as Learn: vector add (one element per
// virtual thread), tiled matmul (tile loads counted, guarded edge
// loads with zero padding), and stable softmax (max subtraction,
// added at Build). RAII Buffer owning memory via unique_ptr,
// templated kernels and runner, STL containers.
//
// The kernels run on the CPU and mirror the CUDA indexing in
// ../cuda/kernels.cu. No custom CUDA kernel result is claimed from
// this file. The GPU numbers in the level README come from PyTorch
// native ops measured on the fleet GPU node (see ../README.md).
//
// main() is guarded by TOOL_NO_MAIN so cpp/tests.cpp can include
// this file and test the same functions the tool runs.

#include <cmath>
#include <cstddef>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// RAII buffer: owns a heap array through unique_ptr, move only.
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

// Naive matmul on square matrices: one output element at a time,
// full row and column walks. Reference for the tiled form.
template <typename T>
std::vector<T> MatmulNaive(const std::vector<T>& a, const std::vector<T>& b,
                           int n, long long& element_loads) {
  std::vector<T> c(static_cast<std::size_t>(n) * n, T{0});
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      T total = T{0};
      for (int p = 0; p < n; ++p) {
        total += a[static_cast<std::size_t>(i) * n + p] *
                 b[static_cast<std::size_t>(p) * n + j];
      }
      c[static_cast<std::size_t>(i) * n + j] = total;
    }
  }
  element_loads = 2LL * n * n * n;
  return c;
}

// Tiled matmul on square matrices. Tile loads are counted as element
// slots loaded, including zero padded edge slots, matching Learn and
// matching the guarded cooperative load in ../cuda/kernels.cu:
//
//   as[ty][tx] = (row < n && acol < n) ? a[row * n + acol] : 0
//   bs[ty][tx] = (brow < n && col < n) ? b[brow * n + col] : 0
//
// tile_loads = blocks_row * blocks_col * k_tiles * 2 * tile * tile
// where each block count is ceil(n / tile). The output store is
// guarded the same way: positions outside the n by n result are
// never written. That guard is the fix recorded in DEBUG-SESSION.md.
template <typename T>
std::vector<T> MatmulTiled(const std::vector<T>& a, const std::vector<T>& b,
                           int n, int tile, long long& tile_loads) {
  std::vector<T> c(static_cast<std::size_t>(n) * n, T{0});
  tile_loads = 0;
  std::vector<T> a_tile(static_cast<std::size_t>(tile) * tile);
  std::vector<T> b_tile(static_cast<std::size_t>(tile) * tile);
  for (int br = 0; br < n; br += tile) {
    for (int bc = 0; bc < n; bc += tile) {
      for (int t = 0; t < n; t += tile) {
        for (int i = 0; i < tile; ++i) {
          for (int j = 0; j < tile; ++j) {
            int gi = br + i, gj = t + j;
            a_tile[static_cast<std::size_t>(i) * tile + j] =
                (gi < n && gj < n)
                    ? a[static_cast<std::size_t>(gi) * n + gj]
                    : T{0};
            int gi2 = t + i, gj2 = bc + j;
            b_tile[static_cast<std::size_t>(i) * tile + j] =
                (gi2 < n && gj2 < n)
                    ? b[static_cast<std::size_t>(gi2) * n + gj2]
                    : T{0};
          }
        }
        tile_loads += 2LL * tile * tile;
        for (int i = 0; i < tile; ++i) {
          for (int j = 0; j < tile; ++j) {
            int gi = br + i, gj = bc + j;
            if (gi >= n || gj >= n) continue;  // bounds guard (the fix)
            T total = c[static_cast<std::size_t>(gi) * n + gj];
            for (int p = 0; p < tile; ++p) {
              total += a_tile[static_cast<std::size_t>(i) * tile + p] *
                       b_tile[static_cast<std::size_t>(p) * tile + j];
            }
            c[static_cast<std::size_t>(gi) * n + gj] = total;
          }
        }
      }
    }
  }
  return c;
}

// Stable softmax of one row: subtract the row maximum first, so the
// largest exponent is exp(0) = 1 and nothing overflows. Same math as
// Learn softmax_stable and the softmax_rows kernel in kernels.cu.
template <typename T>
std::vector<T> SoftmaxStable(const std::vector<T>& x) {
  T peak = x[0];
  for (T v : x) peak = v > peak ? v : peak;
  std::vector<T> out;
  out.reserve(x.size());
  double total = 0.0;
  for (T v : x) {
    double e = std::exp(static_cast<double>(v - peak));
    out.push_back(static_cast<T>(e));
    total += e;
  }
  for (T& v : out) v = static_cast<T>(v / total);
  return out;
}

// Stable softmax applied row by row to a rows by cols matrix.
template <typename T>
std::vector<T> SoftmaxRows(const std::vector<T>& x, int rows, int cols) {
  std::vector<T> out;
  out.reserve(x.size());
  for (int r = 0; r < rows; ++r) {
    std::vector<T> row(x.begin() + r * cols, x.begin() + (r + 1) * cols);
    std::vector<T> done = SoftmaxStable<T>(row);
    out.insert(out.end(), done.begin(), done.end());
  }
  return out;
}

// Self-check runner used by main: compares against expected values.
template <typename T>
bool CheckVec(const std::string& name, const std::vector<T>& got,
              const std::vector<T>& want, double tol = 1e-9) {
  bool ok = got.size() == want.size();
  for (std::size_t i = 0; ok && i < got.size(); ++i) {
    if (std::fabs(static_cast<double>(got[i] - want[i])) > tol) ok = false;
  }
  std::cout << name << ": " << (ok ? "PASS" : "FAIL") << "\n";
  return ok;
}

#ifndef TOOL_NO_MAIN
int main() {
  bool all = true;

  Buffer<double> a(3), b(3);
  a[0] = 1; a[1] = 2; a[2] = 3;
  b[0] = 10; b[1] = 20; b[2] = 30;
  long long adds = 0;
  auto sum = VectorAdd<double>(a, b, adds);
  all &= CheckVec<double>("vector_add", sum, {11, 22, 33});
  std::cout << "vector_add adds=" << adds << "\n";

  // 2x2 hand case: [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]].
  std::vector<double> m1 = {1, 2, 3, 4};
  std::vector<double> m2 = {5, 6, 7, 8};
  long long loads = 0;
  auto prod = MatmulTiled<double>(m1, m2, 2, 2, loads);
  all &= CheckVec<double>("matmul_tiled_2x2", prod, {19, 22, 43, 50});
  std::cout << "matmul_tiled_2x2 tile_loads=" << loads << "\n";

  // Edge size: 3x3 at tile 2 (3 is not a multiple of 2).
  std::vector<double> m3a = {2, 0, 1, 1, 3, 2, 0, 1, 1};
  std::vector<double> m3b = {1, 2, 0, 0, 1, 3, 2, 1, 1};
  long long loads3 = 0;
  auto prod3 = MatmulTiled<double>(m3a, m3b, 3, 2, loads3);
  all &= CheckVec<double>("matmul_tiled_3x3_tile2_edge", prod3,
                          {4, 5, 1, 5, 7, 11, 2, 2, 4});
  std::cout << "matmul_tiled_3x3_tile2_edge tile_loads=" << loads3 << "\n";

  auto sm = SoftmaxStable<double>({1000.0, 1001.0, 999.0});
  double row_sum = sm[0] + sm[1] + sm[2];
  bool sm_ok = std::fabs(row_sum - 1.0) < 1e-12 && sm[1] > sm[0] &&
               sm[0] > sm[2];
  std::cout << "softmax_stable_large: " << (sm_ok ? "PASS" : "FAIL")
            << " row_sum=" << row_sum << "\n";
  all &= sm_ok;

  std::cout << "tool: " << (all ? "ALL PASS" : "FAILURES") << "\n";
  return all ? 0 : 1;
}
#endif  // TOOL_NO_MAIN
