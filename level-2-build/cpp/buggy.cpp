// buggy.cpp - deliberately buggy tiled matmul for the Level 2
// debugging session. DO NOT USE as a kernel. See ../DEBUG-SESSION.md.
//
// The bug: the tile loop below is missing the bounds guard that
// tool.cpp and ../cuda/kernels.cu carry:
//
//   tool.cpp:      if (gi >= n || gj >= n) continue;
//   kernels.cu:    (row < n && acol < n) ? a[...] : 0.0f
//                  if (row < n && col < n) c[...] = sum;
//
// At a size that is not a multiple of the tile (here n = 3, tile =
// 2), the edge tile covers positions with gi = 3 or gj = 3. The
// loads read from zero padded source buffers sized to the padded
// dimension, so the reads stay in bounds and the product terms for
// padded positions are zero. The store has no guard, so the edge
// tile writes c[gi * n + gj] for gi or gj equal to 3, past the end
// of the heap allocation for c (9 doubles). AddressSanitizer
// reports that write as a heap-buffer-overflow. That is the real
// crash the debugging session is written up from.
//
// Build for the session:
//   g++ -std=c++17 -fsanitize=address,undefined -g buggy.cpp -o buggy

#include <cstddef>
#include <iostream>
#include <vector>

// Tiled matmul, n by n, with the bounds guard missing on the store.
static double* MatmulTiledBuggy(const std::vector<double>& a,
                                const std::vector<double>& b, int n,
                                int tile) {
  const int padded = ((n + tile - 1) / tile) * tile;
  // Zero padded sources, indexed with the padded stride, so tile
  // loads at edge positions read 0 and stay inside an allocation.
  std::vector<double> ap(static_cast<std::size_t>(padded) * padded, 0.0);
  std::vector<double> bp(static_cast<std::size_t>(padded) * padded, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      ap[static_cast<std::size_t>(i) * padded + j] =
          a[static_cast<std::size_t>(i) * n + j];
      bp[static_cast<std::size_t>(i) * padded + j] =
          b[static_cast<std::size_t>(i) * n + j];
    }
  }
  double* c = new double[static_cast<std::size_t>(n) * n]();
  std::vector<double> a_tile(static_cast<std::size_t>(tile) * tile);
  std::vector<double> b_tile(static_cast<std::size_t>(tile) * tile);
  for (int br = 0; br < n; br += tile) {
    for (int bc = 0; bc < n; bc += tile) {
      std::vector<double> acc(static_cast<std::size_t>(tile) * tile, 0.0);
      for (int t = 0; t < n; t += tile) {
        for (int i = 0; i < tile; ++i) {
          for (int j = 0; j < tile; ++j) {
            int gi = br + i, gj = t + j;
            a_tile[static_cast<std::size_t>(i) * tile + j] =
                ap[static_cast<std::size_t>(gi) * padded + gj];
            int gi2 = t + i, gj2 = bc + j;
            b_tile[static_cast<std::size_t>(i) * tile + j] =
                bp[static_cast<std::size_t>(gi2) * padded + gj2];
          }
        }
        for (int i = 0; i < tile; ++i) {
          for (int j = 0; j < tile; ++j) {
            double total = acc[static_cast<std::size_t>(i) * tile + j];
            for (int p = 0; p < tile; ++p) {
              total += a_tile[static_cast<std::size_t>(i) * tile + p] *
                       b_tile[static_cast<std::size_t>(p) * tile + j];
            }
            acc[static_cast<std::size_t>(i) * tile + j] = total;
          }
        }
      }
      for (int i = 0; i < tile; ++i) {
        for (int j = 0; j < tile; ++j) {
          int gi = br + i, gj = bc + j;
          // BUG: no (gi < n && gj < n) guard here. When the edge
          // tile overruns the matrix, this writes past c.
          c[static_cast<std::size_t>(gi) * n + gj] =
              acc[static_cast<std::size_t>(i) * tile + j];
        }
      }
    }
  }
  return c;
}

int main() {
  const int n = 3;  // not a multiple of tile 2: the edge case
  std::vector<double> a = {2, 0, 1, 1, 3, 2, 0, 1, 1};
  std::vector<double> b = {1, 2, 0, 0, 1, 3, 2, 1, 1};
  double* c = MatmulTiledBuggy(a, b, n, 2);
  std::cout << "buggy matmul finished, c[0]=" << c[0] << "\n";
  delete[] c;
  return 0;
}
