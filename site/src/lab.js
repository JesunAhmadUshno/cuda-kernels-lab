// Browser-side math for the dashboard. Ports of level-1-learn/kernels.py and of the
// throughput formulas in level-2-build/analysis.py and level-3-deep/analysis_deep.py.
// Checked against the repo's result files by lab.check.mjs.

const cdiv = (a, b) => Math.ceil(a / b);

// Naive matmul: one A and one B element per multiply, no reuse.
export const naiveLoads = (m, n, k) => 2 * m * n * k;
// Tiled matmul as kernels.py counts it: full tile slots, zero-padded edges included.
export const tiledSlots = (m, n, k, t) => 2 * t * t * cdiv(m, t) * cdiv(n, t) * cdiv(k, t);
// Same loop, counting only the guarded loads that land inside the matrix.
// Each A element is loaded once per block column, each B element once per block row.
export const tiledInBounds = (m, n, k, t) => m * k * cdiv(n, t) + k * n * cdiv(m, t);
export const multiplies = (m, n, k) => m * n * k;
export const adds = (m, n, k) => (k > 0 ? m * n * (k - 1) : 0);

// run_learn.py det_matrix: deterministic small integers, no randomness.
export const detMatrix = (n, seed) =>
  Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => ((i * n + j + seed) % 7) - 3));

export function matmulNaive(a, b) {
  const m = a.length, k = b.length, n = b[0].length;
  const c = Array.from({ length: m }, (_, i) => Array.from({ length: n }, (_, j) => {
    let total = 0;
    for (let p = 0; p < k; p++) total += a[i][p] * b[p][j];
    return total;
  }));
  return { c, counts: { multiplies: multiplies(m, n, k), adds: adds(m, n, k), element_loads: naiveLoads(m, n, k) } };
}

// Line-for-line port of kernels.matmul_tiled, plus an in-bounds load counter.
export function matmulTiled(a, b, tile) {
  const m = a.length, k = b.length, n = b[0].length;
  const c = Array.from({ length: m }, () => Array(n).fill(0));
  let tileLoads = 0, tilesLoaded = 0, inBounds = 0;
  for (let br = 0; br < cdiv(m, tile); br++) for (let bc = 0; bc < cdiv(n, tile); bc++) {
    const acc = Array.from({ length: tile }, () => Array(tile).fill(0));
    for (let t = 0; t < cdiv(k, tile); t++) {
      const at = Array.from({ length: tile }, () => Array(tile).fill(0));
      const bt = Array.from({ length: tile }, () => Array(tile).fill(0));
      for (let i = 0; i < tile; i++) for (let j = 0; j < tile; j++) {
        const gi = br * tile + i, gj = t * tile + j;
        if (gi < m && gj < k) { at[i][j] = a[gi][gj]; inBounds++; }
        const gj2 = bc * tile + j, gi2 = t * tile + i;
        if (gi2 < k && gj2 < n) { bt[i][j] = b[gi2][gj2]; inBounds++; }
      }
      tilesLoaded += 2;
      tileLoads += 2 * tile * tile;
      for (let i = 0; i < tile; i++) for (let j = 0; j < tile; j++) {
        let total = acc[i][j];
        for (let p = 0; p < tile; p++) total += at[i][p] * bt[p][j];
        acc[i][j] = total;
      }
    }
    for (let i = 0; i < tile; i++) for (let j = 0; j < tile; j++) {
      const gi = br * tile + i, gj = bc * tile + j;
      if (gi < m && gj < n) c[gi][gj] = acc[i][j];
    }
  }
  return { c, counts: { multiplies: multiplies(m, n, k), adds: adds(m, n, k), tile_loads: tileLoads, tiles_loaded: tilesLoaded, in_bounds_loads: inBounds } };
}

// Softmax in IEEE float64 (JavaScript numbers). Naive overflows to Infinity, then NaN.
export function softmaxNaive(x) {
  const e = x.map(Math.exp);
  const total = e.reduce((s, v) => s + v, 0);
  return e.map((v) => v / total);
}
export function softmaxStable(x) {
  const peak = Math.max(...x);
  const e = x.map((v) => Math.exp(v - peak));
  const total = e.reduce((s, v) => s + v, 0);
  return e.map((v) => v / total);
}
// Largest x with finite exp(x): ln of the largest finite float.
// float32 max = (2 - 2^-23) * 2^127, float64 max = Number.MAX_VALUE.
export const overflowAt = (bits) => Math.log(bits === 32 ? (2 - 2 ** -23) * 2 ** 127 : Number.MAX_VALUE);

// Throughput models from analysis.py / analysis_deep.py.
export const matmulGflops = (size, ms) => (2 * size ** 3) / (ms * 1e-3) / 1e9;
export const softmaxGBs = (rows, cols, ms, passes = 2) => (passes * rows * cols * 4) / (ms * 1e-3) / 1e9;
export const vectorGBs = (n, ms) => (3 * n * 4) / (ms * 1e-3) / 1e9;
export function recompute(row) {
  if (row.kernel === 'matmul_torch') return { unit: 'GFLOP/s', stored: row.gflops, recomputed: matmulGflops(row.size, row.median_ms) };
  if (row.kernel === 'vector_add_torch') return { unit: 'GB/s', stored: row.gb_per_s, recomputed: vectorGBs(row.n, row.median_ms) };
  return { unit: 'GB/s', stored: row.gb_per_s, recomputed: softmaxGBs(row.rows, row.cols, row.median_ms) };
}
export const relDiff = (a, b) => Math.abs(a - b) / Math.abs(b);

// Traffic model: passes over a rows x cols float32 matrix at a given bandwidth.
// Fused minimum = 1 read + 1 write = 2; fused_softmax.cu = 2 reads + 1 write = 3.
export const predictMs = (rows, cols, passes, gbs) => ((passes * rows * cols * 4) / (gbs * 1e9)) * 1e3;
// Unfused op sequence, global passes per op (read + write of a full matrix; row vectors ignored).
export const UNFUSED_OPS = [['amax', 1], ['sub', 2], ['exp', 2], ['sum', 1], ['div', 2]];
export const unfusedPasses = () => UNFUSED_OPS.reduce((s, [, p]) => s + p, 0);
export const MiB = (bytes) => bytes / 2 ** 20;
