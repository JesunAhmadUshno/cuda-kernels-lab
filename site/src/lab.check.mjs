// Known-answer self-check, run by `npm test` and before every `npm run build`.
// Every formula the page computes in the browser is checked here against the
// repo's own result files and write-ups, so the site cannot drift from the repo.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {
  naiveLoads, tiledSlots, tiledInBounds, matmulNaive, matmulTiled, detMatrix,
  softmaxNaive, softmaxStable, overflowAt, recompute, relDiff, predictMs,
  unfusedPasses, MiB,
} from './lab.js';

const read = (p) => fs.readFileSync(new URL(`../../${p}`, import.meta.url), 'utf8');
const json = (p) => JSON.parse(read(p));
const learn = json('level-1-learn/results_learn.json');
const prof = json('level-3-deep/results_profiler.json');
const bench = json('level-2-build/results_torch_bench.json');
const deep = json('level-3-deep/results_deep.json');

// 1. Level 1 load table: closed forms AND a line-for-line port of kernels.py agree with all rows.
for (const r of learn.load_table) {
  const { size: n, tile } = r;
  assert.equal(naiveLoads(n, n, n), r.naive_element_loads, `naive ${n}`);
  assert.equal(tiledSlots(n, n, n, tile), r.tiled_tile_loads, `tiled ${n}`);
  const a = detMatrix(n, 1), b = detMatrix(n, 2);
  const nv = matmulNaive(a, b), tl = matmulTiled(a, b, tile);
  assert.equal(nv.counts.element_loads, r.naive_element_loads);
  assert.equal(tl.counts.tile_loads, r.tiled_tile_loads);
  assert.equal(nv.counts.multiplies, r.multiplies);
  assert.equal(nv.counts.adds, r.adds);
  assert.equal(tl.counts.multiplies, r.multiplies);
  assert.deepEqual(nv.c, tl.c, `results identical ${n}`);
  assert.equal(r.results_identical, true);
}
// Hand case from run_learn.py.
assert.deepEqual(matmulTiled([[1, 2], [3, 4]], [[5, 6], [7, 8]], 2).c, [[19, 22], [43, 50]]);

// 2. In-bounds tiled count: closed form equals a brute-force count of the guarded loads in the port,
// never exceeds the padded count, and equals it when the tile divides every dimension.
for (const [m, n, k, t] of [[3, 3, 3, 2], [5, 7, 3, 2], [16, 16, 16, 4], [10, 6, 9, 4], [1, 1, 1, 8]]) {
  const a = Array.from({ length: m }, () => Array(k).fill(1));
  const b = Array.from({ length: k }, () => Array(n).fill(1));
  const { counts } = matmulTiled(a, b, t);
  assert.equal(tiledInBounds(m, n, k, t), counts.in_bounds_loads, `in-bounds ${m}x${n}x${k}/${t}`);
  assert.equal(tiledSlots(m, n, k, t), counts.tile_loads);
  assert.ok(counts.in_bounds_loads <= counts.tile_loads);
  if (m % t === 0 && n % t === 0 && k % t === 0) assert.equal(counts.in_bounds_loads, counts.tile_loads);
}

// 3. Softmax: the run_learn.py cases.
const s = softmaxStable([1000, 1001, 999]);
assert.ok(Math.abs(s.reduce((x, y) => x + y) - 1) < 1e-12 && s[1] > s[0] && s[0] > s[2]);
assert.ok(softmaxNaive([1000, 1001]).some((v) => !Number.isFinite(v)), 'naive overflows');
const base = softmaxStable([1, 2, 3]), shifted = softmaxStable([501, 502, 503]);
base.forEach((v, i) => assert.ok(Math.abs(v - shifted[i]) < 1e-12, 'shift invariance'));
// learn README: "exp overflows near 709"; Deep notes float32 overflows near 88.7.
assert.equal(Math.floor(overflowAt(64)), 709);
assert.equal(Math.round(overflowAt(32) * 10) / 10, 88.7);

// 4. Every recorded GPU throughput recomputes within 1 percent (analysis.py, analysis_deep.py).
for (const row of [...prof.benchmarks, ...bench.benchmarks]) {
  const { stored, recomputed } = recompute(row);
  assert.ok(relDiff(recomputed, stored) <= 0.01, `${row.kernel} ${recomputed} vs ${stored}`);
  assert.equal(row.correct, true);
}
assert.equal(prof.benchmarks.length, deep.gpu_benchmark_rows_verified);

// 5. Fused-kernel prediction reproduces ANALYSIS.md, and the headline ratio reproduces the README.
const native = prof.benchmarks.find((r) => r.kernel === 'softmax_rows_torch' && r.rows === 4096);
const unfused = prof.benchmarks.find((r) => r.kernel === 'softmax_rows_unfused_manual' && r.rows === 4096);
const analysis = read('level-3-deep/cuda/ANALYSIS.md');
const predicted = analysis.match(/median near (\d+\.\d+) ms/)[1];
assert.equal(predictMs(4096, 1024, 3, native.gb_per_s).toFixed(3), predicted);
assert.equal((unfused.median_ms / native.median_ms).toFixed(1), read('README.md').match(/a factor\s+of (\d+\.\d)/)[1]);
// "each one materialises a full 16 MiB intermediate"
assert.equal(MiB(4096 * 1024 * 4), 16);
// Per-route model: amax 1 + sub 2 + exp 2 + sum 1 + div 2 = 8 passes.
assert.equal(unfusedPasses(), 8);

// 6. Move benchmark: "about 24 times faster on this run".
assert.equal(Math.round(deep.move_bench.copy_ms / deep.move_bench.move_ms), 24);

console.log('lab ok', {
  loadRows: learn.load_table.length,
  gpuRows: prof.benchmarks.length + bench.benchmarks.length,
  predictedMs: predicted,
});
