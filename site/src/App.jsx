import { useMemo, useState, useEffect, useId } from 'react';
import learn from '../../level-1-learn/results_learn.json';
import prof from '../../level-3-deep/results_profiler.json';
import bench from '../../level-2-build/results_torch_bench.json';
import build from '../../level-2-build/results_build.json';
import deep from '../../level-3-deep/results_deep.json';
import readmeMd from '../../README.md?raw';
import learnMd from '../../level-1-learn/README.md?raw';
import buildMd from '../../level-2-build/README.md?raw';
import deepMd from '../../level-3-deep/README.md?raw';
import debugMd from '../../level-2-build/DEBUG-SESSION.md?raw';
import roadmapMd from '../../ROADMAP.md?raw';
import {
  naiveLoads, tiledSlots, tiledInBounds, multiplies, adds, detMatrix, matmulNaive, matmulTiled,
  softmaxNaive, softmaxStable, overflowAt, recompute, relDiff, softmaxGBs, predictMs,
  UNFUSED_OPS, unfusedPasses, MiB,
} from './lab.js';

const REPO = 'https://github.com/JesunAhmadUshno/cuda-kernels-lab';
const blob = (p) => `${REPO}/blob/main/${p}`;
const fmt = (n) => Math.round(n).toLocaleString('en-US');
const fx = (n, d = 2) => Number(n).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

// ---- facts read from the repo at build time ----
const RUN_DATE = deepMd.match(/Status: DONE, (\d{4}-\d{2}-\d{2})/)[1];
const grade = (md) => md.match(/^(\d,\d{3}) of 10,000/m)[1];
const GRADES = [['Learn', grade(learnMd)], ['Build', grade(buildMd)], ['Deep', grade(deepMd)]];
const PY_TESTS = readmeMd.match(/(\d+) Python tests pass/)[1];
const DEBUG_CASE = debugMd.match(/n = (\d+) at tile = (\d+)/);
const ASAN = debugMd.match(/SUMMARY: AddressSanitizer: (\S+) \S*?(buggy\.cpp:\d+)/);
const ROADMAP = [...roadmapMd.split('## v1.1 backlog, in order')[1].split('## Standing rules')[0].matchAll(/^(\d+)\. (.+?)\.(?=\s)/gm)].map((m) => [m[1], m[2]]);
const DEVICE_SHORT = prof.device.replace('NVIDIA GeForce ', '');
const softmaxRows = prof.benchmarks.filter((r) => r.kernel.startsWith('softmax'));
const matmulRows = prof.benchmarks.filter((r) => r.kernel === 'matmul_torch');
const SHAPES = [...new Set(softmaxRows.map((r) => `${r.rows}x${r.cols}`))];
const pick = (shape, route) => softmaxRows.find((r) => `${r.rows}x${r.cols}` === shape && r.route.startsWith(route));
const NATIVE = pick('4096x1024', 'native'), UNFUSED = pick('4096x1024', 'unfused');
const RATIO = UNFUSED.median_ms / NATIVE.median_ms;
const UNFUSED_PROFILE = prof.profiler.find((p) => p.route.startsWith('unfused'));
const CHECKED = [...prof.benchmarks, ...bench.benchmarks].map((r) => ({ row: r, ...recompute(r) }));
const VERIFIED_DEEP = CHECKED.slice(0, prof.benchmarks.length).filter((c) => relDiff(c.recomputed, c.stored) <= 0.01 && c.row.correct).length;
const LEARN_PASS = learn.checks.filter((c) => c.pass).length;
const CPP_TESTS = build.cpp_tests_passed + deep.cpp_deep_tests_passed;
const sizeOf = (r) => (r.size ? `${r.size}` : r.rows ? `${r.rows} × ${r.cols}` : `n = ${fmt(r.n)}`);

const SECTIONS = [['top', 'Overview', '◎'], ['status', 'Status', '☰'], ['tiling', 'Tiling', '▦'], ['softmax', 'Softmax', 'eˣ'], ['profiler', 'Profiler', '▥'], ['fused', 'Fused model', '⇉'], ['variance', 'Variance', '±'], ['cpp', 'C++ track', '{}'], ['limits', 'Limits', '!'], ['roadmap', 'Roadmap', '❯_']];

function useReveal() {
  useEffect(() => {
    const els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) { els.forEach((e) => e.classList.add('in')); return; }
    const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.08 });
    els.forEach((e) => io.observe(e));
    return () => io.disconnect();
  }, []);
}

function MenuBar() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => { const t = setInterval(() => setNow(new Date()), 30000); return () => clearInterval(t); }, []);
  return (
    <div className="menubar" role="banner">
      <span className="apple" aria-hidden="true">◆</span>
      <a className="app" href="#top">CUDA Kernels Lab</a>
      <a className="m" href={blob('SPEC.md')}>Spec</a>
      <a className="m" href={blob('level-3-deep/README.md')}>Deep results</a>
      <a className="m" href={blob('level-3-deep/cuda/ANALYSIS.md')}>Kernel analysis</a>
      <span className="spacer" />
      <span className="status"><i /> {PY_TESTS} Python + {CPP_TESTS} C++ tests in CI</span>
      <a href={REPO}>GitHub ↗</a>
      <span className="clock">{now.toLocaleString('en-US', { weekday: 'short', hour: 'numeric', minute: '2-digit' })}</span>
    </div>
  );
}

function Dock() {
  const [active, setActive] = useState('top');
  useEffect(() => {
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setActive(e.target.id)), { rootMargin: '-45% 0px -50% 0px' });
    SECTIONS.forEach(([id]) => { const el = document.getElementById(id); if (el) io.observe(el); });
    return () => io.disconnect();
  }, []);
  return (
    <nav className="dock" aria-label="Sections">
      {SECTIONS.map(([id, label, icon]) => (
        <a key={id} href={`#${id}`} className={active === id ? 'on' : ''} aria-current={active === id ? 'true' : undefined} aria-label={label}>
          <span className="ico" aria-hidden="true">{icon}</span><span className="lbl">{label}</span>
        </a>
      ))}
    </nav>
  );
}

function Range({ label, value, min, max, step, onChange, show }) {
  const id = useId();
  return (
    <div className="field">
      <label htmlFor={id}>{label} <span>{show}</span></label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} />
    </div>
  );
}

function Seg({ label, value, options, onChange }) {
  return (
    <div className="seg" role="group" aria-label={label}>
      {options.map(([k, l]) => <button key={k} aria-pressed={value === k} onClick={() => onChange(k)}>{l}</button>)}
    </div>
  );
}

const Recorded = ({ children = `recorded run, ${DEVICE_SHORT}, ${RUN_DATE}` }) => <span className="pill info">● {children}</span>;

export default function App() {
  useReveal();
  return (
    <div className="wrap">
      <MenuBar />
      <div className="glow" aria-hidden="true" />
      <Hero />
      <Status />
      <Tiling />
      <Runner />
      <Softmax />
      <Profiler />
      <Stages />
      <Fused />
      <Variance />
      <Cpp />
      <Limits />
      <Roadmap />
      <Dock />
      <footer>
        Code: Apache-2.0. Every number on this page is read at build time from <code>level-*/results*.json</code> and the level write-ups, or computed in your browser from them.
        GPU numbers are recorded PyTorch runs on one {DEVICE_SHORT}; the lab's own CUDA kernels have not been compiled.
      </footer>
    </div>
  );
}

function Hero() {
  const ops = UNFUSED_PROFILE.aten_ops.length;
  return (
    <header className="card hero window" id="top" data-title="cuda-kernels-lab — README.md">
      <div className="head">
        <p className="eyebrow">GPU programming lab · three levels · recorded on {DEVICE_SHORT}</p>
        <span className="pill warn">● Custom kernels: written, not compiled</span>
      </div>
      <h1>Fusing softmax into <em>one kernel</em> ran {RATIO.toFixed(1)}× faster than {ops} separate ops.</h1>
      <p className="sub">
        On an {prof.device} (torch {prof.torch_version}), PyTorch's native fused softmax took {NATIVE.median_ms} ms at {NATIVE.rows} by {NATIVE.cols}; the same math as {ops} separate PyTorch ops took {UNFUSED.median_ms} ms.
        Both routes are PyTorch. The lab's own CUDA kernels (<a href={blob('level-2-build/cuda/kernels.cu')}>kernels.cu</a>, <a href={blob('level-3-deep/cuda/fused_softmax.cu')}>fused_softmax.cu</a>) are written but have never been compiled, because the GPU node has no CUDA toolkit.
        Below: the reasoning in calculators you can run, and the recorded runs you can explore.
      </p>
      <div className="actions">
        <a className="btn primary" href="#tiling">Open the tiling calculator</a>
        <a className="btn ghost" href="#status">See what ran and what did not</a>
      </div>
      <div className="tiles">
        <div className="tile"><div className="num">{NATIVE.median_ms}<small> ms</small></div><div className="cap">Native fused softmax, median, {NATIVE.rows}×{NATIVE.cols}</div></div>
        <div className="tile"><div className="num">{UNFUSED.median_ms}<small> ms</small></div><div className="cap">Unfused, {ops} PyTorch ops, same input</div></div>
        <div className="tile"><div className="num">{VERIFIED_DEEP}/{prof.benchmarks.length}</div><div className="cap">Deep rows whose throughput your browser recomputes within 1%</div></div>
        <div className="tile"><div className="num">{build.cpp_tests_passed}+{deep.cpp_deep_tests_passed}</div><div className="cap">C++ tests passing (Build + Deep), sanitizers {deep.asan_ubsan_deep_tool}</div></div>
      </div>
    </header>
  );
}

function Status() {
  const cupti = prof.profiler.some((p) => p.cuda_activity_available);
  const rows = [
    ['CPU reference kernels', 'level-1-learn/kernels.py', 'good', 'Ran, verified', `${LEARN_PASS} of ${learn.checks.length} correctness checks pass; load counts exact.`],
    ['PyTorch native and unfused routes', 'level-3-deep/profiler_bench.py', 'good', 'Measured', `${VERIFIED_DEEP} of ${prof.benchmarks.length} Deep rows and ${build.gpu_rows_verified} Build rows verified by recomputation.`],
    ['Custom Build kernels', 'level-2-build/cuda/kernels.cu', 'bad', 'Written, not compiled', `nvcc_on_path: ${String(prof.nvcc_on_path)} · cuda_home: ${JSON.stringify(prof.cuda_home)}`],
    ['Custom fused softmax', 'level-3-deep/cuda/fused_softmax.cu', 'bad', 'Written, not compiled', `custom_kernel_compiled: ${String(deep.custom_kernel_compiled)}. Its speed is a prediction only (see fused model).`],
    ['torch.compile route', 'compile_softmax', 'warn', 'Attempted, failed', `${prof.compile_softmax.error.split(':')[0]}: no working Triton on the node, so no compiled row.`],
    ['CUPTI kernel rows', 'torch.profiler', 'warn', cupti ? 'Available' : 'Unavailable', `cuda_activity_available: ${String(cupti)}. Profiler evidence is CPU-side op counts.`],
    ['C++ tool, tests, sanitizers', 'level-3-deep/cpp/deep_tool.cpp', 'good', 'Ran, verified', `${CPP_TESTS} tests pass; ASan + UBSan ${deep.asan_ubsan_deep_tool} on both tools.`],
  ];
  return (
    <section id="status" className="card window reveal" data-title="kernel_status.md">
      <p className="eyebrow">Status board</p>
      <h2>What ran, what was measured, and what is <em>only written</em>.</h2>
      <p className="sub">Read straight from the results files. Anything marked red has no number anywhere in this repo, and none is shown on this page.</p>
      <div style={{ overflowX: 'auto', marginTop: 16 }}>
        <table>
          <thead><tr><th>Component</th><th>Status</th><th>Evidence</th><th>Source</th></tr></thead>
          <tbody>
            {rows.map(([n, src, tone, label, ev]) => (
              <tr key={n}><td><b>{n}</b></td><td><span className={`pill ${tone}`}>{label}</span></td><td className="hint">{ev}</td><td className="mono">{src}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Tiling() {
  const [m, setM] = useState(18), [n, setN] = useState(14), [k, setK] = useState(16), [t, setT] = useState(4);
  const [step, setStep] = useState(0);
  const [play, setPlay] = useState(() => !(typeof matchMedia !== 'undefined' && matchMedia('(prefers-reduced-motion: reduce)').matches));
  const bm = Math.ceil(m / t), bn = Math.ceil(n / t), bk = Math.ceil(k / t), total = bm * bn * bk;
  useEffect(() => { setStep(0); }, [m, n, k, t]);
  useEffect(() => {
    if (!play) return undefined;
    const id = setInterval(() => setStep((s) => (s + 1) % total), 650);
    return () => clearInterval(id);
  }, [play, total]);
  const s = step % total, br = Math.floor(s / (bn * bk)), bc = Math.floor(s / bk) % bn, kt = s % bk;
  const naive = naiveLoads(m, n, k), padded = tiledSlots(m, n, k, t), inb = tiledInBounds(m, n, k, t);
  // geometry: A (M x K) left-bottom, B (K x N) right-top, C (M x N) right-bottom, padded extents dashed
  const Mp = bm * t, Np = bn * t, Kp = bk * t, gap = 18, W = 520, H = 440;
  const sc = Math.min((W - gap - 8) / (Kp + Np), (H - gap - 24) / (Kp + Mp));
  const ax = 4, ay = 4 + Kp * sc + gap, bx = ax + Kp * sc + gap, by = 4;
  const grid = (x0, y0, rows, cols, real) => (
    <g>
      <rect x={x0} y={y0} width={cols * sc} height={rows * sc} fill="none" stroke="var(--grid)" strokeDasharray="3 3" />
      <rect x={x0} y={y0} width={real[1] * sc} height={real[0] * sc} fill="var(--primary-pale)" stroke="var(--border)" />
      {Array.from({ length: rows / t - 1 }, (_, i) => <line key={`r${i}`} className="axis" x1={x0} x2={x0 + cols * sc} y1={y0 + (i + 1) * t * sc} y2={y0 + (i + 1) * t * sc} />)}
      {Array.from({ length: cols / t - 1 }, (_, i) => <line key={`c${i}`} className="axis" y1={y0} y2={y0 + rows * sc} x1={x0 + (i + 1) * t * sc} x2={x0 + (i + 1) * t * sc} />)}
    </g>
  );
  const hl = (x, y, fill) => <rect x={x} y={y} width={t * sc} height={t * sc} fill={fill} opacity="0.55" rx="2" style={{ transition: 'x .3s, y .3s' }} />;
  return (
    <section id="tiling" className="card window reveal" data-title="tiling.py — load counter (live)">
      <div className="head">
        <div>
          <p className="eyebrow">Level 1 · computed in your browser</p>
          <h2>Tiling leaves the arithmetic alone and <em>cuts the loads</em>.</h2>
          <p className="sub">
            A naive matmul thread reads one A and one B element per multiply, {`2·M·N·K`} loads with no reuse. A tiled block loads one tile of A and one of B into shared memory per k-step and every thread in the block reuses them.
            The counts use the same formulas as <a href={blob('level-1-learn/kernels.py')}>kernels.py</a>. The padded count includes zero slots at the edges, as kernels.py does; the in-bounds count drops them, closing the edge-padding caveat in the Learn write-up.
          </p>
        </div>
        <span className="pill good">Counts, not timings</span>
      </div>
      <div className="grid2">
        <div>
          <div className="controls" style={{ gridTemplateColumns: '1fr 1fr' }}>
            <Range label="M (rows of A)" value={m} min={1} max={64} step={1} onChange={setM} show={m} />
            <Range label="N (cols of B)" value={n} min={1} max={64} step={1} onChange={setN} show={n} />
            <Range label="K (inner)" value={k} min={1} max={64} step={1} onChange={setK} show={k} />
            <Range label="Tile size" value={t} min={1} max={16} step={1} onChange={setT} show={`${t}×${t}`} />
          </div>
          <div className="tiles" style={{ gridTemplateColumns: '1fr 1fr', marginTop: 0 }}>
            <div className="tile"><div className="num">{fmt(naive)}</div><div className="cap">Naive element loads</div></div>
            <div className="tile"><div className="num">{fmt(padded)}</div><div className="cap">Tiled slots, padded (kernels.py count) · {fx(naive / padded, 2)}× fewer</div></div>
            <div className="tile"><div className="num">{fmt(inb)}</div><div className="cap">Tiled loads, in bounds only · {fx(naive / inb, 2)}× fewer</div></div>
            <div className="tile"><div className="num">{fmt(multiplies(m, n, k))}</div><div className="cap">Multiplies, both forms ({fmt(adds(m, n, k))} adds)</div></div>
          </div>
          <p className="hint" style={{ marginTop: 12 }}>
            {padded === inb ? 'The tile divides every dimension, so padding costs nothing here.' : `Edge padding inflates the padded count by ${fx((padded / inb - 1) * 100, 1)}% at this shape.`}
            {' '}In-bounds rule: each A element is loaded once per block column, each B element once per block row.
          </p>
        </div>
        <div>
          <div className="head" style={{ marginBottom: 8 }}>
            <h3>Block ({br},{bc}), k-step {kt + 1} of {bk}</h3>
            <button className="btn ghost" onClick={() => setPlay((p) => !p)} aria-pressed={play}>{play ? 'Pause' : 'Play'}</button>
          </div>
          <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Tile walk: block row ${br}, block column ${bc}, k-step ${kt + 1} of ${bk}`}>
            {grid(ax, ay, Mp, Kp, [m, k])}
            {grid(bx, by, Kp, Np, [k, n])}
            {grid(bx, ay, Mp, Np, [m, n])}
            {hl(ax + kt * t * sc, ay + br * t * sc, 'var(--warn)')}
            {hl(bx + bc * t * sc, by + kt * t * sc, 'var(--good)')}
            {hl(bx + bc * t * sc, ay + br * t * sc, 'var(--primary)')}
            <text x={ax} y={H - 4}>A ({m}×{k})</text>
            <text x={bx + Np * sc} y={by + Kp * sc + 13} textAnchor="end">B ({k}×{n})</text>
            <text x={bx + Np * sc} y={H - 4} textAnchor="end">C ({m}×{n})</text>
          </svg>
          <div className="legend"><span><i style={{ background: 'var(--warn)' }} />A tile in shared memory</span><span><i style={{ background: 'var(--good)' }} />B tile</span><span><i style={{ background: 'var(--primary)' }} />output block</span><span>dashed = zero padding</span></div>
        </div>
      </div>
      <h3 style={{ marginTop: 22 }}>Known-answer check against <a href={blob('level-1-learn/results_learn.json')}>results_learn.json</a></h3>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Size</th><th>Tile</th><th>Naive, recorded</th><th>Naive, browser</th><th>Tiled, recorded</th><th>Tiled, browser</th><th>In-bounds, browser</th><th>Match</th></tr></thead>
          <tbody>
            {learn.load_table.map((r) => {
              const a = naiveLoads(r.size, r.size, r.size), b = tiledSlots(r.size, r.size, r.size, r.tile);
              const ok = a === r.naive_element_loads && b === r.tiled_tile_loads;
              return <tr key={r.size}><td>{r.size}×{r.size}</td><td>{r.tile}</td><td>{fmt(r.naive_element_loads)}</td><td>{fmt(a)}</td><td>{fmt(r.tiled_tile_loads)}</td><td>{fmt(b)}</td><td>{fmt(tiledInBounds(r.size, r.size, r.size, r.tile))}</td><td className={ok ? 'ok' : 'no'}>{ok ? 'exact' : 'differs'}</td></tr>;
            })}
          </tbody>
        </table>
      </div>
      <p className="hint">The same assertion runs in <code>npm run build</code>, so the site cannot ship if these formulas drift from the repo.</p>
    </section>
  );
}

const toText = (mat) => mat.map((r) => r.join(', ')).join('\n');
const parse = (txt) => txt.trim().split('\n').map((l) => l.split(/[,\s]+/).filter(Boolean).map(Number));

function Heat({ mat, label }) {
  const nr = mat.length, nc = mat[0].length, flat = mat.flat(), mx = Math.max(1, ...flat.map(Math.abs));
  const S = 12;
  return (
    <svg className="chart heat" viewBox={`0 0 ${nc * S} ${nr * S}`} role="img" aria-label={label}>
      {mat.map((row, i) => row.map((v, j) => (
        <rect key={`${i}-${j}`} x={j * S} y={i * S} width={S - 1} height={S - 1} rx="2" fill={v < 0 ? 'var(--bad)' : 'var(--primary)'} opacity={0.12 + 0.88 * Math.abs(v) / mx}><title>{`[${i},${j}] = ${v}`}</title></rect>
      )))}
    </svg>
  );
}

function Runner() {
  const [size, setSize] = useState(6), [tile, setTile] = useState(4);
  const [aTxt, setATxt] = useState(() => toText(detMatrix(6, 1))), [bTxt, setBTxt] = useState(() => toText(detMatrix(6, 2)));
  const reset = (n) => { setSize(n); setATxt(toText(detMatrix(n, 1))); setBTxt(toText(detMatrix(n, 2))); };
  const randomize = () => {
    const r = () => Array.from({ length: size }, () => Array.from({ length: size }, () => Math.floor(Math.random() * 19) - 9));
    setATxt(toText(r())); setBTxt(toText(r()));
  };
  const a = parse(aTxt), b = parse(bTxt);
  const valid = a.length === size && b.length === size && [...a, ...b].every((r) => r.length === size && r.every(Number.isFinite));
  const res = useMemo(() => {
    if (!valid) return null;
    const nv = matmulNaive(a, b), tl = matmulTiled(a, b, tile);
    const diff = Math.max(...nv.c.flatMap((row, i) => row.map((v, j) => Math.abs(v - tl.c[i][j]))));
    return { nv, tl, diff };
  }, [aTxt, bTxt, tile, valid]); // eslint-disable-line react-hooks/exhaustive-deps
  return (
    <section id="runner" className="card window reveal" data-title="matmul_runner.py — live">
      <div className="head">
        <div>
          <p className="eyebrow">Level 1 · runs in your browser</p>
          <h2>Run both matmuls yourself and <em>compare</em> the outputs.</h2>
          <p className="sub">A JavaScript port of <span className="mono">matmul_naive</span> and <span className="mono">matmul_tiled</span> from kernels.py, loop for loop. The default matrices are the deterministic ones run_learn.py uses. Edit any value, or randomize.</p>
        </div>
        {res && <span className={`pill ${res.diff === 0 ? 'good' : 'bad'}`}>{res.diff === 0 ? '● naive = tiled, exactly' : `● max diff ${res.diff}`}</span>}
      </div>
      <div className="controls">
        <Range label="Matrix size n" value={size} min={2} max={16} step={1} onChange={reset} show={`${size}×${size}`} />
        <Range label="Tile size" value={tile} min={1} max={8} step={1} onChange={setTile} show={tile} />
      </div>
      <div className="actions" style={{ marginTop: 0, marginBottom: 16 }}>
        <button className="btn primary" onClick={randomize}>Randomize matrices</button>
        <button className="btn ghost" onClick={() => reset(size)}>Use run_learn.py matrices</button>
      </div>
      <div className="grid3">
        <div className="field"><label htmlFor="matA">Matrix A</label><textarea id="matA" value={aTxt} onChange={(e) => setATxt(e.target.value)} rows={Math.min(size, 8)} spellCheck="false" /></div>
        <div className="field"><label htmlFor="matB">Matrix B</label><textarea id="matB" value={bTxt} onChange={(e) => setBTxt(e.target.value)} rows={Math.min(size, 8)} spellCheck="false" /></div>
        <div>
          <div className="field"><label>C = A·B (heat map, hover a cell)</label></div>
          {res ? <Heat mat={res.nv.c} label={`Heat map of the ${size} by ${size} product`} /> : <p className="hint">Each matrix needs {size} rows of {size} numbers.</p>}
        </div>
      </div>
      {res && (
        <div style={{ overflowX: 'auto', marginTop: 16 }}>
          <table>
            <thead><tr><th>Form</th><th>Multiplies</th><th>Adds</th><th>Element loads</th><th>In-bounds loads</th><th>Tiles loaded</th></tr></thead>
            <tbody>
              <tr><td>Naive</td><td>{fmt(res.nv.counts.multiplies)}</td><td>{fmt(res.nv.counts.adds)}</td><td>{fmt(res.nv.counts.element_loads)}</td><td>{fmt(res.nv.counts.element_loads)}</td><td>none</td></tr>
              <tr><td>Tiled, tile {tile}</td><td>{fmt(res.tl.counts.multiplies)}</td><td>{fmt(res.tl.counts.adds)}</td><td>{fmt(res.tl.counts.tile_loads)} (padded)</td><td>{fmt(res.tl.counts.in_bounds_loads)}</td><td>{fmt(res.tl.counts.tiles_loaded)}</td></tr>
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function Softmax() {
  const [txt, setTxt] = useState('1, 2, 3, 4.5');
  const [shift, setShift] = useState(0);
  const base = txt.split(/[,\s]+/).filter(Boolean).map(Number);
  const ok = base.length > 0 && base.length <= 12 && base.every(Number.isFinite);
  const x = ok ? base.map((v) => v + shift) : [];
  const nv = ok ? softmaxNaive(x) : [], st = ok ? softmaxStable(x) : [], st0 = ok ? softmaxStable(base) : [];
  const broken = nv.some((v) => !Number.isFinite(v));
  const lim64 = overflowAt(64), lim32 = overflowAt(32);
  const maxDev = ok ? Math.max(...st.map((v, i) => Math.abs(v - st0[i]))) : 0;
  const show = (v) => (Number.isFinite(v) ? v.toFixed(6) : String(v));
  return (
    <section id="softmax" className="card window reveal" data-title="softmax_overflow.py — live">
      <div className="head">
        <div>
          <p className="eyebrow">Numerics · float64 in your browser</p>
          <h2>Subtract the max first, or exp() <em>overflows</em>.</h2>
          <p className="sub">
            Naive softmax computes exp(x) directly. In float64 (what this page uses) exp overflows once x passes {fx(lim64, 2)}; in the float32 the C++ and CUDA kernels use, already past {fx(lim32, 1)}.
            The stable form subtracts the row max, so the largest term is exp(0) = 1. Shifting every input by the same constant leaves softmax unchanged, which is why the subtraction is free. Python's kernels.py raises OverflowError; JavaScript returns Infinity and then NaN.
          </p>
        </div>
        <span className={`pill ${broken ? 'bad' : 'good'}`}>● naive {broken ? 'overflowed' : 'finite'}</span>
      </div>
      <div className="controls">
        <div className="field"><label htmlFor="logits">Input vector (up to 12 values)</label><input id="logits" type="text" value={txt} onChange={(e) => setTxt(e.target.value)} /></div>
        <Range label="Add a constant to every value" value={shift} min={0} max={800} step={1} onChange={setShift} show={`+${shift}`} />
      </div>
      {ok ? (
        <>
          <div style={{ overflowX: 'auto' }}>
            <table>
              <thead><tr><th>i</th><th>x + c</th><th>Naive</th><th>Stable</th><th>Stable probability</th></tr></thead>
              <tbody>
                {x.map((v, i) => (
                  <tr key={i}><td>{i}</td><td>{fx(v, 2)}</td><td className={Number.isFinite(nv[i]) ? '' : 'no'}>{show(nv[i])}</td><td>{show(st[i])}</td>
                    <td style={{ minWidth: 120 }}><div className="track"><div className="fill" style={{ width: `${st[i] * 100}%`, background: 'var(--primary)' }} /></div></td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="hint">
            Largest input {fx(Math.max(...x), 2)} vs float64 limit {fx(lim64, 2)}. Stable sum = {st.reduce((a, b) => a + b, 0).toFixed(12)}.
            Shift invariance: the stable output differs from the unshifted one by at most {maxDev.toExponential(1)}.
          </p>
        </>
      ) : <p className="hint">Enter 1 to 12 numbers separated by commas.</p>}
    </section>
  );
}

function Bars({ rows, unit, label, max }) {
  const top = max ?? Math.max(...rows.map((r) => r.v)) * 1.08;
  return (
    <div className="bars chartbars" role="img" aria-label={label}>
      {rows.map((r) => (
        <div className="bar-row" key={r.k}>
          <div className="lab"><span>{r.k}</span><b>{r.label ?? `${r.v} ${unit}`}</b></div>
          <div className="track"><div className="fill" style={{ width: `${Math.max(0.5, (r.v / top) * 100)}%`, background: r.c || 'var(--primary)' }} /></div>
        </div>
      ))}
    </div>
  );
}

function Profiler() {
  const [view, setView] = useState(SHAPES[0]);
  const isMat = view === 'matmul';
  const nat = !isMat && pick(view, 'native'), unf = !isMat && pick(view, 'unfused');
  const passes = unfusedPasses();
  return (
    <section id="profiler" className="card window reveal" data-title="profiler_bench.json — recorded">
      <div className="head">
        <div>
          <p className="eyebrow">Level 3 Deep · replay of a recorded run</p>
          <h2>The fused op wins at every shape, and every row <em>recomputes</em>.</h2>
          <p className="sub">
            CUDA-event medians from <a href={blob('level-3-deep/results_profiler.json')}>results_profiler.json</a>. Your browser recomputes each throughput from size and median, the same check as <a href={blob('level-3-deep/analysis_deep.py')}>analysis_deep.py</a>: matmul as 2n³ FLOP, softmax as 2·rows·cols·4 bytes.
          </p>
        </div>
        <Recorded />
      </div>
      <Seg label="Choose a shape" value={view} onChange={setView} options={[...SHAPES.map((s) => [s, `Softmax ${s.replace('x', '×')}`]), ['matmul', 'Matmul']]} />
      <div style={{ marginTop: 16 }}>
        {isMat ? (
          <Bars label="Matmul GFLOP/s by size, native op" unit="GFLOP/s" rows={matmulRows.map((r) => ({ k: `matmul ${r.size}, native op`, v: r.gflops }))} />
        ) : (
          <>
            <Bars label={`Softmax median milliseconds at ${view}, native vs unfused`} unit="ms" rows={[
              { k: 'native fused op', v: nat.median_ms, c: 'var(--good)' },
              { k: `unfused, ${UNFUSED_OPS.length} ops`, v: unf.median_ms, c: 'var(--bad)' },
            ]} />
            <p className="hint">Unfused is {fx(unf.median_ms / nat.median_ms, 2)}× slower at this shape. Lower is better.</p>
            <div className="tile model">
              <div className="head" style={{ marginBottom: 6 }}><h3>Per-route traffic</h3><span className="pill warn">MODEL, not measured</span></div>
              <p className="hint" style={{ margin: 0 }}>
                The shared axis credits both routes with 2 passes over the matrix. Counting the unfused sequence op by op ({UNFUSED_OPS.map(([o, p]) => `${o} ${p}`).join(' + ')} = {passes} passes, row vectors ignored)
                gives it {fx(softmaxGBs(unf.rows, unf.cols, unf.median_ms, passes), 1)} GB/s of effective traffic, close to the fused op's {nat.gb_per_s} GB/s.
                Reading: the unfused route is not slow per byte; it moves about {passes / 2}× the bytes, which matches the {fx(unf.median_ms / nat.median_ms, 1)}× gap.
              </p>
            </div>
          </>
        )}
      </div>
      <h3 style={{ marginTop: 20 }}>All {prof.benchmarks.length} recorded rows, recomputed</h3>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Kernel</th><th>Route</th><th>Size</th><th>Median</th><th>Stored</th><th>Recomputed</th><th>Difference</th></tr></thead>
          <tbody>
            {CHECKED.slice(0, prof.benchmarks.length).map(({ row, stored, recomputed, unit }, i) => (
              <tr key={i}><td className="mono">{row.kernel}</td><td>{row.route}</td><td>{sizeOf(row)}</td><td>{row.median_ms} ms</td><td>{stored} {unit}</td><td>{fx(recomputed)}</td>
                <td className={relDiff(recomputed, stored) <= 0.01 ? 'ok' : 'no'}>{(relDiff(recomputed, stored) * 100).toFixed(3)}%</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Stages() {
  const st = prof.stage_profile_softmax_4096x1024;
  const color = (s) => (s.startsWith('native') ? 'var(--good)' : s.startsWith('unfused') ? 'var(--bad)' : 'var(--primary)');
  const inter = MiB(NATIVE.rows * NATIVE.cols * 4);
  return (
    <section id="stages" className="card window reveal" data-title="stage_profile.svg — recorded">
      <div className="head">
        <div>
          <p className="eyebrow">Where the unfused time goes</p>
          <h2>Two stages <em>dominate</em>: exp and divide.</h2>
          <p className="sub">
            Each unfused stage timed on its own at {NATIVE.rows}×{NATIVE.cols}, then the whole sequence, then the fused op. exp and div each write a full {inter} MiB float32 intermediate to global memory; the fused op writes none.
            Stage times do not add up to the sequence, because each includes its own launch and write.
          </p>
        </div>
        <Recorded />
      </div>
      <Bars label="Per-stage median milliseconds for unfused softmax at 4096 by 1024" unit="ms" rows={st.map((s) => ({ k: s.stage, v: s.median_ms, c: color(s.stage) }))} />
    </section>
  );
}

const PASS_OPTS = [[2, 'Native fused (1 read)'], [3, 'fused_softmax.cu (2 reads)'], [4, 'Build 3-pass kernel (3 reads)']];

function Fused() {
  const [rows, setRows] = useState(NATIVE.rows), [cols, setCols] = useState(NATIVE.cols);
  const [passes, setPasses] = useState(3), [bw, setBw] = useState(NATIVE.gb_per_s);
  const ms = predictMs(rows, cols, passes, bw);
  const rec = softmaxRows.filter((r) => r.rows === rows && r.cols === cols);
  const bars = [{ k: `PREDICTED, ${PASS_OPTS.find((p) => p[0] === passes)[1]}`, v: ms, label: `${ms.toFixed(4)} ms`, c: 'var(--warn)' },
    ...rec.map((r) => ({ k: `recorded, ${r.route}`, v: r.median_ms, label: `${r.median_ms} ms`, c: r.route.startsWith('native') ? 'var(--good)' : 'var(--bad)' }))];
  return (
    <section id="fused" className="card window reveal" data-title="fused_softmax.cu — traffic model">
      <div className="head">
        <div>
          <p className="eyebrow">The lab's own kernel · never run</p>
          <h2>A <em>prediction</em> for the kernel that has not compiled yet.</h2>
          <p className="sub">
            fused_softmax.cu caches exp(x − max) in shared memory, so it reads the input twice and writes once: 3 × rows × cols × 4 bytes. At the native op's measured {NATIVE.gb_per_s} GB/s,
            <a href={blob('level-3-deep/cuda/ANALYSIS.md')}> ANALYSIS.md</a> predicts about {predictMs(NATIVE.rows, NATIVE.cols, 3, NATIVE.gb_per_s).toFixed(3)} ms at {NATIVE.rows}×{NATIVE.cols}. This calculator reproduces that and lets you vary the inputs. It is arithmetic, not a measurement.
          </p>
        </div>
        <span className="pill warn">● PREDICTION, not a result</span>
      </div>
      <Seg label="Global passes per element" value={passes} onChange={setPasses} options={PASS_OPTS} />
      <div className="controls" style={{ marginTop: 16 }}>
        <Range label="Rows" value={rows} min={256} max={16384} step={256} onChange={setRows} show={fmt(rows)} />
        <Range label="Columns" value={cols} min={128} max={4096} step={128} onChange={setCols} show={fmt(cols)} />
        <Range label="Assumed bandwidth" value={bw} min={100} max={1200} step={0.01} onChange={setBw} show={`${fx(bw)} GB/s`} />
      </div>
      {passes === 3 && cols > 1024 && <p className="pill bad" role="alert" style={{ whiteSpace: 'normal' }}>Over 1024 columns the kernel's shared exp cache overflows; ANALYSIS.md says the harness must refuse this launch.</p>}
      <Bars label="Predicted median against recorded medians at the same shape" unit="ms" rows={bars} max={Math.max(...bars.map((b) => b.v), UNFUSED.median_ms) * 1.12} />
      <p className="hint">{rec.length ? `Green and red bars are recorded PyTorch medians at ${rows}×${cols}.` : `No recorded run at ${rows}×${cols}; recorded shapes are ${SHAPES.join(', ')}.`} Traffic {fx(passes * rows * cols * 4 / 2 ** 20, 1)} MiB per call.</p>
    </section>
  );
}

function Variance() {
  const key = (r) => `${r.kernel}|${r.size ?? `${r.rows}x${r.cols}`}`;
  const pairs = bench.benchmarks.map((b) => [b, prof.benchmarks.find((d) => key(d) === key(b) && d.route === 'native op')]).filter(([, d]) => d);
  return (
    <section id="variance" className="card window reveal" data-title="run_variance — same op, two sessions">
      <div className="head">
        <div>
          <p className="eyebrow">Honest noise</p>
          <h2>Same op, same laptop, two sessions, <em>different</em> numbers.</h2>
          <p className="sub">
            Build (<a href={blob('level-2-build/results_torch_bench.json')}>results_torch_bench.json</a>) and Deep (<a href={blob('level-3-deep/results_profiler.json')}>results_profiler.json</a>) both timed these native ops. Laptop GPU clocks move with heat and power state, and no clocks were logged.
            Read single-session throughputs with this spread in mind; ratios inside one session (fused vs unfused) are the stable finding.
          </p>
        </div>
        <Recorded>two recorded runs, {DEVICE_SHORT}</Recorded>
      </div>
      <div style={{ overflowX: 'auto' }}>
        <table>
          <thead><tr><th>Op</th><th>Size</th><th>Build run</th><th>Deep run</th><th>Deep vs Build</th></tr></thead>
          <tbody>
            {pairs.map(([b, d]) => {
              const unit = b.gflops ? 'GFLOP/s' : 'GB/s', vb = b.gflops ?? b.gb_per_s, vd = d.gflops ?? d.gb_per_s, ch = (vd / vb - 1) * 100;
              return <tr key={key(b)}><td className="mono">{b.kernel}</td><td>{sizeOf(b)}</td><td>{vb} {unit}</td><td>{vd} {unit}</td><td className={Math.abs(ch) > 10 ? 'no' : ''}>{ch > 0 ? '+' : ''}{fx(ch, 1)}%</td></tr>;
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Cpp() {
  const mb = deep.move_bench, at = deep.atomics, ho = deep.handoff;
  const expected = at.threads * at.per_thread;
  return (
    <section id="cpp" className="card window reveal" data-title="deep_tool.cpp — recorded">
      <div className="head">
        <div>
          <p className="eyebrow">C++ track · recorded run, x86 build VM, g++ -std=c++17</p>
          <h2>Moving a buffer beat copying it <em>{fx(mb.copy_ms / mb.move_ms, 1)}×</em>.</h2>
          <p className="sub">
            From <a href={blob('level-3-deep/results_deep.json')}>results_deep.json</a>. The C++ track ran on the x86 build VM (named "Muse VM" in the write-ups), not on the GPU node (BlackBox). Counts and totals are exact and asserted by tests; timings are one run.
          </p>
        </div>
        <Recorded>recorded run, x86 build VM, {RUN_DATE}</Recorded>
      </div>
      <h3>LargeBuffer, {fmt(mb.elements)} doubles, {mb.reps} reps</h3>
      <Bars label="Copy versus move wall time in milliseconds" unit="ms" rows={[
        { k: `copy (${mb.copy_count} copies)`, v: mb.copy_ms, c: 'var(--bad)' },
        { k: `move (${mb.move_count} moves)`, v: mb.move_ms, c: 'var(--good)' },
      ]} />
      <div className="grid2" style={{ marginTop: 16 }}>
        <div>
          <h3>Atomic counter, {at.threads} threads × {fmt(at.per_thread)}</h3>
          <table>
            <thead><tr><th>Ordering</th><th>Total</th><th>Wall time</th></tr></thead>
            <tbody>
              <tr><td>relaxed</td><td className={at.relaxed_total === expected ? 'ok' : 'no'}>{fmt(at.relaxed_total)}</td><td>{at.relaxed_ms} ms</td></tr>
              <tr><td>seq_cst</td><td className={at.seqcst_total === expected ? 'ok' : 'no'}>{fmt(at.seqcst_total)}</td><td>{at.seqcst_ms} ms</td></tr>
            </tbody>
          </table>
          <p className="hint">Both reach the expected {fmt(expected)}. No ordering winner is claimed: timings overlap across runs on x86.</p>
        </div>
        <div>
          <h3>Producer-consumer handoff</h3>
          <table>
            <thead><tr><th>Form</th><th>Delivered</th><th>Wall time</th></tr></thead>
            <tbody>
              <tr><td>acquire-release</td><td className="ok">{ho.ordered_successes}/{ho.ordered_trials}</td><td>{ho.ordered_ms} ms</td></tr>
              <tr><td>relaxed (control)</td><td>{ho.relaxed_successes}/{ho.relaxed_trials}</td><td>{ho.relaxed_ms} ms</td></tr>
            </tbody>
          </table>
          <p className="hint">x86's strong memory model masks the missing ordering, so the relaxed control passing here is not evidence it is correct. Only the ordered form has a C++ guarantee.</p>
        </div>
      </div>
      <div className="grid2" style={{ marginTop: 16 }}>
        <div className="tile">
          <div className="head" style={{ marginBottom: 6 }}><h3>Sanitizers</h3><span className="pill good">{deep.asan_ubsan_deep_tool}</span></div>
          <p className="hint" style={{ margin: 0 }}>Deep tool ASan + UBSan: {deep.asan_ubsan_deep_tool}. Build tool: {deep.asan_ubsan_build_tool}. ThreadSanitizer passed in a manual run only; it is not a CI gate.</p>
        </div>
        <div className="tile">
          <div className="head" style={{ marginBottom: 6 }}><h3>Debugging session finding</h3><span className="pill warn">fixed</span></div>
          <p className="hint" style={{ margin: 0 }}>AddressSanitizer caught a <span className="mono">{ASAN[1]}</span> write at <span className="mono">{ASAN[2]}</span> in a deliberately unguarded tiled matmul ({DEBUG_CASE[1]}×{DEBUG_CASE[1]} at tile {DEBUG_CASE[2]}: the edge tile runs past the matrix). Restoring the bounds guard fixed it. <a href={blob('level-2-build/DEBUG-SESSION.md')}>DEBUG-SESSION.md</a></p>
        </div>
      </div>
    </section>
  );
}

function Limits() {
  const items = [
    ['No custom kernel has run', `Every GPU number compares PyTorch with PyTorch. The ${RATIO.toFixed(1)}× is the native fused op against unfused PyTorch ops, not the lab's own kernel against PyTorch.`],
    ['One device, one session per number', `All GPU rows are one ${DEVICE_SHORT}. Each is a single median; dispersion and raw samples were not stored. Session-to-session drift is shown above.`],
    ['No hardware roofline', 'Achieved GB/s and GFLOP/s are never compared with the device peak, so efficiency is unknown. The repo records no peak figure, so this page shows none.'],
    ['Profiler evidence is shallow', 'CUPTI failed, so there are no kernel names or GPU-side times; torch.compile failed, so the compiled arm is missing.'],
    ['Tiling evidence is counts', 'Level 1 counts loads exactly but measures no shared-memory speedup on a GPU.'],
    ['Ordering argued, not observed', 'Atomics ran on x86 only. A weakly ordered machine (ARM) is needed to observe the relaxed-ordering hazard.'],
    ['C++ timings are one run', 'Copy, move, atomics and handoff wall times come from a single recorded run; counts and totals are exact, the times are not repeated.'],
    ['Models are labelled models', 'The per-route traffic count and the fused-kernel prediction are arithmetic on recorded medians. Neither is a measurement.'],
  ];
  return (
    <section id="limits" className="card window reveal" data-title="LIMITATIONS.md">
      <p className="eyebrow">Threats to validity</p>
      <h2>What this lab <em>does not</em> show yet.</h2>
      <div className="grid2" style={{ marginTop: 16 }}>
        {items.map(([t, d]) => (
          <div className="tile" key={t}><h3>{t}</h3><p className="hint" style={{ margin: '6px 0 0' }}>{d}</p></div>
        ))}
      </div>
      <p className="hint" style={{ marginTop: 14 }}>Full grading against a research rubric: <a href={blob('docs/research-review.md')}>docs/research-review.md</a>.</p>
    </section>
  );
}

function Roadmap() {
  const docs = [
    ['SPEC.md', 'Spec', 'Scope and acceptance criteria'],
    ['level-1-learn/README.md', 'Level 1 Learn', 'Kernels and load counts'],
    ['level-2-build/README.md', 'Level 2 Build', 'C++ tests, GPU benchmarks'],
    ['level-3-deep/README.md', 'Level 3 Deep', 'Profiler, atomics, move'],
    ['level-3-deep/cuda/ANALYSIS.md', 'Kernel analysis', 'Blocker and prediction'],
    ['docs/research-review.md', 'Research review', 'Rubric grades and gaps'],
  ];
  return (
    <section id="roadmap" className="card window reveal" data-title="ROADMAP.md — v1.1 backlog">
      <p className="eyebrow">Next</p>
      <h2>First job: <em>compile</em> the kernels.</h2>
      <div className="grid2" style={{ marginTop: 16 }}>
        <ol className="checklist">
          {ROADMAP.map(([n, t]) => <li key={n}><span className="box" aria-hidden="true" /><span><b>{n}.</b> {t}</span></li>)}
        </ol>
        <div>
          <h3>Grades on record (of 10,000)</h3>
          <div className="tiles" style={{ gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', marginTop: 10 }}>
            {GRADES.map(([l, g]) => <div className="tile" key={l}><div className="num">{g}</div><div className="cap">{l}</div></div>)}
          </div>
          <h3 style={{ marginTop: 18 }}>Reproduce the CPU side</h3>
          <ol className="steps" style={{ marginTop: 10 }}>
            <li><span className="n">1</span><div><code>python3 level-1-learn/run_learn.py</code></div></li>
            <li><span className="n">2</span><div><code>python3 level-3-deep/analysis_deep.py</code></div></li>
            <li><span className="n">3</span><div><code>cd site && npm ci && npm test</code></div></li>
          </ol>
        </div>
      </div>
      <div className="docs" style={{ marginTop: 20 }}>
        {docs.map(([p, t, d]) => <a key={p} href={blob(p)}><b>{t}</b><small>{d}</small></a>)}
      </div>
    </section>
  );
}
