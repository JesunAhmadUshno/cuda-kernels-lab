"""CPU reference kernels for cuda-kernels-lab Level 1 Learn.

Stdlib only. These functions mirror the structure of the CUDA kernels
they teach: vector add is the one thread per element pattern, the
tiled matmul uses the same block, tile, and shared memory indexing a
CUDA tiled kernel uses, and softmax shows why the stable form matters.

Correctness is the claim, not speed. No GPU ran this code. Operation
and load counts are exact integers computed from the loop structure,
not measured time.
"""
import math


def vector_add(a, b):
    """Elementwise add. Returns (result, counts).

    CUDA mapping: thread i computes c[i] = a[i] + b[i].
    counts: adds = len(a), element loads = 2 * len(a).
    """
    if len(a) != len(b):
        raise ValueError("vector_add needs equal lengths")
    result = [x + y for x, y in zip(a, b)]
    return result, {"adds": len(a), "element_loads": 2 * len(a)}


def _shape(m):
    rows = len(m)
    cols = len(m[0]) if rows else 0
    for row in m:
        if len(row) != cols:
            raise ValueError("ragged matrix")
    return rows, cols


def matmul_naive(a, b):
    """Naive matmul. Returns (result, counts).

    CUDA mapping: one thread per output element, each thread walks a
    full row of a and a full column of b (global memory every step).
    counts: multiplies = M * N * K, adds = M * N * (K - 1) for a
    non-empty inner dimension (the first product needs no add),
    element loads = 2 * M * N * K (one a element and one b element
    per multiply, no reuse).
    """
    m, k = _shape(a)
    k2, n = _shape(b)
    if k != k2:
        raise ValueError("inner dimensions must match")
    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            total = 0.0
            for p in range(k):
                total += a[i][p] * b[p][j]
            result[i][j] = total
    adds = m * n * (k - 1) if k > 0 else 0
    counts = {"multiplies": m * n * k, "adds": adds,
              "element_loads": 2 * m * n * k}
    return result, counts


def matmul_tiled(a, b, tile):
    """Tiled matmul with the CUDA shared memory indexing pattern.

    The output is split into tile by tile blocks. For each block and
    each step along the inner dimension, a tile of a and a tile of b
    are loaded once (the shared memory load in CUDA) and every output
    in the block reuses them. Tile loads are counted as element slots
    loaded, including zero padded slots at the edges, which matches a
    guarded cooperative load of a full tile by tile tile threads.

    Returns (result, counts) with counts holding multiplies and adds
    (identical arithmetic to naive), tile_loads, tiles_loaded, and
    element_loads set to tile_loads for comparison with naive.
    """
    if tile <= 0:
        raise ValueError("tile size must be positive")
    m, k = _shape(a)
    k2, n = _shape(b)
    if k != k2:
        raise ValueError("inner dimensions must match")
    result = [[0.0] * n for _ in range(m)]
    blocks_row = (m + tile - 1) // tile
    blocks_col = (n + tile - 1) // tile
    k_tiles = (k + tile - 1) // tile
    tiles_loaded = 0
    tile_loads = 0
    for br in range(blocks_row):
        for bc in range(blocks_col):
            acc = [[0.0] * tile for _ in range(tile)]
            for t in range(k_tiles):
                # Cooperative tile loads (shared memory in CUDA).
                a_tile = [[0.0] * tile for _ in range(tile)]
                b_tile = [[0.0] * tile for _ in range(tile)]
                for i in range(tile):
                    for j in range(tile):
                        gi = br * tile + i
                        gj = t * tile + j
                        if gi < m and gj < k:
                            a_tile[i][j] = a[gi][gj]
                        gj2 = bc * tile + j
                        gi2 = t * tile + i
                        if gi2 < k and gj2 < n:
                            b_tile[i][j] = b[gi2][gj2]
                tiles_loaded += 2
                tile_loads += 2 * tile * tile
                for i in range(tile):
                    for j in range(tile):
                        total = acc[i][j]
                        for p in range(tile):
                            total += a_tile[i][p] * b_tile[p][j]
                        acc[i][j] = total
            for i in range(tile):
                for j in range(tile):
                    gi = br * tile + i
                    gj = bc * tile + j
                    if gi < m and gj < n:
                        result[gi][gj] = acc[i][j]
    adds = m * n * (k - 1) if k > 0 else 0
    counts = {"multiplies": m * n * k, "adds": adds,
              "tile_loads": tile_loads, "tiles_loaded": tiles_loaded,
              "element_loads": tile_loads}
    return result, counts


def softmax_naive(x):
    """Softmax without max subtraction. Raises OverflowError on
    large inputs (for example 1000.0), which is the failure mode the
    stable form fixes. Returns (result, counts)."""
    exps = [math.exp(v) for v in x]
    total = sum(exps)
    return [e / total for e in exps], {"exps": len(x)}


def softmax_stable(x):
    """Softmax with max subtraction. The maximum element maps to
    exp(0) = 1, so no intermediate can overflow. Subtracting a
    constant from every input leaves softmax unchanged.
    Returns (result, counts)."""
    peak = max(x)
    exps = [math.exp(v - peak) for v in x]
    total = sum(exps)
    return [e / total for e in exps], {"exps": len(x), "max_subtracted": peak}
