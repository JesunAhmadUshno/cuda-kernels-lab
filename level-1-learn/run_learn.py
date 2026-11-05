"""Level 1 Learn runner for cuda-kernels-lab.

Runs correctness checks against a plain independent reference on
labelled small cases, prints a tiled vs naive load count table at
three matrix sizes, and writes results_learn.json deterministically.

The JSON holds counts only. No wall time is claimed anywhere: this is
a CPU reference run and correctness is the claim, not speed.
Run twice and the JSON is byte identical.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernels import matmul_naive, matmul_tiled, softmax_naive, softmax_stable, vector_add

HERE = os.path.dirname(os.path.abspath(__file__))


def plain_matmul(a, b):
    """Independent plain reference, written separately from kernels."""
    return [[sum(a[i][p] * b[p][j] for p in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]


def det_matrix(n, seed):
    """Deterministic small integer matrix, no randomness."""
    return [[((i * n + j + seed) % 7) - 3 for j in range(n)]
            for i in range(n)]


def main():
    checks = []

    got, counts = vector_add([1, 2, 3], [10, 20, 30])
    ok = got == [11, 22, 33]
    checks.append({"case": "vector_add_small", "pass": ok,
                   "adds": counts["adds"]})
    print("vector_add_small:", "PASS" if ok else "FAIL", counts)

    a2 = [[1, 2], [3, 4]]
    b2 = [[5, 6], [7, 8]]
    want2 = [[19, 22], [43, 50]]
    naive2, _ = matmul_naive(a2, b2)
    tiled2, _ = matmul_tiled(a2, b2, 2)
    ok = naive2 == want2 and tiled2 == want2
    checks.append({"case": "matmul_2x2_hand", "pass": ok})
    print("matmul_2x2_hand:", "PASS" if ok else "FAIL")

    a3 = det_matrix(3, 1)
    b3 = det_matrix(3, 2)
    want3 = plain_matmul(a3, b3)
    naive3, _ = matmul_naive(a3, b3)
    tiled3, _ = matmul_tiled(a3, b3, 2)
    ok = naive3 == want3 and tiled3 == want3
    checks.append({"case": "matmul_3x3_tiled_tile2", "pass": ok})
    print("matmul_3x3_tiled_tile2:", "PASS" if ok else "FAIL")

    probs, _ = softmax_stable([1000.0, 1001.0, 999.0])
    ok = abs(sum(probs) - 1.0) < 1e-12 and probs[1] > probs[0] > probs[2]
    checks.append({"case": "softmax_stable_large", "pass": ok})
    print("softmax_stable_large:", "PASS" if ok else "FAIL",
          [round(p, 6) for p in probs])
    try:
        softmax_naive([1000.0, 1001.0])
        naive_overflow = False
    except OverflowError:
        naive_overflow = True
    checks.append({"case": "softmax_naive_overflows_large",
                   "pass": naive_overflow})
    print("softmax_naive_overflows_large:",
          "PASS" if naive_overflow else "FAIL")

    table = []
    for n, tile in [(4, 2), (8, 2), (16, 4)]:
        a = det_matrix(n, 1)
        b = det_matrix(n, 2)
        _, cn = matmul_naive(a, b)
        _, ct = matmul_tiled(a, b, tile)
        same, _ = matmul_tiled(a, b, tile)
        ref, _ = matmul_naive(a, b)
        row = {"size": n, "tile": tile,
               "naive_element_loads": cn["element_loads"],
               "tiled_tile_loads": ct["tile_loads"],
               "multiplies": cn["multiplies"], "adds": cn["adds"],
               "results_identical": same == ref}
        table.append(row)
        print(f"n={n} tile={tile} naive_loads={cn['element_loads']} "
              f"tiled_loads={ct['tile_loads']} "
              f"multiplies={cn['multiplies']} adds={cn['adds']} "
              f"identical={same == ref}")

    all_pass = all(c["pass"] for c in checks) and \
        all(r["results_identical"] for r in table)
    results = {
        "level": "1-learn",
        "project": "P04 cuda-kernels-lab",
        "device": "CPU reference only, no CUDA GPU ran",
        "checks": checks,
        "load_table": table,
        "all_pass": all_pass,
    }
    out = os.path.join(HERE, "results_learn.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print("all_pass:", all_pass)
    print("wrote", out)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
