// cuda-kernels-lab, Level 2 Build: real CUDA kernels on a real GPU.
//
// Kernels: vector_add, matmul_naive, matmul_tiled (shared memory),
// softmax_rows (numerically stable, one block per row).
// The harness checks every kernel against a CPU reference, times each
// kernel with CUDA events (median of repeated runs), and runs a
// canary-buffer debugging demonstration: a vector_add variant without
// its bounds guard writes into a canary region placed inside the same
// allocation, the canary check catches it, and the guarded kernel
// leaves the canary intact. See ../DEBUG-SESSION.md.
//
// Build on the fleet GPU node (BlackBox, Windows): build_blackbox.bat
// Output: results_cuda.json in the current directory.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>

#define CHECK_CUDA(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(err_)); \
        std::exit(1); \
    } \
} while (0)

// ---------------------------------------------------------------- kernels

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

// Deliberately buggy variant for the debugging session: no bounds guard.
// Launched with one extra block, threads past n write into whatever
// follows the output region (the canary, in this harness).
__global__ void vector_add_kernel_noguard(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i % n] + b[i % n];
}

__global__ void matmul_naive_kernel(const float* a, const float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) sum += a[row * n + k] * b[k * n + col];
        c[row * n + col] = sum;
    }
}

#define TILE 16

__global__ void matmul_tiled_kernel(const float* a, const float* b, float* c, int n) {
    __shared__ float as[TILE][TILE];
    __shared__ float bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (n + TILE - 1) / TILE; ++t) {
        int acol = t * TILE + threadIdx.x;
        int brow = t * TILE + threadIdx.y;
        as[threadIdx.y][threadIdx.x] = (row < n && acol < n) ? a[row * n + acol] : 0.0f;
        bs[threadIdx.y][threadIdx.x] = (brow < n && col < n) ? b[brow * n + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; ++k) sum += as[threadIdx.y][k] * bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) c[row * n + col] = sum;
}

// One block per row, block-wide reductions via shared memory.
// Assumes cols <= 1024 and blockDim.x == 256 threads cooperating in
// strided loops; the harness only calls it with cols <= 1024.
__global__ void softmax_rows_kernel(const float* x, float* y, int rows, int cols) {
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* xr = x + row * cols;
    float* yr = y + row * cols;
    __shared__ float red[256];
    float local_max = -1.0e30f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) local_max = fmaxf(local_max, xr[j]);
    red[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = red[0];
    float local_sum = 0.0f;
    for (int j = threadIdx.x; j < cols; j += blockDim.x) local_sum += expf(xr[j] - row_max);
    red[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = red[0];
    for (int j = threadIdx.x; j < cols; j += blockDim.x) yr[j] = expf(xr[j] - row_max) / row_sum;
}

// ---------------------------------------------------------------- helpers

static float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

template <typename Launch>
static float time_kernel(Launch launch, int warmup, int reps) {
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    for (int i = 0; i < warmup; ++i) launch();
    CHECK_CUDA(cudaDeviceSynchronize());
    std::vector<float> times;
    for (int i = 0; i < reps; ++i) {
        CHECK_CUDA(cudaEventRecord(start));
        launch();
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        times.push_back(ms);
    }
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    return median(times);
}

static void fill_pattern(float* p, int n, float scale, float offset) {
    for (int i = 0; i < n; ++i) p[i] = std::sin(i * 0.001f) * scale + offset;
}

static double max_abs_diff(const float* a, const float* b, int n) {
    double m = 0.0;
    for (int i = 0; i < n; ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

// ---------------------------------------------------------------- main

int main() {
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    std::printf("device: %s, compute capability %d.%d, %lld MiB\n", prop.name,
                prop.major, prop.minor,
                (long long)(prop.totalGlobalMem / (1024 * 1024)));

    std::vector<std::string> check_json;
    std::vector<std::string> bench_json;
    char buf[512];
    auto record_check = [&](const char* name, bool pass, double detail) {
        std::snprintf(buf, sizeof(buf),
                      "{\"name\": \"%s\", \"pass\": %s, \"max_abs_diff\": %.6g}",
                      name, pass ? "true" : "false", detail);
        check_json.push_back(buf);
        std::printf("check %-28s %s (max_abs_diff %.3g)\n", name, pass ? "PASS" : "FAIL", detail);
        if (!pass) { std::fprintf(stderr, "check failed: %s\n", name); std::exit(1); }
    };

    // ---- vector_add correctness + bench
    const int VN = 1 << 22;
    float *ha = new float[VN], *hb = new float[VN], *hc = new float[VN], *href = new float[VN];
    fill_pattern(ha, VN, 2.0f, 1.0f);
    fill_pattern(hb, VN, 3.0f, -0.5f);
    for (int i = 0; i < VN; ++i) href[i] = ha[i] + hb[i];
    float *da, *db, *dc;
    CHECK_CUDA(cudaMalloc(&da, VN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&db, VN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dc, VN * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(da, ha, VN * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(db, hb, VN * sizeof(float), cudaMemcpyHostToDevice));
    auto launch_vadd = [&]() {
        vector_add_kernel<<<(VN + 255) / 256, 256>>>(da, db, dc, VN);
    };
    launch_vadd();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hc, dc, VN * sizeof(float), cudaMemcpyDeviceToHost));
    record_check("vector_add", max_abs_diff(hc, href, VN) < 1e-4, max_abs_diff(hc, href, VN));
    float vadd_ms = time_kernel(launch_vadd, 3, 20);
    double vadd_gbs = 3.0 * VN * sizeof(float) / (vadd_ms * 1e-3) / 1e9;
    std::printf("bench vector_add n=%d median %.4f ms, %.1f GB/s\n", VN, vadd_ms, vadd_gbs);

    // ---- canary debugging demonstration (same allocation, guard on/off)
    const float CANARY = 12345.678f;
    const int CN = 100000;  // not a multiple of 256: the guard matters
    float *ca, *cb, *cc;
    CHECK_CUDA(cudaMalloc(&ca, CN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&cb, CN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&cc, (CN + 1024) * sizeof(float)));
    std::vector<float> ones(CN, 1.0f), canary_init(1024, CANARY), canary_out(1024);
    CHECK_CUDA(cudaMemcpy(ca, ones.data(), CN * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(cb, ones.data(), CN * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(cc + CN, canary_init.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice));
    int buggy_blocks = (CN + 255) / 256 + 1;  // one extra block on purpose
    vector_add_kernel_noguard<<<buggy_blocks, 256>>>(ca, cb, cc, CN);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(canary_out.data(), cc + CN, 1024 * sizeof(float), cudaMemcpyDeviceToHost));
    bool buggy_intact = true;
    for (float v : canary_out) if (v != CANARY) { buggy_intact = false; break; }
    std::printf("debug canary: unguarded kernel canary intact = %s (expected false)\n",
                buggy_intact ? "true" : "false");
    CHECK_CUDA(cudaMemcpy(cc + CN, canary_init.data(), 1024 * sizeof(float), cudaMemcpyHostToDevice));
    vector_add_kernel<<<(CN + 255) / 256, 256>>>(ca, cb, cc, CN);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(canary_out.data(), cc + CN, 1024 * sizeof(float), cudaMemcpyDeviceToHost));
    bool fixed_intact = true;
    for (float v : canary_out) if (v != CANARY) { fixed_intact = false; break; }
    std::printf("debug canary: guarded kernel canary intact = %s (expected true)\n",
                fixed_intact ? "true" : "false");
    if (buggy_intact || !fixed_intact) { std::fprintf(stderr, "canary demo failed\n"); return 1; }
    CHECK_CUDA(cudaFree(ca)); CHECK_CUDA(cudaFree(cb)); CHECK_CUDA(cudaFree(cc));

    // ---- matmul correctness + bench (two sizes)
    std::snprintf(buf, sizeof(buf),
                  "{\"kernel\": \"vector_add\", \"n\": %d, \"median_ms\": %.4f, \"gb_per_s\": %.2f}",
                  VN, vadd_ms, vadd_gbs);
    bench_json.push_back(buf);
    for (int n : {512, 1024}) {
        int N = n * n;
        float *ma = new float[N], *mb = new float[N], *mc = new float[N], *mref = new float[N];
        fill_pattern(ma, N, 0.5f, 0.25f);
        fill_pattern(mb, N, 0.5f, -0.125f);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j) {
                float s = 0.0f;
                for (int k = 0; k < n; ++k) s += ma[i * n + k] * mb[k * n + j];
                mref[i * n + j] = s;
            }
        float *mda, *mdb, *mdc;
        CHECK_CUDA(cudaMalloc(&mda, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mdb, N * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&mdc, N * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(mda, ma, N * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(mdb, mb, N * sizeof(float), cudaMemcpyHostToDevice));
        dim3 block(TILE, TILE), grid((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        auto launch_naive = [&]() { matmul_naive_kernel<<<grid, block>>>(mda, mdb, mdc, n); };
        auto launch_tiled = [&]() { matmul_tiled_kernel<<<grid, block>>>(mda, mdb, mdc, n); };
        launch_naive();
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(mc, mdc, N * sizeof(float), cudaMemcpyDeviceToHost));
        record_check(n == 512 ? "matmul_naive_512" : "matmul_naive_1024",
                     max_abs_diff(mc, mref, N) < 0.05, max_abs_diff(mc, mref, N));
        float naive_ms = time_kernel(launch_naive, 3, 10);
        launch_tiled();
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaMemcpy(mc, mdc, N * sizeof(float), cudaMemcpyDeviceToHost));
        record_check(n == 512 ? "matmul_tiled_512" : "matmul_tiled_1024",
                     max_abs_diff(mc, mref, N) < 0.05, max_abs_diff(mc, mref, N));
        float tiled_ms = time_kernel(launch_tiled, 3, 10);
        double flops = 2.0 * n * n * n;
        std::snprintf(buf, sizeof(buf),
                      "{\"kernel\": \"matmul_naive\", \"size\": %d, \"median_ms\": %.4f, \"gflops\": %.2f}",
                      n, naive_ms, flops / (naive_ms * 1e-3) / 1e9);
        bench_json.push_back(buf);
        std::snprintf(buf, sizeof(buf),
                      "{\"kernel\": \"matmul_tiled16\", \"size\": %d, \"median_ms\": %.4f, \"gflops\": %.2f}",
                      n, tiled_ms, flops / (tiled_ms * 1e-3) / 1e9);
        bench_json.push_back(buf);
        std::printf("bench matmul n=%d naive %.4f ms (%.1f GFLOP/s), tiled %.4f ms (%.1f GFLOP/s)\n",
                    n, naive_ms, flops / (naive_ms * 1e-3) / 1e9,
                    tiled_ms, flops / (tiled_ms * 1e-3) / 1e9);
        CHECK_CUDA(cudaFree(mda)); CHECK_CUDA(cudaFree(mdb)); CHECK_CUDA(cudaFree(mdc));
        delete[] ma; delete[] mb; delete[] mc; delete[] mref;
    }

    // ---- softmax correctness + bench
    const int ROWS = 4096, COLS = 1024;
    int SN = ROWS * COLS;
    float *sx = new float[SN], *sy = new float[SN], *sref = new float[SN];
    fill_pattern(sx, SN, 4.0f, 0.0f);
    for (int r = 0; r < ROWS; ++r) {
        float m = -1.0e30f;
        for (int j = 0; j < COLS; ++j) m = std::max(m, sx[r * COLS + j]);
        double s = 0.0;
        for (int j = 0; j < COLS; ++j) s += std::exp(sx[r * COLS + j] - m);
        for (int j = 0; j < COLS; ++j) sref[r * COLS + j] = (float)(std::exp(sx[r * COLS + j] - m) / s);
    }
    float *sdx, *sdy;
    CHECK_CUDA(cudaMalloc(&sdx, SN * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&sdy, SN * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(sdx, sx, SN * sizeof(float), cudaMemcpyHostToDevice));
    auto launch_softmax = [&]() { softmax_rows_kernel<<<ROWS, 256>>>(sdx, sdy, ROWS, COLS); };
    launch_softmax();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(sy, sdy, SN * sizeof(float), cudaMemcpyDeviceToHost));
    record_check("softmax_rows", max_abs_diff(sy, sref, SN) < 1e-5, max_abs_diff(sy, sref, SN));
    float sm_ms = time_kernel(launch_softmax, 3, 20);
    double sm_gbs = 2.0 * SN * sizeof(float) / (sm_ms * 1e-3) / 1e9;
    std::snprintf(buf, sizeof(buf),
                  "{\"kernel\": \"softmax_rows\", \"rows\": %d, \"cols\": %d, \"median_ms\": %.4f, \"gb_per_s\": %.2f}",
                  ROWS, COLS, sm_ms, sm_gbs);
    bench_json.push_back(buf);
    std::printf("bench softmax rows=%d cols=%d median %.4f ms, %.1f GB/s\n", ROWS, COLS, sm_ms, sm_gbs);
    CHECK_CUDA(cudaFree(sdx)); CHECK_CUDA(cudaFree(sdy));
    delete[] sx; delete[] sy; delete[] sref;

    FILE* out = std::fopen("results_cuda.json", "w");
    if (!out) { std::fprintf(stderr, "cannot open results_cuda.json\n"); return 1; }
    std::fprintf(out, "{\n  \"device\": \"%s\",\n  \"compute_capability\": \"%d.%d\",\n  \"checks\": [\n",
                 prop.name, prop.major, prop.minor);
    for (size_t i = 0; i < check_json.size(); ++i)
        std::fprintf(out, "    %s%s\n", check_json[i].c_str(), i + 1 < check_json.size() ? "," : "");
    std::fprintf(out, "  ],\n  \"benchmarks\": [\n");
    for (size_t i = 0; i < bench_json.size(); ++i)
        std::fprintf(out, "    %s%s\n", bench_json[i].c_str(), i + 1 < bench_json.size() ? "," : "");
    std::fprintf(out, "  ],\n  \"debug_canary\": {\"unguarded_canary_intact\": %s, \"guarded_canary_intact\": %s}\n}\n",
                 buggy_intact ? "true" : "false", fixed_intact ? "true" : "false");
    std::fclose(out);
    delete[] ha; delete[] hb; delete[] hc; delete[] href;
    CHECK_CUDA(cudaFree(da)); CHECK_CUDA(cudaFree(db)); CHECK_CUDA(cudaFree(dc));
    std::printf("ALL CHECKS PASS, results_cuda.json written\n");
    return 0;
}
