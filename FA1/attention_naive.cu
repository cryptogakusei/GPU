#include <cstdio>
#include <cstdlib>
#include <cmath>


// naive attention kernel baseline
// 1. S = (1/sqrt(d)) * Q @ K^T --> use my GEMM warptiling kernel
// 2. P = softmax(S) --> use my softmax kernel with shuffle
// 3. O = P @ V   --> use my GEMM warptiling kernel


#define BM 128
#define BN 64
#define BK 8
#define WM 64
#define WN 32
#define WNITER 2
#define WMITER 2
#define TM 4
#define TN 4
#define WARPSIZE 32

#define BLOCKSIZE 256
#define N_MAX 2048

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

static_assert((WM * WN) % (WNITER * WARPSIZE * TM * TN) == 0, "bad warp config");
static_assert(WMITER == (WM * WN) / (WNITER * WARPSIZE * TM * TN), "WMITER mismatch");

// ---- Kernel 1 & 3: warptiling GEMM  C = alpha*(A@B) + beta*C ----
__global__ void gemm_warptiling(const float *A, const float *B, float *C,
                                 float beta, float alpha,
                                 unsigned int M, unsigned int N, unsigned int K) {
    const int WSUBM = WM / WMITER;
    const int WSUBN = WN / WNITER;

    uint warpIdx = threadIdx.x / WARPSIZE;
    uint warpRow = warpIdx / (BN / WN);
    uint warpCol = warpIdx % (BN / WN);
    uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);
    uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};

    const int numThreadsPerBlock = (BM * BN) / (TM * TN * WMITER * WNITER);
    float outerproduct[WMITER * WNITER * TM * TN] = {0.0};

    for (int p = 0; p < K / BK; p++) {
        for (int k = 4 * threadIdx.x; k < BM * BK; k += 4 * numThreadsPerBlock) {
            unsigned int Arow = blockIdx.x * BM + (k / BK);
            unsigned int tilerow = k / BK;
            unsigned int tilecol = k % BK;
            float4 tmp = reinterpret_cast<const float4*>(&A[Arow * K + (p * BK + tilecol)])[0];
            As[(tilecol + 0) * BM + tilerow] = tmp.x;
            As[(tilecol + 1) * BM + tilerow] = tmp.y;
            As[(tilecol + 2) * BM + tilerow] = tmp.z;
            As[(tilecol + 3) * BM + tilerow] = tmp.w;
        }
        for (int k = 4 * threadIdx.x; k < BK * BN; k += 4 * numThreadsPerBlock) {
            unsigned int Bcol = blockIdx.y * BN + (k % BN);
            unsigned int tilerow = k / BN;
            unsigned int tilecol = k % BN;
            float4 tmp = reinterpret_cast<const float4*>(&B[(p * BK + tilerow) * N + Bcol])[0];
            reinterpret_cast<float4*>(&Bs[tilerow * BN + tilecol])[0] = tmp;
        }
        __syncthreads();

        for (int k = 0; k < BK; k++) {
            for (int wRow = 0; wRow < WMITER; wRow++)
                for (int i = 0; i < TM; i += 4) {
                    float4 m = reinterpret_cast<float4*>(
                        &As[k * BM + warpRow * WM + wRow * WSUBM + threadRowInWarp * TM + i])[0];
                    regM[wRow * TM + i + 0] = m.x; regM[wRow * TM + i + 1] = m.y;
                    regM[wRow * TM + i + 2] = m.z; regM[wRow * TM + i + 3] = m.w;
                }
            for (int wCol = 0; wCol < WNITER; wCol++)
                for (int i = 0; i < TN; i += 4) {
                    float4 n = reinterpret_cast<float4*>(
                        &Bs[k * BN + warpCol * WN + wCol * WSUBN + threadColInWarp * TN + i])[0];
                    regN[wCol * TN + i + 0] = n.x; regN[wCol * TN + i + 1] = n.y;
                    regN[wCol * TN + i + 2] = n.z; regN[wCol * TN + i + 3] = n.w;
                }
            for (int wRow = 0; wRow < WMITER; wRow++)
                for (int wCol = 0; wCol < WNITER; wCol++)
                    for (int i = 0; i < TM; i++)
                        for (int j = 0; j < TN; j++)
                            outerproduct[(wRow * TM + i) * (WNITER * TN) + (wCol * TN) + j]
                                += regM[wRow * TM + i] * regN[wCol * TN + j];
        }
        __syncthreads();
    }

    for (int wRow = 0; wRow < WMITER; wRow++)
        for (int wCol = 0; wCol < WNITER; wCol++)
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j += 4) {
                    unsigned int Crow = blockIdx.x * BM + warpRow * WM + wRow * WSUBM + threadRowInWarp * TM + i;
                    unsigned int Ccol = blockIdx.y * BN + warpCol * WN + wCol * WSUBN + threadColInWarp * TN + j;
                    float4 c = reinterpret_cast<float4*>(&C[Crow * N + Ccol])[0];
                    c.x = alpha * outerproduct[(wRow * TM + i) * (WNITER * TN) + (wCol * TN) + 0] + beta * c.x;
                    c.y = alpha * outerproduct[(wRow * TM + i) * (WNITER * TN) + (wCol * TN) + 1] + beta * c.y;
                    c.z = alpha * outerproduct[(wRow * TM + i) * (WNITER * TN) + (wCol * TN) + 2] + beta * c.z;
                    c.w = alpha * outerproduct[(wRow * TM + i) * (WNITER * TN) + (wCol * TN) + 3] + beta * c.w;
                    reinterpret_cast<float4*>(&C[Crow * N + Ccol])[0] = c;
                }
}

// ---- Kernel 2: YOUR softmax_tile kernel, verbatim ----
__global__ void softmax_tile(const float *A, float *out, unsigned int M, unsigned int N) {
    __shared__ float tile[BLOCKSIZE];
    __shared__ float row_data[N_MAX];
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    if (row >= M) return;

    for (int col = tid; col < N; col += blockDim.x) {
        int idx = row * N + col;
        float a = A[idx];
        row_data[col] = a;
        if (A[idx] > local_max) {
            local_norm *= expf(local_max - a);
            local_max = a;
        }
        local_norm += expf(a - local_max);
    }

    float val = local_max;
    for (int offset = WARPSIZE/2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));

    if (blockDim.x > WARPSIZE) {
        if (tid % WARPSIZE == 0) {
            int idx = tid / WARPSIZE;
            tile[idx] = val;
        }
        __syncthreads();
        if (tid < WARPSIZE) {
            val = (tid < CEIL_DIV(min(N, blockDim.x), WARPSIZE)) ? tile[tid] : -INFINITY;
            for (int offset = WARPSIZE / 2; offset > 0; offset /= 2)
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            if (tid == 0) tile[0] = val;
        }
    } else {
        if (tid == 0) tile[0] = val;
    }
    __syncthreads();

    float row_max = tile[0];
    __syncthreads();

    val = local_norm * expf(local_max - row_max);
    for (int offset = WARPSIZE / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (blockDim.x > WARPSIZE) {
        if (tid % WARPSIZE == 0) {
            tile[tid / WARPSIZE] = val;
        }
        __syncthreads();
        if (tid < WARPSIZE) {
            val = (tid < CEIL_DIV(min(N, blockDim.x), WARPSIZE)) ? tile[tid] : 0.0f;
            for (int offset = WARPSIZE / 2; offset > 0; offset /= 2)
                val += __shfl_down_sync(0xffffffff, val, offset);
            if (tid == 0) tile[0] = val;
        }
    } else {
        if (tid == 0) tile[0] = val;
    }
    __syncthreads();

    float row_norm = tile[0];
    __syncthreads();

    for (int col = tid; col < N; col += blockDim.x) {
        int idx = row * N + col;
        out[idx] = expf(row_data[col] - row_max) / row_norm;
    }
}

int main() {
    const int N = 2048, d = 2048;
    const float scale = 1.0f / sqrtf((float)d);

    size_t szQ  = (size_t)N * d * sizeof(float);
    size_t szKt = (size_t)d * N * sizeof(float);
    size_t szS  = (size_t)N * N * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float *hQ  = (float*)malloc(szQ);
    float *hKt = (float*)malloc(szKt);
    float *hV  = (float*)malloc(szQ);
    float *hO  = (float*)malloc(szQ);
    srand(0);
    for (size_t i = 0; i < (size_t)N * d; ++i) hQ[i]  = (float)rand() / RAND_MAX - 0.5f;
    for (size_t i = 0; i < (size_t)d * N; ++i) hKt[i] = (float)rand() / RAND_MAX - 0.5f;
    for (size_t i = 0; i < (size_t)N * d; ++i) hV[i]  = (float)rand() / RAND_MAX - 0.5f;

    float *dQ, *dKt, *dV, *dO, *dS, *dP;
    CUDA_CHECK(cudaMalloc(&dQ,  szQ));
    CUDA_CHECK(cudaMalloc(&dKt, szKt));
    CUDA_CHECK(cudaMalloc(&dV,  szQ));
    CUDA_CHECK(cudaMalloc(&dO,  szQ));
    CUDA_CHECK(cudaMalloc(&dS,  szS));
    CUDA_CHECK(cudaMalloc(&dP,  szS));
    CUDA_CHECK(cudaMemcpy(dQ,  hQ,  szQ,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dKt, hKt, szKt, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV,  hV,  szQ,  cudaMemcpyHostToDevice));

    dim3 gemmBlock((BM * BN) / (TM * TN * WMITER * WNITER));
    auto gemmGrid = [](int M, int Ncol) { return dim3(CEIL_DIV(M, BM), CEIL_DIV(Ncol, BN), 1); };

    auto run_pipeline = [&]() {
        gemm_warptiling<<<gemmGrid(N, N), gemmBlock>>>(dQ, dKt, dS, 0.0f, scale, N, N, d);
        softmax_tile<<<dim3(N), dim3(BLOCKSIZE)>>>(dS, dP, N, N);
        gemm_warptiling<<<gemmGrid(N, d), gemmBlock>>>(dP, dV, dO, 0.0f, 1.0f, N, d, N);
    };

    run_pipeline();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < 50; ++it) run_pipeline();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop)); ms /= 50;
    double gflops = (2.0 * N * N * d + 2.0 * N * d * N) / (ms * 1e6);
    printf("attention (naive, 3 kernels): %.3f ms, %.1f GFLOP/s\n", ms, gflops);

    CUDA_CHECK(cudaMemcpy(hO, dO, szQ, cudaMemcpyDeviceToHost));

    auto cpu_check_row = [&](int qr) {
        float *s = (float*)malloc(N * sizeof(float));
        float mx = -INFINITY;
        for (int c = 0; c < N; ++c) {
            float acc = 0.0f;
            for (int k = 0; k < d; ++k) acc += hQ[qr * d + k] * hKt[k * N + c];
            s[c] = acc * scale;
            if (s[c] > mx) mx = s[c];
        }
        float nrm = 0.0f;
        for (int c = 0; c < N; ++c) { s[c] = expf(s[c] - mx); nrm += s[c]; }
        for (int c = 0; c < N; ++c) s[c] /= nrm;
        float max_err = 0.0f;
        for (int j = 0; j < d; ++j) {
            float acc = 0.0f;
            for (int c = 0; c < N; ++c) acc += s[c] * hV[c * d + j];
            float e = fabsf(hO[qr * d + j] - acc);
            if (e > max_err) max_err = e;
        }
        free(s);
        return max_err;
    };

    float e0 = cpu_check_row(0);
    float e1 = cpu_check_row(N / 2);
    float e2 = cpu_check_row(N - 1);
    printf("max abs error  row 0: %e   row N/2: %e   row N-1: %e\n", e0, e1, e2);
    printf("O[0,0] = %f\n", hO[0]);

    cudaFree(dQ); cudaFree(dKt); cudaFree(dV); cudaFree(dO); cudaFree(dS); cudaFree(dP);
    free(hQ); free(hKt); free(hV); free(hO);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}