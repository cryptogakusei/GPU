#include <cstdio>
#include <cstdlib>
#include <cmath>

#define WARPSIZE 32
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// ---- TEMPLATED KERNEL (your warptiling kernel, #defines -> template params) ----
template<int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN>
__global__ void gemm_warptiled(const float *A, const float *B, float *C,
                               float beta, float alpha,
                               unsigned int M, unsigned int N, unsigned int K) {
    constexpr int WMITER = (WM * WN) / (WNITER * WARPSIZE * TM * TN);
    static_assert(WMITER >= 1, "invalid config: WMITER < 1");
    static_assert((WM * WN) % (WNITER * WARPSIZE * TM * TN) == 0, "invalid config: WMITER not integer");


    // get warp subtile 
    const int WSUBM = WM / WMITER;
    const int WSUBN = WN / WNITER;


    // location of thread in warp and in the subtike
    uint warpIdx = threadIdx.x / WARPSIZE;
    uint warpRow = warpIdx / (BN / WN);
    uint warpCol = warpIdx % (BN / WN);
    uint threadIdxInWarp = threadIdx.x % WARPSIZE;
    uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);
    uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);

    // define SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // get the registers for doing outer product
    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};
    
    const int numThreadsPerBlock = (BM * BN) / (TM * TN * WMITER * WNITER);

    // for storing the output of partial summaions of outer products
    float outerproduct[WMITER * WNITER * TM * TN] = {0.0};

    
    // now loop over the phases (referring to pmpp terminlogy from chap 5)
    for (int p = 0; p < K/BK; p++) {

        // load the data for A from GMEM to SMEM
        for (int k = 4 * threadIdx.x; k < BM * BK; k += 4 * numThreadsPerBlock) {
            unsigned int Arow = blockIdx.x * BM + (k / BK);
            unsigned int tilerow = k / BK;
            unsigned int tilecol = k % BK;
            // load in reverse order
            float4 tmp = reinterpret_cast<const float4*>(&A[Arow * K + (p * BK + tilecol)])[0];
            As[(tilecol + 0) * BM + tilerow] = tmp.x;
            As[(tilecol + 1) * BM + tilerow] = tmp.y;
            As[(tilecol + 2) * BM + tilerow] = tmp.z;
            As[(tilecol + 3) * BM + tilerow] = tmp.w;
        }    

        // load the data for B from GMEM to SMEM
        for (int k = 4 * threadIdx.x; k < BK * BN; k += 4 * numThreadsPerBlock) {
            unsigned int Bcol = blockIdx.y * BN + (k % BN);
            unsigned int tilerow = k / BN;
            unsigned int tilecol = k % BN;
            float4 tmp = reinterpret_cast<const float4*>(&B[(p*BK + tilerow) * N + Bcol])[0];
            reinterpret_cast<float4*>(&Bs[tilerow * BN + tilecol])[0] = tmp;
        }
        __syncthreads();

        // compute the outerproducts for current thread
        for (int k = 0; k < BK; k++) {

            // prepare the registers
            for (int warpRowIdx = 0; warpRowIdx < WMITER; warpRowIdx++) {
                for (int i = 0; i < TM; i+=4) {
                    float4 m = reinterpret_cast<float4*>(&As[k * BM + warpRow * WM + warpRowIdx * WSUBM + threadRowInWarp * TM + i])[0];
                    regM[warpRowIdx * TM + i + 0] = m.x; regM[warpRowIdx * TM + i + 1] = m.y; regM[warpRowIdx * TM + i + 2] = m.z; regM[warpRowIdx * TM + i + 3] = m.w;
                }
            }

            for (int warpColIdx = 0; warpColIdx < WNITER; warpColIdx++) {
                for (int i = 0; i < TN; i+=4) {
                    float4 n = reinterpret_cast<float4*>(&Bs[k * BN + warpCol * WN + warpColIdx * WSUBN + threadColInWarp * TN + i])[0];
                    regN[warpColIdx * TN + i + 0] = n.x; regN[warpColIdx * TN + i + 1] = n.y; regN[warpColIdx * TN + i + 2] = n.z; regN[warpColIdx * TN + i + 3] = n.w;
                }
            }

            // compute the outerproducts
            for (int warpRowIdx = 0; warpRowIdx < WMITER; warpRowIdx++) {
                for (int warpColIdx = 0; warpColIdx < WNITER; warpColIdx++) {
                    for (int i = 0; i < TM; i++) {
                        for (int j = 0; j < TN; j++) {
                            outerproduct[(warpRowIdx * TM + i) * (WNITER * TN) + (warpColIdx * TN) + j] += regM[warpRowIdx * TM + i] * regN[warpColIdx * TN + j];
                        }
                    }
                }
            }
        }
        __syncthreads();

    }

    for (int warpRowIdx = 0; warpRowIdx < WMITER; warpRowIdx++) {
        for (int warpColIdx = 0; warpColIdx < WNITER; warpColIdx++) {
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j+=4) {
                    unsigned int Crow = blockIdx.x * BM + warpRow * WM + warpRowIdx * WSUBM + threadRowInWarp * TM + i;
                    unsigned int Ccol = blockIdx.y * BN + warpCol * WN + warpColIdx * WSUBN + threadColInWarp * TN + j;
                    float4 c = reinterpret_cast<float4*>(&C[Crow * N + Ccol])[0];
                    c.x = alpha * outerproduct[(warpRowIdx * TM + i) * (WNITER * TN) + (warpColIdx * TN) + 0] + beta * c.x;
                    c.y = alpha * outerproduct[(warpRowIdx * TM + i) * (WNITER * TN) + (warpColIdx * TN) + 1] + beta * c.y;
                    c.z = alpha * outerproduct[(warpRowIdx * TM + i) * (WNITER * TN) + (warpColIdx * TN) + 2] + beta * c.z;
                    c.w = alpha * outerproduct[(warpRowIdx * TM + i) * (WNITER * TN) + (warpColIdx * TN) + 3] + beta * c.w;
                    reinterpret_cast<float4*>(&C[Crow * N + Ccol])[0] = c;
                }
            }
        }
    }
}

// ---- harness: launch one config, time it, check correctness, track best ----
float *g_dA, *g_dB, *g_dC, *g_hC, *g_ref;
int g_M, g_N, g_K;
float g_alpha = 1.0f, g_beta = 0.0f;
double g_bestGflops = 0.0;
const char *g_bestCfg = "none";

template<int BM, int BN, int BK, int WM, int WN, int WNITER, int TM, int TN>
void runConfig(const char *name) {
    constexpr int WMITER = (WM * WN) / (WNITER * WARPSIZE * TM * TN);
    constexpr int numThreads = (BM * BN) / (TM * TN * WMITER * WNITER);

    dim3 grid(CEIL_DIV(g_M, BM), CEIL_DIV(g_N, BN), 1);
    dim3 block(numThreads);

    // warm-up
    gemm_warptiled<BM,BN,BK,WM,WN,WNITER,TM,TN><<<grid, block>>>(
        g_dA, g_dB, g_dC, g_beta, g_alpha, g_M, g_N, g_K);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("%-40s LAUNCH FAIL: %s\n", name, cudaGetErrorString(e)); return; }

    cudaEvent_t s, t; cudaEventCreate(&s); cudaEventCreate(&t);
    cudaEventRecord(s);
    for (int it = 0; it < 50; ++it)
        gemm_warptiled<BM,BN,BK,WM,WN,WNITER,TM,TN><<<grid, block>>>(
            g_dA, g_dB, g_dC, g_beta, g_alpha, g_M, g_N, g_K);
    cudaEventRecord(t); cudaEventSynchronize(t);
    float ms; cudaEventElapsedTime(&ms, s, t); ms /= 50;
    cudaEventDestroy(s); cudaEventDestroy(t);

    double gflops = 2.0 * g_M * g_N * g_K / (ms * 1e6);

    // correctness: compare against precomputed reference (g_ref), spot-check is enough
    cudaMemcpy(g_hC, g_dC, (size_t)g_M*g_N*sizeof(float), cudaMemcpyDeviceToHost);
    float maxerr = 0.f;
    for (int i = 0; i < g_M*g_N; i += 97)   // sample every 97th element for speed
        maxerr = fmaxf(maxerr, fabsf(g_hC[i] - g_ref[i]));

    bool ok = maxerr < 1e-2f;
    printf("%-40s %8.1f GFLOP/s  err=%.1e  threads=%d %s\n",
           name, gflops, maxerr, numThreads, ok ? "" : "  <-- WRONG");
    if (ok && gflops > g_bestGflops) { g_bestGflops = gflops; g_bestCfg = name; }
}

#define RUN_CONFIG(bm,bn,bk,wm,wn,wniter,tm,tn) \
    runConfig<bm,bn,bk,wm,wn,wniter,tm,tn>(#bm"x"#bn"x"#bk" W"#wm"x"#wn" WN"#wniter" T"#tm"x"#tn)

int main() {
    g_M = g_N = g_K = 2048;
    size_t szA = (size_t)g_M*g_K*4, szB = (size_t)g_K*g_N*4, szC = (size_t)g_M*g_N*4;

    float *hA = (float*)malloc(szA), *hB = (float*)malloc(szB);
    g_hC = (float*)malloc(szC); g_ref = (float*)malloc(szC);
    srand(0);
    for (int i = 0; i < g_M*g_K; ++i) hA[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < g_K*g_N; ++i) hB[i] = (float)rand()/RAND_MAX;

    // CPU reference once 
    printf("computing CPU reference...\n");
    for (int r = 0; r < g_M; ++r)
        for (int c = 0; c < g_N; ++c) {
            float acc = 0.f;
            for (int k = 0; k < g_K; ++k) acc += hA[r*g_K+k]*hB[k*g_N+c];
            g_ref[r*g_N+c] = acc;
        }

    CUDA_CHECK(cudaMalloc(&g_dA, szA)); CUDA_CHECK(cudaMalloc(&g_dB, szB)); CUDA_CHECK(cudaMalloc(&g_dC, szC));
    CUDA_CHECK(cudaMemcpy(g_dA, hA, szA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_dB, hB, szB, cudaMemcpyHostToDevice));

    printf("sweeping configs...\n");

    RUN_CONFIG(128, 64, 8, 64, 32, 2, 4, 4);  
    RUN_CONFIG(128,128, 8, 64, 64, 4, 8, 4);
    RUN_CONFIG(128,128, 8, 64, 64, 2, 8, 8);
    RUN_CONFIG(128,128,16, 64, 64, 4, 8, 4);
    RUN_CONFIG(128, 64, 8, 64, 32, 1, 8, 4);
    RUN_CONFIG(128,128, 8, 32, 64, 2, 4, 8);
    RUN_CONFIG( 64,128, 8, 32, 64, 2, 4, 8);
    RUN_CONFIG(256,128, 8, 64, 64, 4, 8, 4);
    RUN_CONFIG(128,128, 8, 64, 32, 2, 8, 4);

    
    printf("\nBEST: %s at %.1f GFLOP/s\n", g_bestCfg, g_bestGflops);

    cudaFree(g_dA); cudaFree(g_dB); cudaFree(g_dC);
    free(hA); free(hB); free(g_hC); free(g_ref);
    return 0;
}