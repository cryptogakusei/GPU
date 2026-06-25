#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cublas_v2.h>


// dimension of block for load from GMEM to SMEM 
#define BM 128
#define BN 64
#define BK 8

// dimension of each warp
#define WM 64
#define WN 32

// number of subtiles 
#define WNITER 2
#define WMITER 2

// note that this for each thread how much they computing
#define TM 4
#define TN 4

// number of threads in a warp
#define WARPSIZE 32

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// correctness check
static_assert((WM * WN) % (WNITER * WARPSIZE * TM * TN) == 0, "bad warp config");
static_assert(WMITER == (WM * WN) / (WNITER * WARPSIZE * TM * TN), "WMITER mismatch");

__global__ void gemm_one_d_blocktiling(const float *A, const float *B, float *C, float beta, 
    float alpha, unsigned int M, unsigned int N, unsigned int K) {

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


    int main() {
        int M = 2048, N = 2048, K = 2048;
        size_t szA = M*K*sizeof(float), szB = K*N*sizeof(float), szC = M*N*sizeof(float);
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        //allocate and fill on the host (CPU)
        float *hA = (float*)malloc(szA), *hB = (float*)malloc(szB), *hC = (float*)malloc(szC);
        srand(0);
        for (int i = 0; i < M*K; ++i) hA[i] = (float)rand() / RAND_MAX;
        for (int i = 0; i < K*N; ++i) hB[i] = (float)rand() / RAND_MAX;
        for (int i = 0; i < M*N; ++i) hC[i] = (float)rand() / RAND_MAX;
        float alpha = 1.0f, beta = 0.0f;

        // qlocation on the device (GPU global mem)
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, szA));
        CUDA_CHECK(cudaMalloc(&dB, szB));
        CUDA_CHECK(cudaMalloc(&dC, szC));

        // copy inputs from host to device
        CUDA_CHECK(cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, szB, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dC, hC, szC, cudaMemcpyHostToDevice));

        // launch the kernel - warmup
        dim3 gridDim(CEIL_DIV(M, BM), CEIL_DIV(N, BN), 1); // gird sizing is done based on tile size, not block size
        dim3 blockDim((BM * BN) / (TM * TN * WMITER * WNITER));
        gemm_one_d_blocktiling<<<gridDim, blockDim>>>(dA, dB, dC, beta, alpha, M, N, K);
        CUDA_CHECK(cudaGetLastError()); 
        CUDA_CHECK(cudaDeviceSynchronize());

        /// timed loop
        CUDA_CHECK(cudaEventRecord(start));
        for (int it = 0; it < 100; ++it)
            gemm_one_d_blocktiling<<<gridDim, blockDim>>>(dA, dB, dC, beta, alpha, M, N, K);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        ms /= 100;
        double gflops = 2.0 * M * N * K / (ms * 1e6);
        printf("%.3f ms, %.1f GFLOP/s\n", ms, gflops);

        // copy result to host
        CUDA_CHECK(cudaMemcpy(hC, dC, szC, cudaMemcpyDeviceToHost));
        printf("C[0] = %f (expect %d)\n", hC[0], K);
        printf("C[last] = %f (expect %d)\n", hC[M*N - 1], K);   // row 255, col 255

        // Correctness
        float max_err = 0.0f;
        for (int r = 0; r < M; ++r) {
            for (int c = 0; c < N; ++c) {
                float ref = 0.0f;
                for (int k = 0; k < K; ++k)
                    ref += hA[r*K + k] * hB[k*N + c];
                float e = fabsf(hC[r*N + c] - ref);
                if (e > max_err) max_err = e;
            }
        }
        printf("max abs error: %e\n", max_err);

        // // ---- cuBLAS benchmark
        // cublasHandle_t handle;
        // cublasCreate(&handle);
        // cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH);   // force TRUE fp32 (no TF32)

        // // row-major C = A*B  ==  col-major C^T = B^T * A^T -> pass B then A, all OP_N
        // cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        //             N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);   // warm-up
        // CUDA_CHECK(cudaDeviceSynchronize());

        // CUDA_CHECK(cudaEventRecord(start));
        // for (int it = 0; it < 100; ++it)
        //     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        //                 N, M, K, &alpha, dB, N, dA, K, &beta, dC, N);
        // CUDA_CHECK(cudaEventRecord(stop));
        // CUDA_CHECK(cudaEventSynchronize(stop));

        // float ms_cublas;
        // CUDA_CHECK(cudaEventElapsedTime(&ms_cublas, start, stop));
        // ms_cublas /= 100;
        // double gflops_cublas = 2.0 * M * N * K / (ms_cublas * 1e6);
        // printf("cuBLAS (fp32): %.3f ms, %.1f GFLOP/s\n", ms_cublas, gflops_cublas);
        // printf("your kernel = %.1f%% of cuBLAS\n", 100.0 * gflops / gflops_cublas);

        // // verify cuBLAS matches CPU too (sanity on the swap trick)
        // CUDA_CHECK(cudaMemcpy(hC, dC, szC, cudaMemcpyDeviceToHost));
        // float cublas_err = 0.0f;
        // for (int r = 0; r < M; ++r)
        //     for (int c = 0; c < N; ++c) {
        //         float ref = 0.0f;
        //         for (int k = 0; k < K; ++k) ref += hA[r*K + k] * hB[k*N + c];
        //         float e = fabsf(hC[r*N + c] - ref);
        //         if (e > cublas_err) cublas_err = e;
        //     }
        // printf("cuBLAS max abs error: %e (confirms swap-trick layout)\n", cublas_err);

        // cublasDestroy(handle);



        // cleanup
        cudaFree(dA);
        cudaFree(dB);
        cudaFree(dC);
        free(hA);
        free(hB);
        free(hC);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return 0;
    }
