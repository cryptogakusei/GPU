#include <cstdio>
#include <cstdlib>
#include <cmath>

#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void gemm_one_d_blocktiling(const float *A, const float *B, float *C, float beta, 
    float alpha, unsigned int M, unsigned int N, unsigned int K) {

        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];
        
        const int numThreadsPerTile = BM * BN / TM;
        
        float dotprod[TM];
        for (int k = 0; k < TM; k++) dotprod[k] = 0.0f; // this is for storing rows for each thread --   will increase artihmetic intensity (FLOPs/byte)

        // now loop over the phases (referring to pmpp terminlogy from chap 5)
        for (int p = 0; p < K/BK; p++) {

            // load the data for A from GMEM to SMEM
            for (int k = threadIdx.x; k < BM*BK; k += numThreadsPerTile) {
                unsigned int Arow = blockIdx.x * BM + (k / BK);
                unsigned int tilerow = k / BK;
                unsigned int tilecol = k % BK;
                As[tilerow * BK + tilecol] = A[Arow * K + (p * BK + tilecol)];
                // As[tilecol * BM + tilerow] = A[Arow * K + (p * BK + tilecol)];
            }    

            // load the data for B from GMEM to SMEM
            for (int k = threadIdx.x; k < BK*BN; k += numThreadsPerTile) {
                unsigned int Bcol = blockIdx.y * BN + (k % BN);
                unsigned int tilerow = k / BN;
                unsigned int tilecol = k % BN;
                Bs[tilerow * BN + tilecol] = B[(p * BK + tilerow) * N + Bcol];
            }
            __syncthreads();

            // compute the dotproduct for the thread in the tile within the current phase
            // unsigned int tilerow = threadIdx.x % (BM / TM);
            // unsigned int tilecol = threadIdx.x / (BM / TM);
            unsigned int tilerow = threadIdx.x / BN;
            unsigned int tilecol = threadIdx.x % BN;
            for (int k = 0; k < BK; k++) {
                float Btmp = Bs[(k * BN) + tilecol];
                for (int j = 0; j < TM; j++) {
                    dotprod[j] += As[(tilerow * TM + j) * BK + k] * Btmp;
                    // dotprod[j] += As[k * BM + (tilerow * TM + j)] * Btmp; 
                }
            }
            __syncthreads();

        }

        unsigned int Crow_start = (blockIdx.x * BM) + (threadIdx.x / BN) * TM;
        unsigned int Ccol = (blockIdx.y * BN) + (threadIdx.x % BN);

        for (unsigned int row = 0; row < TM; row++) {
            C[(Crow_start + row) * N + Ccol] = alpha * dotprod[row] + beta * C[(Crow_start + row) * N + Ccol];
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
        dim3 blockDim((BM * BN) / (TM));
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
