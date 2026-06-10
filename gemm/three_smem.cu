#include <cstdio>
#include <cstdlib>
#include <cmath>

#define TILEWIDTH 32
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void gemm_smem_cache(const float *A, const float *B, float *C, float beta, 
    float alpha, unsigned int M, unsigned int N, unsigned int K) {

        __shared__ float As[TILEWIDTH * TILEWIDTH];
        __shared__ float Bs[TILEWIDTH * TILEWIDTH];
        
        // calculate the position in C that this thread will be responsible for
        unsigned int row = blockIdx.x * TILEWIDTH + (threadIdx.x / TILEWIDTH);
        unsigned int col = blockIdx.y * TILEWIDTH + (threadIdx.x % TILEWIDTH);
        unsigned int tilerow = threadIdx.x / TILEWIDTH;
        unsigned int tilecol = threadIdx.x % TILEWIDTH;
        
        float dotprod = 0.0f;

        // now loop over the phases (referring to pmpp terminlogy from chap 5)
        for (int p = 0; p < K/TILEWIDTH; p++) {

            // first load the data from GMEM to SMEM
            As[tilerow * TILEWIDTH + tilecol] = A[row * K + (p * TILEWIDTH + tilecol)];
            Bs[tilerow * TILEWIDTH + tilecol] = B[(p * TILEWIDTH + tilerow) * N + col];
            __syncthreads(); // needed so that when we do dot product, all data in tile should have been loaded

            // compute the dotproduct for the thread in the tile within the current phase
            for (int k = 0; k < TILEWIDTH; k++) {
                dotprod += As[tilerow * TILEWIDTH + k] * Bs[k * TILEWIDTH + tilecol];
            }
            __syncthreads(); // this needed so that smem doesn't get overwritten by faster threads in next phase before slower threads are done computing dot product for this phase
        }
        
        if (row < M && col < N) {
            C[row * N + col] = alpha * dotprod + beta * C[row * N + col];
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
        dim3 gridDim(CEIL_DIV(M, TILEWIDTH), CEIL_DIV(N, TILEWIDTH), 1); // gird sizing is done based on tile size, not block size
        dim3 blockDim(TILEWIDTH * TILEWIDTH);
        gemm_smem_cache<<<gridDim, blockDim>>>(dA, dB, dC, beta, alpha, M, N, K);
        CUDA_CHECK(cudaGetLastError()); 
        CUDA_CHECK(cudaDeviceSynchronize());

        /// timed loop
        CUDA_CHECK(cudaEventRecord(start));
        for (int it = 0; it < 100; ++it)
            gemm_smem_cache<<<gridDim, blockDim>>>(dA, dB, dC, beta, alpha, M, N, K);
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
