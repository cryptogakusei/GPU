#include <cstdio>
#include <cstdlib>

#define BLOCKSIZE 32
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void gemm_naive(const float *A, const float *B, float *C, float beta, 
    float alpha, unsigned int M, unsigned int N, unsigned int K) {
        
        // calculate the position in C that this thread will be responsible for
        unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
        unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

        // do the matmul
        if (x < M && y < N) {
            float tmp = 0.0f;
            for (int i = 0; i < K; i++) {
                tmp += A[x * K + i] * B[i * N + y];
            }
            C[x * N + y] = alpha * tmp + beta * C[x * N + y];
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
        for (int i = 0; i < M*K; ++i) hA[i] = 1.0f;
        for (int i = 0; i < K*N; ++i) hB[i] = 1.0f;
        for (int i = 0; i < M*N; ++i) hC[i] = 1.0f;
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
        dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE), 1);
        dim3 blockDim(BLOCKSIZE, BLOCKSIZE, 1);
        gemm_naive<<<gridDim, blockDim>>>(dA, dB, dC, beta, alpha, M, N, K);
        CUDA_CHECK(cudaGetLastError()); 
        CUDA_CHECK(cudaDeviceSynchronize());

        /// timed loop
        CUDA_CHECK(cudaEventRecord(start));
        for (int it = 0; it < 100; ++it)
            gemm_naive<<<gridDim, blockDim>>>(dA, dB, dC, beta, alpha, M, N, K);
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
