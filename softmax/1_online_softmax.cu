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


__global__ void softmax_naive(const float *A, float *out, unsigned int M, unsigned int N) {
    unsigned int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        // compute max for this row
        float x_max = -INFINITY;
        float norm = 0.0f;

        for (int col = 0; col < N; col++) {
            int idx = row * N + col;

            // correction if max is changing
            if (A[idx] > x_max) {
                norm *= expf(x_max - A[idx]);
                x_max = A[idx];
            }
            norm += expf(A[idx] - x_max);
        }

        // compute the ratios
        for (int col = 0; col < N; col++) {
            int idx = row * N + col;
            out[idx] = expf(A[idx] - x_max) / norm;
        }
    }
}

int main() {
    int M = 2048, N = 2048;
    size_t szA = M*N*sizeof(float), szOut = M*N*sizeof(float);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    //allocate and fill on the host (CPU)
    float *hA = (float*)malloc(szA), *hOut = (float*)malloc(szOut);
    for (int i = 0; i < M*N; ++i) hA[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;

    // location on the device (GPU global mem)
    float *dA, *dOut;
    CUDA_CHECK(cudaMalloc(&dA, szA));
    CUDA_CHECK(cudaMalloc(&dOut, szOut));

    // copy inputs from host to device
    CUDA_CHECK(cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice));

    // launch the kernel - warmup
    dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), 1);
    dim3 blockDim(BLOCKSIZE, 1);
    softmax_naive<<<gridDim, blockDim>>>(dA, dOut, M, N);
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize());

    /// timed loop
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < 100; ++it)
        softmax_naive<<<gridDim, blockDim>>>(dA, dOut, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= 100;
    double bw = 3.0 * M * N * sizeof(float) / (ms * 1e6);
    printf("%.3f ms, %.1f GB/s\n", ms, bw);

    // copy result to host
    CUDA_CHECK(cudaMemcpy(hOut, dOut, szOut, cudaMemcpyDeviceToHost));

    // CPU reference for row 0
    float cpu_max = -INFINITY;
    for (int j = 0; j < N; j++) cpu_max = fmaxf(cpu_max, hA[j]);
    float cpu_norm = 0.0f;
    for (int j = 0; j < N; j++) cpu_norm += expf(hA[j] - cpu_max);
    float cpu_out0 = expf(hA[0] - cpu_max) / cpu_norm;

    printf("out[0] = %f (expect %f)\n", hOut[0], cpu_out0);

    // cleanup
    cudaFree(dA);
    cudaFree(dOut);
    free(hA);
    free(hOut);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
