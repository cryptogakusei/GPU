#include <cstdio>
#include <cstdlib>


#define BLOCKSIZE 256
#define WARPSIZE 32
#define N_MAX 2048
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


__global__ void softmax_tile(const float *A, float *out, unsigned int M, unsigned int N) {
    __shared__ float tile[BLOCKSIZE]; // number of threads from the block working on this row
    __shared__ float row_data[N_MAX]; // for sotring A's row
    unsigned int row = blockIdx.x; // all threads in this block will work on same row
    unsigned int tid = threadIdx.x; 
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    if (row >= M) return; // access check

    // THREAD-level: first get set of local max and local norm for each thread
    for (int col = tid; col < N; col += blockDim.x) {
        int idx = row * N + col;
        float a = A[idx];
        row_data[col] = a;

        // correction if max is changing
        if (A[idx] > local_max) {
            local_norm *= expf(local_max - a);
            local_max = a;
        }
        local_norm += expf(a - local_max);
    }
    
    // WARP level: do shuffle thing to get max within a warp
    float val = local_max;
    for (int offset = WARPSIZE/2; offset > 0; offset /=2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    
    // BLOCK level: do shuffle thing to get max within a block
    if (blockDim.x > WARPSIZE) { 
        // checking if block greater than warp size

        if (tid % WARPSIZE == 0) {
            // only the first threads in each warp - they contaim max within the warp
            int idx = tid / WARPSIZE; 
            tile[idx] = val;
        }
        __syncthreads();

        // now reductin within the tile
        if (tid < WARPSIZE) {
            val = (tid < CEIL_DIV(min(N, blockDim.x), WARPSIZE)) ? tile[tid] : -INFINITY;
            for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            }
            if (tid == 0) tile[0] = val;
        }
    } else {
        if (tid == 0) tile[0] = val;
    }
    __syncthreads();

    float row_max = tile[0];
    __syncthreads();

    val = local_norm * expf(local_max - row_max);
    for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (blockDim.x > WARPSIZE) {
        if (tid % WARPSIZE == 0) {
            tile[tid / WARPSIZE] = val;
        }
        __syncthreads();

        if (tid < WARPSIZE) {
            val = (tid < CEIL_DIV(min(N, blockDim.x), WARPSIZE)) ? tile[tid] : 0.0f;
            for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (tid == 0) tile[0] = val;
        }
    } else {
        if (tid == 0) tile[0] = val;
    }
    __syncthreads();

    float row_norm = tile[0];
    __syncthreads();
    

    // compute the ratios
    for (int col = tid; col < N; col += blockDim.x) {
        int idx = row * N + col;
        out[idx] = expf(row_data[col] - row_max) / row_norm;
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
    dim3 gridDim(M, 1);
    dim3 blockDim(BLOCKSIZE, 1);
    softmax_tile<<<gridDim, blockDim>>>(dA, dOut, M, N);
    CUDA_CHECK(cudaGetLastError()); 
    CUDA_CHECK(cudaDeviceSynchronize());

    /// timed loop
    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < 100; ++it)
        softmax_tile<<<gridDim, blockDim>>>(dA, dOut, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= 100;
    double bw = 2.0 * M * N * sizeof(float) / (ms * 1e6);
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
