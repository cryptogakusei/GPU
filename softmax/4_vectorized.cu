// this code is Claude generated: only for learning purpose

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


__global__ void softmax_float4(const float *A, float *out, unsigned int M, unsigned int N) {
    __shared__ float tile[BLOCKSIZE / WARPSIZE]; // one slot per warp
    __shared__ float row_data[N_MAX];            // cached row (smem)
    unsigned int row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    float local_max = -INFINITY;
    float local_norm = 0.0f;

    if (row >= M) return;

    // Vectorized base pointers for this row (16-byte aligned: cudaMalloc + N%4==0)
    const float4 *A4   = reinterpret_cast<const float4*>(A + row * N);
    float4       *row4 = reinterpret_cast<float4*>(row_data);
    unsigned int N4 = N / 4; // assumes N % 4 == 0

    // THREAD-level: one 128-bit load brings 4 elements; run online-softmax over all 4
    for (int c4 = tid; c4 < N4; c4 += blockDim.x) {
        float4 v = A4[c4];
        row4[c4] = v;  // stash all 4 into smem (one 128-bit store)

        // unroll the online correction over v.x, v.y, v.z, v.w
        float vals[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            float a = vals[j];
            if (a > local_max) {
                local_norm *= expf(local_max - a);
                local_max = a;
            }
            local_norm += expf(a - local_max);
        }
    }

    int n_warps = CEIL_DIV(min((int)N, (int)blockDim.x), WARPSIZE);

    // ---- MAX reduction ----
    float val = local_max;
    for (int offset = WARPSIZE/2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));

    if (blockDim.x > WARPSIZE) {
        if (tid % WARPSIZE == 0) tile[tid / WARPSIZE] = val;
        __syncthreads();
        if (tid < WARPSIZE) {
            val = (tid < n_warps) ? tile[tid] : -INFINITY;
            for (int offset = WARPSIZE/2; offset > 0; offset /= 2)
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
    for (int offset = WARPSIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);

    if (blockDim.x > WARPSIZE) {
        if (tid % WARPSIZE == 0) tile[tid / WARPSIZE] = val;
        __syncthreads();
        if (tid < WARPSIZE) {
            val = (tid < n_warps) ? tile[tid] : 0.0f;
            for (int offset = WARPSIZE/2; offset > 0; offset /= 2)
                val += __shfl_down_sync(0xffffffff, val, offset);
            if (tid == 0) tile[0] = val;
        }
    } else {
        if (tid == 0) tile[0] = val;
    }
    __syncthreads();
    float row_norm = tile[0];
    __syncthreads();

    // output
    float4 *out4 = reinterpret_cast<float4*>(out + row * N);
    float inv = 1.0f / row_norm; 
    for (int c4 = tid; c4 < N4; c4 += blockDim.x) {
        float4 v = row4[c4];
        float4 o;
        o.x = expf(v.x - row_max) * inv;
        o.y = expf(v.y - row_max) * inv;
        o.z = expf(v.z - row_max) * inv;
        o.w = expf(v.w - row_max) * inv;
        out4[c4] = o;
    }
}

int main() {
    int M = 2048, N = 2048;
    size_t szA = M*N*sizeof(float), szOut = M*N*sizeof(float);
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float *hA = (float*)malloc(szA), *hOut = (float*)malloc(szOut);
    for (int i = 0; i < M*N; ++i) hA[i] = ((float)rand() / RAND_MAX) * 10.0f - 5.0f;

    float *dA, *dOut;
    CUDA_CHECK(cudaMalloc(&dA, szA));
    CUDA_CHECK(cudaMalloc(&dOut, szOut));
    CUDA_CHECK(cudaMemcpy(dA, hA, szA, cudaMemcpyHostToDevice));

    dim3 gridDim(M, 1);
    dim3 blockDim(BLOCKSIZE, 1);
    softmax_float4<<<gridDim, blockDim>>>(dA, dOut, M, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int it = 0; it < 100; ++it)
        softmax_float4<<<gridDim, blockDim>>>(dA, dOut, M, N);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= 100;
    double bw = 2.0 * M * N * sizeof(float) / (ms * 1e6);
    printf("%.3f ms, %.1f GB/s\n", ms, bw);

    CUDA_CHECK(cudaMemcpy(hOut, dOut, szOut, cudaMemcpyDeviceToHost));

    // CPU reference for row 0
    float cpu_max = -INFINITY;
    for (int j = 0; j < N; j++) cpu_max = fmaxf(cpu_max, hA[j]);
    float cpu_norm = 0.0f;
    for (int j = 0; j < N; j++) cpu_norm += expf(hA[j] - cpu_max);
    float cpu_out0 = expf(hA[0] - cpu_max) / cpu_norm;
    printf("out[0] = %f (expect %f)\n", hOut[0], cpu_out0);

    cudaFree(dA);
    cudaFree(dOut);
    free(hA);
    free(hOut);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}