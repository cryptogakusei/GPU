#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cublas_v2.h>


//. this is for tiling to load from GMEM to SMEM 
#define BM 64
#define BN 64
#define BK 8

// note that this for each thread how much they computing -- to increase Arithmetci intensity
#define TM 8
#define TN 8

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
        
        const int numThreadsPerTile = (BM * BN) / (TM * TN);

        // for storing the output of partial summaions of outer products
        float outerproduct[TM * TN] = {0.0};

        // get the registers for doing outer product
        float regM[TM] = {0.0};
        float regN[TN] = {0.0};

        unsigned threadrow = threadIdx.x / (BN / TN);
        unsigned threadcol = threadIdx.x % (BN / TN);
        
        // now loop over the phases (referring to pmpp terminlogy from chap 5)
        for (int p = 0; p < K/BK; p++) {

            // load the data for A from GMEM to SMEM
            for (int k = 4 * threadIdx.x; k < BM * BK; k += 4 * numThreadsPerTile) {
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
            for (int k = 4 * threadIdx.x; k < BK * BN; k += 4 * numThreadsPerTile) {
                unsigned int Bcol = blockIdx.y * BN + (k % BN);
                unsigned int tilerow = k / BN;
                unsigned int tilecol = k % BN;
                float4 tmp = reinterpret_cast<const float4*>(&B[(p*BK + tilerow) * N + Bcol])[0];
                reinterpret_cast<float4*>(&Bs[tilerow * BN + tilecol])[0] = tmp;
            }
            __syncthreads();

            // compute the outerproducts for current thread
            for (int k = 0; k < BK; k++) {
                for (int i = 0; i < TM; i+=4) {
                    float4 m = reinterpret_cast<float4*>(&As[k * BM + threadrow * TM + i])[0];
                    regM[i + 0] = m.x; regM[i + 1] = m.y; regM[i + 2] = m.z; regM[i + 3] = m.w;
                }
                for (int i = 0; i < TN; i+=4) {
                    float4 n = reinterpret_cast<float4*>(&Bs[k * BN + threadcol * TN + i])[0];
                    regN[i + 0] = n.x; regN[i + 1] = n.y; regN[i + 2] = n.z; regN[i + 3] = n.w;
                }
                for (int i = 0; i < TM; i++) {
                    for (int j = 0; j < TN; j++) {
                        outerproduct[i * TN + j] += regM[i] * regN[j];
                    }
                }
            }
            __syncthreads();

        }

        for (int i = 0; i < TM; i++) {
            for (int j = 0; j < TN; j+=4) {
                unsigned int Crow = blockIdx.x * BM + threadrow * TM + i;
                unsigned int Ccol = blockIdx.y * BN + threadcol * TN + j;
                float4 c = reinterpret_cast<float4*>(&C[Crow * N + Ccol])[0];
                c.x = alpha * outerproduct[i * TN + j + 0] + beta * c.x;
                c.y = alpha * outerproduct[i * TN + j + 1] + beta * c.y;
                c.z = alpha * outerproduct[i * TN + j + 2] + beta * c.z;
                c.w = alpha * outerproduct[i * TN + j + 3] + beta * c.w;
                reinterpret_cast<float4*>(&C[Crow * N + Ccol])[0] = c;
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
        dim3 blockDim((BM * BN) / (TM * TN));
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
