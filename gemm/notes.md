nvcc -arch=sm_86 -O3 gemm/naive.cu -o naive && ./naive   # compile + run on the A10
git add -A && git commit -m "..." && git push            # sync to GitHub