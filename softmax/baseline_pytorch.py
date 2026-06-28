import torch
import torch.nn as nn
import time


m = torch.randn(1024, 32768, device='cuda', dtype=torch.float32)

_ = torch.softmax(m, dim=-1)

# ensure all cuda ops are over
torch.cuda.synchronize()  

tot_time = 0
num_rounds = 5

for i in range(num_rounds):
    # Measure time
    torch.cuda.synchronize()  # Ensure all CUDA operations are finished
    start = time.time()
    _ = torch.nn.functional.softmax(m, dim=-1)
    torch.cuda.synchronize()  # Synchronize again
    end = time.time()
    
    tot_time += (end - start) * 1000
    print(tot_time)

print(f"Softmax computation time (average): {(tot_time/num_rounds):.3f} ms")