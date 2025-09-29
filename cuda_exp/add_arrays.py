import time

import numpy as np

from numba import cuda

start_time = time.time()


# Define a simple GPU kernel
@cuda.jit
def add_kernel(A, B, C):
    # Compute the thread's absolute index
    print(
        "ThreadIdx.x =",
        cuda.threadIdx.x,
        "BlockIdx.x =",
        cuda.blockIdx.x,
        "BlockDim.x =",
        cuda.blockDim.x,
        "Global ID i =",
        cuda.grid(1),
    )
    i = cuda.grid(
        1
    )  # i = threadIdx.x + blockIdx.x * blockDim.x ; threadIdx.x is the index within the block (0 to blockDim.x-1) and blockIdx.x is the block index (0 to gridDim.x-1)
    if i < A.size:
        C[i] = A[i] + B[i]
        print("Thread", i, "adding", A[i], "+", B[i])


# Host data (CPU arrays)
A = np.array([i for i in range(100)], dtype=np.float32)
B = np.array([10 * i for i in range(100)], dtype=np.float32)
C = np.zeros_like(A)

# Copy data to GPU
d_A = cuda.to_device(A)
d_B = cuda.to_device(B)
d_C = cuda.device_array_like(C)

# Launch kernel: 1 block of 5 threads
threads_per_block = 10
blocks = 5
add_kernel[blocks, threads_per_block](d_A, d_B, d_C)

# Copy result back to CPU
d_C.copy_to_host(C)

# print("A:", A)
# print("B:", B)
# print("C = A + B:", C)
print("Total time taken:", time.time() - start_time, "seconds")
