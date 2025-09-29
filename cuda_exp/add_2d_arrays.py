import numpy as np

from numba import cuda


# Kernel to add two 2D arrays
@cuda.jit
def add_2d_kernel(A, B, C):
    # Compute global 2D thread coordinates
    x, y = cuda.grid(2)
    print("x, y", x, y)

    # Check bounds (important for last block)
    if x < A.shape[1] and y < A.shape[0]:
        C[y, x] = A[y, x] + B[y, x]


# Host arrays (4x4 example)
A = np.array(
    [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32
)

B = np.ones_like(A)
C = np.zeros_like(A)

# Threads per block (2x2 threads per block)
threads_per_block = (2, 2)

# Compute number of blocks per grid
blocks_per_grid_x = (A.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (A.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]

blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

# Launch kernel
add_2d_kernel[blocks_per_grid, threads_per_block](A, B, C)

print("A =\n", A)
print("B =\n", B)
print("C = A + B =\n", C)
