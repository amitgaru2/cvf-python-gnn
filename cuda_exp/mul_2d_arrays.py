import numpy as np

from numba import cuda


@cuda.jit
def matmul_2d(A, B, C):
    # Compute global row and column indices
    row, col = cuda.grid(2)

    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.0
        for k in range(A.shape[1]):  # iterate over inner dimension
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


# Example matrices
A = np.array([[1, 2], [3, 4]], dtype=np.float32)  # 2x2
B = np.array([[5, 6], [7, 8]], dtype=np.float32)  # 2x2
C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)

# Threads per block (2x2 threads)
threads_per_block = (2, 2)

# Blocks per grid
blocks_per_grid_x = (C.shape[1] + threads_per_block[0] - 1) // threads_per_block[0]
blocks_per_grid_y = (C.shape[0] + threads_per_block[1] - 1) // threads_per_block[1]
blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

print("blocks_per_grid:", blocks_per_grid, "threads_per_block:", threads_per_block)

# Launch kernel
matmul_2d[blocks_per_grid, threads_per_block](A, B, C)

print("A =\n", A)
print("B =\n", B)
print("C = A x B =\n", C)
