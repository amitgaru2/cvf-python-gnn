import torch

# Define matrix A and B
A = torch.tensor([0, 1, 2, 2]).view(1, 4)  # A is a 1x4 matrix
B = torch.tensor([[1, 1, 0, 1],
                  [1, 1, 1, 0],
                  [0, 1, 1, 1],
                  [1, 0, 1, 1]])  # B is a 4x4 matrix

# Perform matrix multiplication
C = torch.mm(A, B)

print(C)