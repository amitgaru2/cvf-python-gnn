import torch


class GCNConvByHand(torch.nn.Module):
    """
    maps X: B x N x D to y: B x N x O
    A: B x N x N
    """

    def __init__(self, dim_in, dim_out, device="cpu"):
        super().__init__()
        self.linear = torch.nn.Linear(dim_in, dim_out, bias=True)
        self.device = device

    def forward(self, x, A):
        num_nodes = A.shape[1]  # B x N x N
        omega_k = self.linear.weight
        beta_k = self.linear.bias.reshape(1, -1)
        A = A + torch.eye(num_nodes).to(self.device)
        h = torch.matmul(
            torch.reshape(torch.ones(num_nodes).to(self.device), (-1, 1)), beta_k
        ) + torch.matmul(
            torch.matmul(A, x), omega_k.t()
        )  # ( (O x D) x ( (N x N) x (N x D) = N x D ) = O x N)  => B x O x N
        return h
