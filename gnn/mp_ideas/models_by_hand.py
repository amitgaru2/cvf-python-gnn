import torch


class GCNConvByHand(torch.nn.Module):
    """
    maps X: B x N x D to y: B x N x O
    A: B x N x N
    """

    def __init__(self, dim_in, dim_out, bias=False, device="cpu"):
        super().__init__()
        self.linear = torch.nn.Linear(dim_in, dim_out, bias=bias)
        self.device = device

    def forward(self, x, A):
        num_nodes = A.shape[1]  # B x N x N
        omega_k = self.linear.weight
        A = A + torch.eye(num_nodes).to(self.device)  # A_cap = A + I
        # normalization D
        eps = 0
        D = torch.sum(A, dim=2)  # degree of each nodes
        D_inv_root = torch.diag_embed(1 / (torch.sqrt(D) + eps)).to(
            self.device
        )  # D_cap_sqrt
        A = (D_inv_root @ A) @ D_inv_root  # B x N x N
        # end of normalization
        h = (
            A @ x
        ) @ omega_k.t()  # ( (O x D) x ( (N x N) x (N x D) = N x D ) = O x N)  => B x O x N
        if self.linear.bias is not None:
            beta_k = self.linear.bias.reshape(1, -1)
            h = (
                torch.reshape(torch.ones(num_nodes).to(self.device), (-1, 1)) @ beta_k
            ) + h

        return h


if __name__ == "__main__":
    B = 2
    N = 3
    D = 1
    O = N

    x = torch.randn((B, N, D))  # B x D x N
    A = torch.randint(2, (B, N, N))  # adjacency matrix ( B x N x N )
    print("A", A)
    gcn = GCNConvByHand(D, O, bias=True)
    result = gcn(x, A)
    print(x)
    print(result)
    print(result.shape)
