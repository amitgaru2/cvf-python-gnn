import torch

from torch.utils.data import DataLoader, Dataset

from helpers import CVFConfigForGCNWSuccLSTMDataset


class NormalizeTransform:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.std[self.std == 0] = 1  # Avoid divide-by-zero

    def __call__(self, x):
        return (x - self.mean) / self.std


def compute_mean_std(loader):
    sum_ = 0.0
    sum_sq_diff = 0.0
    n_samples = len(loader.dataset)

    for X, _ in loader:
        data = X[0]
        sum_ += data[:, :, 0].sum()

    mean = sum_ / n_samples

    for X, _ in loader:
        diff = X[0][:, :, 0]
        diff = diff - mean
        sum_sq_diff += (diff**2).sum()

    std = torch.sqrt(sum_sq_diff / (n_samples - 1))
    return mean, std


if __name__ == "__main__":
    dataset = CVFConfigForGCNWSuccLSTMDataset(
        "cuda",
        "star_graph_n7_config_rank_dataset.csv",
        "star_graph_n7_edge_index.json",
    )
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False)
    mean, std = compute_mean_std(dataloader)
    print(mean, std)
    print(mean.shape, std.shape)

    # transform = NormalizeTransform(mean, std)

    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # for batch in dataloader:
    #     x = batch[0]
    #     print(x[0])
    #     print(transform(x[0]))
    #     break
