import pandas as pd

from torch.utils.data import Dataset


class CVFConfigDataset(Dataset):
    def __init__(self) -> None:
        self.data = pd.read_csv("small_graph_test_config_rank_dataset.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        return row.config, row.rank
