import os
import ast

import torch
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CVFConfigDataset(Dataset):
    def __init__(self, dataset_file) -> None:
        self.data = pd.read_csv(os.path.join("datasets", dataset_file))
        self.nodes = len(self[0][0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        result = (
            torch.tensor(
                [[i] for i in ast.literal_eval(row["config"])], dtype=torch.float32
            ),
            row["rank"],
        )
        return result


if __name__ == "__main__":
    dataset = CVFConfigDataset("graph_4_config_rank_dataset.csv")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    print(dataset.nodes)
    for dl in loader:
        print(dl)
        print()
        input()
