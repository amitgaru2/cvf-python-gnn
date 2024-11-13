import ast
import pandas as pd

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class CVFConfigDataset(Dataset):
    def __init__(self) -> None:
        self.data = pd.read_csv("small_graph_test_config_rank_dataset.csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        return ast.literal_eval(row["config"]), row["rank"]


if __name__ == "__main__":
    dataset = CVFConfigDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for dl in loader:
        print(dl)
        print()
