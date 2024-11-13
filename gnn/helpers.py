import pandas as pd

from torch.utils.data import Dataset


class CVFConfigDataset(Dataset):
    def __init__(self) -> None:
        self.data = pd.read_csv()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass
