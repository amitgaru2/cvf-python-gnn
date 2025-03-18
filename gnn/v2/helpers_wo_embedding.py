import os
import ast
import json

import torch
import pandas as pd

from torch.utils.data import Dataset


class CVFConfigDataset(Dataset):
    def __init__(self, program, dataset_file, A_file, num_nodes) -> None:
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "v2", "datasets", program
        )
        self.data = pd.read_csv(os.path.join(dataset_dir, dataset_file))
        self.A = torch.tensor(
            json.load(open(os.path.join(dataset_dir, A_file), "r")),
            dtype=torch.long,
        )
        self.num_nodes = num_nodes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]

        result = (
            torch.tensor(
                [[i for i in ast.literal_eval(row["config"])]], dtype=torch.float32
            ),
            torch.tensor([[row["M"]]]).float(),
        )

        return result
