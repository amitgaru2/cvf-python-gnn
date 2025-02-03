import os
import ast
import json

import torch
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset


class CVFConfigDataset(Dataset):
    def __init__(
        self, dataset_file, edge_index_file, num_classes, one_hot_encode=True
    ) -> None:
        self.data = pd.read_csv(os.path.join("datasets", dataset_file))
        self.edge_index = torch.tensor(
            json.load(open(os.path.join("datasets", edge_index_file), "r")),
            dtype=torch.long,
        )
        self.num_classes = num_classes
        self.one_hot_encode = one_hot_encode
        self.nodes = len(self[0][0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]

        if self.one_hot_encode:
            result = (
                F.one_hot(
                    torch.tensor(ast.literal_eval(row["config"])),
                    num_classes=self.num_classes,
                ).to(torch.float32),
                row["rank"],
            )

        else:
            result = (
                torch.tensor(
                    [i for i in ast.literal_eval(row["config"])], dtype=torch.float32
                ),
                torch.tensor([row["rank"]]).float(),
            )

        return result
