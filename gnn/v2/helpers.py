import os
import ast
import json

import torch
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset


class CVFConfigDataset(Dataset):
    def __init__(
        self, program, dataset_file, A_file, num_classes, one_hot_encode=True
    ) -> None:
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "v2", "datasets", program
        )
        self.data = pd.read_csv(os.path.join(dataset_dir, dataset_file))
        # self.edge_index = torch.tensor(
        #     json.load(open(os.path.join(dataset_dir, edge_index_file), "r")),
        #     dtype=torch.long,
        # )
        self.A = torch.tensor(
            json.load(open(os.path.join(dataset_dir, A_file), "r")),
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
                    [[i for i in ast.literal_eval(row["config"])]], dtype=torch.float32
                ),
                torch.tensor([[row["M"]]]).float(),
            )

        return result
