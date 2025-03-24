import os
import ast
import json

import torch
import pandas as pd

from torch.utils.data import Dataset


class CVFConfigDataset(Dataset):
    embedding_map = torch.tensor(
        [
            [-0.5290, -0.0216, -0.0945, 0.2270],
            [-0.4443, -0.0111, -0.1742, -0.2307],
            [-0.3748, -0.0128, -0.2559, -0.1789],
            [-0.5496, -0.0217, -0.1014, 0.2230],
            [-0.4470, -0.0077, 0.0267, -0.1808],
            [-0.5436, -0.0245, -0.3175, 0.1133],
            [-0.4278, -0.0222, -0.3070, 0.1349],
            [-0.4484, -0.0081, 0.0516, -0.1719],
            [-0.4758, -0.0172, -0.1159, 0.0017],
            [-0.4668, -0.0147, 0.0071, 0.0639],
            [-0.5965, -0.0140, 0.0095, 0.0161],
            [-0.5079, -0.0156, -0.1128, 0.0029],
        ]
    )

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

        # result = (
        #     torch.tensor(
        #         [[i for i in ast.literal_eval(row["config"])]], dtype=torch.float32
        #     ),
        #     torch.tensor([[row["M"]]]).float(),
        # )

        result = (
            self.embedding_map[row["config"]].unsqueeze(0),
            torch.FloatTensor([[row["M"]]]),
        )

        return result


if __name__ == "__main__":
    dataset = CVFConfigDataset(
        "coloring",
        "tiny_graph_test_config_rank_dataset.csv",
        "tiny_graph_test_A.json",
        3,
    )

    print(dataset[0], dataset[0][0].shape)
