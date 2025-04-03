import os
import ast
import json
import pickle

import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader


class CVFConfigDataset(Dataset):
    embedding_map = torch.FloatTensor(pickle.load(open("embedding.dump", "rb")))

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
        "graph_random_regular_graph_n4_d3_config_rank_dataset.csv",
        "graph_random_regular_graph_n4_d3_A.json",
        4,
    )

    for data in DataLoader(dataset):
        print(data)
    # print(dataset[0], dataset[0][0].shape)
