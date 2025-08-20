import os
import ast
import json

import torch
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader



class CVFConfigForGCNWSuccWEIDataset(Dataset):
    def __init__(
        self,
        device,
        dataset_file,
        edge_index_file,
        program,
    ) -> None:
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "datasets",
            program,
        )
        self.data = pd.read_csv(os.path.join(dataset_dir, dataset_file))
        self.device = device
        self.dataset_name = dataset_file.split("_config_rank_dataset.csv")[0]
        self.edge_index = (
            torch.LongTensor(
                json.load(open(os.path.join(dataset_dir, edge_index_file), "r")),
            )
            .t()
            .to(self.device)
        )
        self.D = 1

    def get_encoded_config(self, config):
        return torch.FloatTensor([[i[0]] for i in config])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = ast.literal_eval(row["config"])
        config = self.get_encoded_config(config).to(self.device)
        result = (
            config,
            self.edge_index,
            self.dataset_name,
        ), torch.FloatTensor(
            [row["rank"] == 0]
        ).to(self.device)

        return result

    def __repr__(self):
        return f"{self.__class__.__name__} {self.dataset_name}"


if __name__ == "__main__":
    device = "cpu"
    graph_name = "complete_graph_n5"
    dataset_file = f"{graph_name}_config_rank_dataset.csv"
    edge_index_file = f"{graph_name}_edge_index.json"
    dataset = CVFConfigForGCNWSuccWEIDataset(
        device, dataset_file, edge_index_file, program="graph_coloring"
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch in loader:
        x = batch[0]
        y = batch[1]
        print(y)
        break
