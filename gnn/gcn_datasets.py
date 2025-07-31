import os
import ast
import sys
import json

import torch
import pandas as pd
import torch.nn.functional as F


from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis"))
sys.path.append(os.path.join(os.getenv("CVF_PROJECT_DIR"), "utils"))

from command_line_helpers import ColoringProgram, DijkstraProgram


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
        self.no_features = (
            10 if program == ColoringProgram else 3
        )  # graph coloring: highest color value based on degree of nodes, dijkstra: 3

    def get_encoded_config(self, config):
        result = []
        for v in config:
            val = torch.LongTensor([v])
            result.append(
                F.one_hot(val, num_classes=self.no_features)
                .squeeze()
                .type(torch.float32)
            )
        return torch.stack(result)

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
            [row["rank"]]
        ).to(self.device)

        return result

    def __repr__(self):
        return f"{self.__class__.__name__} {self.dataset_name}"


if __name__ == "__main__":
    device = "cpu"
    graph_name = "star_graph_n4"
    dataset_file = f"{graph_name}_config_rank_dataset.csv"
    edge_index_file = f"{graph_name}_edge_index.json"
    dataset = CVFConfigForGCNWSuccWEIDataset(
        device, dataset_file, edge_index_file, program="graph_coloring"
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in loader:
        x = batch[0]
        y = batch[1]
        print(x)
        break
