import os
import ast
import json

import torch
import pandas as pd

from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset, Data


class CVFGATGeometricDataset(Dataset):
    def __init__(
        self,
        device,
        dataset,
        program="graph_coloring",
    ) -> None:
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "datasets", program
        )
        edge_index_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "graphs", "edge_indexes"
        )
        self.data = pd.read_csv(
            os.path.join(dataset_dir, f"{dataset}_config_rank_dataset.csv")
        )
        self.device = device
        self.edge_index = (
            torch.LongTensor(
                json.load(
                    open(
                        os.path.join(edge_index_dir, f"{dataset}_edge_index.json"), "r"
                    )
                )
            )
            .t()
            .to(self.device)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = ast.literal_eval(row["config"])
        config = torch.FloatTensor(config).to(self.device)
        y = torch.FloatTensor([row["rank"]]).to(self.device)
        data = Data(x=config, edge_index=self.edge_index, y=y)
        return data


def main():
    device = "cuda"
    dataset = CVFGATGeometricDataset(device, "star_graph_n4")
    print(dataset[0].num_nodes)
    loader = DataLoader(dataset, batch_size=1)
    for batch in loader:
        print(batch.x)
        break


if __name__ == "__main__":
    main()
