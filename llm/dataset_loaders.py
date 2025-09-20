import os
import sys
import json

import torch


from torch.utils.data import Dataset, DataLoader


sys.path.append(os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis"))

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)

from custom_logger import logger
from cvf_fa_helpers import get_graph
from dijkstra import DijkstraTokenRingCVFAnalysisV2
from graph_coloring import GraphColoringCVFAnalysisV2
from maximal_matching import MaximalMatchingCVFAnalysisV2

device = "cuda"


class MPNNDataset(Dataset):
    def __init__(
        self,
        device,
        graph_name,
        program="graph_coloring",
    ) -> None:
        edge_index_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "graphs", "edge_indexes"
        )
        self.device = device
        self.dataset_name = graph_name
        self.edge_index = (
            torch.LongTensor(
                json.load(
                    open(
                        os.path.join(edge_index_dir, f"{graph_name}_edge_index.json"),
                        "r",
                    )
                ),
            )
            .t()
            .to(self.device)
        )
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "datasets",
            program,
        )
        self.data = torch.load(os.path.join(dataset_dir, f"ml_mpnn__{graph_name}.pt"))
        # self.data = {}
        # self.data["X"] = torch.cat(X_chunks, dim=0)
        # self.data["y"] = torch.cat(y_chunks, dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data["X"][idx].unsqueeze(dim=-1).to(self.device), self.data["y"][
            idx
        ].unsqueeze(dim=-1).to(self.device)


if __name__ == "__main__":
    device = "cpu"

    dataset = MPNNDataset(
        device,
        "graph_random_regular_graph_n7_d4",
        program="graph_coloring",
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in loader:
        print(batch[0])
        break
