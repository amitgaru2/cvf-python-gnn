import os
import ast
import sys
import torch
import pandas as pd
import networkx as nx


from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis"))

from cvf_fa_helpers import get_graph


def get_A_of_graph(graph_path):
    edges = []
    with open(graph_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            nodes = list(map(int, line.strip().split()))
            src = nodes[0]
            for dst in nodes[1:]:
                edges.append((src, dst))

    G = nx.Graph()
    G.add_edges_from(edges)
    nodes = sorted(G.nodes())
    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)
    return adj_matrix


class MessagePassingDataset(Dataset):
    def __init__(
        self,
        device,
        graph_name,
        config_rank_dataset,
        D,
        program="coloring",
    ) -> None:
        graphs_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "graphs"
        )
        graph_path = os.path.join(graphs_dir, f"{graph_name}.txt")
        # graph = get_graph(graph_path)
        self.device = device
        self.dataset_name = graph_name
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "datasets",
            program,
        )
        self.data = pd.read_csv(os.path.join(dataset_dir, config_rank_dataset))
        self.D = D
        self.A = torch.FloatTensor(get_A_of_graph(graph_path))
        self.edge_index = (
            self.A.nonzero(as_tuple=False).t().contiguous().to(self.device)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = [i for i in ast.literal_eval(row["config"])]
        succs = ast.literal_eval(row["succ"])
        if succs:
            temp = []
            for s in succs:
                if s is None:
                    temp.append([-1 for _ in range(len(config))])
                else:
                    temp.append(s)
            succs = torch.FloatTensor(temp)
        else:
            succs = torch.full((len(config), len(config)), -1, dtype=torch.float32)
        config = torch.FloatTensor([config]).T
        labels = torch.FloatTensor([row["rank"]]).to(self.device)
        return config.to(self.device), succs.to(self.device), labels


if __name__ == "__main__":
    device = "cuda"
    dataset = MessagePassingDataset(
        device, "tiny_graph_test", "tiny_graph_test_config_rank_dataset.csv", 3
    )
    dataset = MessagePassingDataset(
        device,
        "graph_random_regular_graph_n7_d4",
        "graph_random_regular_graph_n7_d4_config_rank_dataset.csv",
        D=7,
    )
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    for batch in loader:
        print(batch[0])
        print(batch[1])
        # print(batch[2])
        break
