import json
import os
import sys
import ast
import random

import torch
import pandas as pd
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj
from torch.utils.data import Dataset, DataLoader


sys.path.append(os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "v2"))

from cvf_fa_helpers import get_graph
from graph_coloring import GraphColoringCVFAnalysisV2


class CVFConfigForGCNWSuccLSTMDataset(Dataset):
    def __init__(
        self,
        device,
        dataset_file,
        edge_index_file,
        program="coloring",
    ) -> None:
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "v2",
            "datasets",
            program,
        )
        self.data = pd.read_csv(os.path.join(dataset_dir, dataset_file))
        self.device = device
        self.edge_index = (
            torch.LongTensor(
                json.load(open(os.path.join(dataset_dir, edge_index_file), "r")),
            )
            .t()
            .to(self.device)
        )
        self.A = to_dense_adj(self.edge_index).squeeze(0)
        self.dataset_name = dataset_file.split("_config_rank_dataset.csv")[0]
        self.D = 7  # input dimension
        self.total_succ_select = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = [i for i in ast.literal_eval(row["config"])]
        succ = [i for i in ast.literal_eval(row["succ"])]
        if succ:
            if len(succ) >= self.total_succ_select:
                indxes = random.sample(range(len(succ)), self.total_succ_select)
                # indxes = list(range(self.total_succ_select))
            else:
                indxes = list(range(len(succ)))
            remaining_succ_count = self.total_succ_select - len(indxes)
            selected_succ = [succ[i] for i in indxes]
            for _ in range(remaining_succ_count):
                selected_succ.append([-1 for _ in range(self.D)])
            succ = torch.FloatTensor(selected_succ).to(self.device)
        else:
            # succ = torch.zeros(self.total_succ_select, len(config)).to(self.device)
            succ = torch.full((self.total_succ_select, self.D), -1).to(self.device)
        config = torch.FloatTensor([config]).to(self.device)
        labels = torch.FloatTensor([row["rank"]]).to(self.device)
        result = (
            torch.cat((self.A, config, succ)),
            self.dataset_name,
        ), labels
        # result = config, labels
        return result

    def __repr__(self):
        return f"{self.__class__.__name__} {self.dataset_name}"


class CVFConfigForBertDataset(Dataset):
    def __init__(self, device, graph_name, pt_dataset_file, D, program="coloring") -> None:
        graphs_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "graphs"
        )
        graph_path = os.path.join(graphs_dir, f"{graph_name}.txt")
        graph = get_graph(graph_path)
        self.cvf_analysis = GraphColoringCVFAnalysisV2(
            graph_name,
            graph,
            generate_data_ml=False,
            generate_data_embedding=False,
            generate_test_data_ml=True,
        )

        self.device = device
        self.dataset_name = graph_name
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "v2",
            "datasets",
            program,
        )
        self.data = pd.read_csv(os.path.join(dataset_dir, pt_dataset_file))
        self.sequence_length = len(self.data.loc[0])
        self.D = D


    def vocab_tensor(self):
        result = []
        for i in range(self.cvf_analysis.total_configs):
            result.append(self.cvf_analysis.indx_to_config(i))
        result = torch.FloatTensor(result)
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        na_mask = torch.tensor(list(row.isna()))
        result = torch.FloatTensor([self.cvf_analysis.indx_to_config(i) for i in row])
        result[na_mask] = -1
        attention_mask = ~na_mask
        return result, attention_mask


if __name__ == "__main__":
    device = "cuda"
    # dataset = CVFConfigForGCNWSuccLSTMDataset(
    #     device,
    #     "star_graph_n7_config_rank_dataset.csv",
    #     "star_graph_n7_edge_index.json",
    # )

    # loader = DataLoader(dataset, batch_size=2, shuffle=False)

    dataset = CVFConfigForBertDataset(
        device,
        "graph_random_regular_graph_n4_d3",
        "graph_random_regular_graph_n4_d3_pt_adj_list.txt",
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in loader:
        # x = batch[0]
        print(batch[0])
        print(batch[1])
        break
