import os
import sys
import ast
import json

import torch
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.utils import to_dense_adj

sys.path.append(os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "v2"))

from cvf_fa_helpers import get_graph
from graph_coloring import GraphColoringCVFAnalysisV2


class CVFConfigDataset(Dataset):
    def __init__(self, dataset_file, edge_index_file, num_classes) -> None:
        self.data = pd.read_csv(os.path.join("datasets", dataset_file))
        # self.nodes = len(self[0][0])
        self.edge_index = torch.tensor(
            json.load(open(os.path.join("datasets", edge_index_file), "r")),
            dtype=torch.long,
        )
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]

        result = (
            F.one_hot(
                torch.tensor(ast.literal_eval(row["config"])),
                num_classes=self.num_classes,
            ).to(torch.float32),
            row["rank"],
        )

        # result = (
        #     torch.tensor([[i] for i in ast.literal_eval(row["config"])], dtype=torch.float32),
        #     row["rank"],
        # )

        return result


class CVFConfigForGCNDataset(Dataset):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]

        result = (
            torch.FloatTensor([[i] for i in ast.literal_eval(row["config"])]).to(
                self.device
            ),
            torch.FloatTensor([row["rank"]]).to(self.device),
        )

        return result


class CVFConfigForGCNWSuccP1Dataset(Dataset):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = [i for i in ast.literal_eval(row["config"])]
        succ = [i for i in ast.literal_eval(row["succ"])]
        if succ:
            succ = torch.FloatTensor(succ).to(self.device)
            succ = torch.mean(succ, dim=0).unsqueeze(0)  # column wise
            # succ2 = torch.mean(succ, dim=1).unsqueeze(0)  # row wise
            # succ2 = torch.matmul(succ1.repeat(succ2.shape[1], 1).t(), succ2.t()).t()
        else:
            succ = torch.zeros(1, len(config)).to(self.device)

        config = torch.FloatTensor([config]).to(self.device)
        result = torch.cat((config, succ), dim=0).t(), torch.FloatTensor(
            [row["rank"]]
        ).to(self.device)

        return result


class CVFConfigForGCNWSuccODataset(Dataset):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = [i for i in ast.literal_eval(row["config"])]
        succ = [i for i in ast.literal_eval(row["succ"])]
        if succ:
            succ = torch.FloatTensor(succ).to(self.device)
            succ1 = torch.mean(succ, dim=0).unsqueeze(0)  # column wise
            succ2 = torch.mean(succ, dim=1).unsqueeze(0)  # row wise
            succ2 = torch.matmul(succ1.repeat(succ2.shape[1], 1).t(), succ2.t()).t()
        else:
            succ1 = succ2 = torch.zeros(1, len(config)).to(self.device)

        config = torch.FloatTensor([config]).to(self.device)
        result = torch.cat((config, succ1, succ2), dim=0).t(), torch.FloatTensor(
            [row["rank"]]
        ).to(self.device)

        return result


class CVFConfigForGCNWSuccDataset(Dataset):
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
        self.dataset_name = dataset_file.split("_config_rank_dataset.csv")[0]
        self.edge_index = (
            torch.LongTensor(
                json.load(open(os.path.join(dataset_dir, edge_index_file), "r")),
            )
            .t()
            .to(self.device)
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = [i for i in ast.literal_eval(row["config"])]
        succ = [i for i in ast.literal_eval(row["succ"])]
        if succ:
            succ = torch.FloatTensor(succ).to(self.device)
            # succ1 = torch.rand(1, succ.shape[1]).to(self.device)
            # succ2 = torch.rand(1, succ.shape[1]).to(self.device)
            succ1 = torch.mean(succ, dim=0).unsqueeze(0)  # column wise
            succ2 = torch.mean(succ, dim=1)  # row wise
            succ2 = torch.sum(succ2).repeat(succ1.shape)
            # succ3 = torch.FloatTensor([succ.shape[0]]).to(self.device).repeat(succ1.shape)
        else:
            succ1 = torch.zeros(1, len(config)).to(self.device)
            succ2 = succ1.clone()

        config = torch.FloatTensor([config]).to(self.device)
        result = torch.cat((config, succ1, succ2), dim=0).t(), torch.FloatTensor(
            [row["rank"]]
        ).to(self.device)

        return result

    def __repr__(self):
        return f"{self.__class__.__name__} {self.dataset_name}"


class CVFConfigForGCNWSuccWEIDataset(Dataset):
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
        self.dataset_name = dataset_file.split("_config_rank_dataset.csv")[0]
        self.edge_index = (
            torch.LongTensor(
                json.load(open(os.path.join(dataset_dir, edge_index_file), "r")),
            )
            .t()
            .to(self.device)
        )
        self.A = to_dense_adj(self.edge_index).squeeze(0)
        self.D = 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = [i for i in ast.literal_eval(row["config"])]
        succ = [i for i in ast.literal_eval(row["succ"])]
        if succ:
            succ = torch.FloatTensor(succ).to(self.device)
            succ1 = torch.mean(succ, dim=0).unsqueeze(0)  # column wise
            succ2 = torch.mean(succ, dim=1)  # row wise
            succ2 = torch.sum(succ2).repeat(succ1.shape)
        else:
            succ1 = torch.zeros(1, len(config)).to(self.device)
            succ2 = succ1.clone()

        config = torch.FloatTensor([config]).to(self.device)
        result = (
            torch.cat((config, succ1, succ2), dim=0).t(),
            self.A,
            self.dataset_name,
        ), torch.FloatTensor([row["rank"]]).to(self.device)

        # result = (
        #     config.t(),
        #     self.A,
        #     self.dataset_name
        # ), torch.FloatTensor([row["rank"]]).to(self.device)

        return result

    def __repr__(self):
        return f"{self.__class__.__name__} {self.dataset_name}"


class CVFConfigForGCNWSuccLSTMDataset(Dataset):
    def __init__(
        self,
        device,
        dataset_file,
        edge_index_file=None,
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
        self.dataset_name = dataset_file.split("_config_rank_dataset.csv")[0]
        self.D = 3  # input dimension

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = [i for i in ast.literal_eval(row["config"])]
        succ = [i for i in ast.literal_eval(row["succ"])]
        if succ:
            succ = torch.FloatTensor(succ).to(self.device)
            succ1 = torch.mean(succ, dim=0).unsqueeze(0)  # column wise
            succ2 = torch.mean(succ, dim=1)  # row wise
            succ2 = torch.sum(succ2).repeat(succ1.shape)
        else:
            succ1 = torch.zeros(1, len(config)).to(self.device)
            succ2 = succ1.clone()

        config = torch.FloatTensor([config]).to(self.device)
        result = (
            torch.cat((config, succ1, succ2), dim=0).t(),
            self.dataset_name,
        ), torch.FloatTensor([row["rank"]]).to(self.device)

        return result

    def __repr__(self):
        return f"{self.__class__.__name__} {self.dataset_name}"


class CVFConfigForAnalysisDataset(Dataset):
    def __init__(
        self,
        device,
        graph_name,
        program="coloring",
    ) -> None:
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
        self.cache = {}

    def __len__(self):
        return self.cvf_analysis.total_configs

    def _get_succ_encoding(self, idx, config):
        program_transition_idxs = self.cvf_analysis._get_program_transitions(config)
        self.cache[idx] = program_transition_idxs
        succ = [self.cvf_analysis.indx_to_config(i) for i in program_transition_idxs]
        if succ:
            succ = torch.FloatTensor(succ).to(self.device)
            succ1 = torch.mean(succ, dim=0).unsqueeze(0)  # column wise
            succ2 = torch.mean(succ, dim=1)  # row wise
            succ2 = torch.sum(succ2).repeat(succ1.shape)
        else:
            succ1 = torch.zeros(1, len(config)).to(self.device)
            succ2 = succ1.clone()

        return succ1, succ2

    def __getitem__(self, idx):
        config = self.cvf_analysis.indx_to_config(idx)
        succ1, succ2 = self._get_succ_encoding(idx, config)
        config = torch.FloatTensor([config]).to(self.device)
        result = (torch.cat((config, succ1, succ2), dim=0).t(), idx)
        return result

    def get_pts(self, idx):
        if idx in self.cache:
            result = self.cache[idx]
            del self.cache[idx]
            return result
        config = self.cvf_analysis.indx_to_config(idx)
        program_transition_idxs = self.cvf_analysis._get_program_transitions(config)
        return program_transition_idxs


class CVFConfigForGCNWSuccFDataset(Dataset):
    def __init__(
        self,
        device,
        dataset_file,
        edge_index_file,
        adjacency_file,
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
        self.A = torch.FloatTensor(
            json.load(open(os.path.join(dataset_dir, adjacency_file), "r")),
        ).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = [i for i in ast.literal_eval(row["config"])]
        succ = [i for i in ast.literal_eval(row["succ"])]
        if succ:
            succ = torch.FloatTensor(succ).to(self.device)
            succ1 = torch.mean(succ, dim=0).unsqueeze(0)  # column wise
            succ2 = torch.matmul(succ, self.A)
            succ2 = torch.mean(succ2, dim=0).unsqueeze(0)
        else:
            succ1 = succ2 = torch.zeros(1, len(config)).to(self.device)

        config = torch.FloatTensor([config]).to(self.device)
        result = torch.cat((config, succ1, succ2), dim=0).t(), torch.FloatTensor(
            [row["rank"]]
        ).to(self.device)

        return result


class CVFConfigForGCNWSuccConvDataset(Dataset):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = [i for i in ast.literal_eval(row["config"])]
        succ = [i for i in ast.literal_eval(row["succ"])]
        if succ:
            succ = torch.FloatTensor(succ).to(self.device)

            expanded_matrix = succ.unsqueeze(2) * succ.unsqueeze(1)
            row_wise_conv = expanded_matrix.sum(dim=2)

            expanded_matrix = succ.unsqueeze(0) * succ.unsqueeze(1)
            column_wise_conv = expanded_matrix.sum(dim=1)

            succ1 = torch.mean(row_wise_conv, dim=0).unsqueeze(0)  # column wise
            succ2 = torch.mean(column_wise_conv, dim=0).unsqueeze(0)  # column wise
        else:
            succ1 = succ2 = torch.zeros(1, len(config)).to(self.device)

        config = torch.FloatTensor([config]).to(self.device)
        result = torch.cat((config, succ1, succ2), dim=0).t(), torch.FloatTensor(
            [row["rank"]]
        ).to(self.device)

        return result


class CVFConfigForGCNGridSearchDataset(Dataset):
    def __init__(
        self,
        device,
        dataset_file,
        edge_index_file,
    ) -> None:
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "v2",
            "datasets",
            "coloring",
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]

        result = (
            torch.FloatTensor([[i] for i in ast.literal_eval(row["config"])]).to(
                self.device
            ),
            torch.FloatTensor([[row["rank"]]]).to(self.device),
        )

        return result


class CVFConfigForGATDataset(Dataset):
    def __init__(self, device, dataset_file, edge_index_file, num_classes) -> None:
        self.data = pd.read_csv(os.path.join("datasets", dataset_file))
        self.edge_index = (
            torch.tensor(
                json.load(open(os.path.join("datasets", edge_index_file), "r")),
                dtype=torch.long,
            )
            .t()
            .to(device)
        )
        self.num_classes = num_classes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]

        result = (
            F.one_hot(
                torch.tensor(ast.literal_eval(row["config"])),
                num_classes=self.num_classes,
            ).to(torch.float32),
            row["rank"],
        )

        return result


if __name__ == "__main__":
    # dataset = CVFConfigDataset(
    #     "graph_4_config_rank_dataset.csv", "graph_4_edge_index.json"
    # )
    device = "cpu"
    # dataset = CVFConfigForGCNDataset(
    #     device,
    #     "implicit_graph_n5_config_rank_dataset.csv",
    #     "implicit_graph_n5_edge_index.json",
    # )
    # loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # # print(dataset.nodes)
    # for dl in loader:
    #     print(dl)
    #     print()
    #     input()

    # dataset = CVFConfigForGCNWSuccDataset(
    #     device,
    #     "implicit_graph_n10_config_rank_dataset.csv",
    #     "implicit_graph_n10_edge_index.json",
    #     "dijkstra",
    # )
    # dataset = CVFConfigForGCNWSuccConvDataset(
    #     device,
    #     "implicit_graph_n5_config_rank_dataset.csv",
    #     "implicit_graph_n5_edge_index.json",
    #     "dijkstra",
    # )
    # dataset = CVFConfigForGCNWSuccFDataset(
    #     device,
    #     "implicit_graph_n10_config_rank_dataset.csv",
    #     "implicit_graph_n10_edge_index.json",
    #     "implicit_graph_n10_A.json",
    #     "dijkstra",
    # )
    # dataset = CVFConfigForGCNWSuccDataset(
    #     device,
    #     "tiny_graph_test_config_rank_w_succ_dataset.csv",
    #     "tiny_graph_edge_index.json",
    # )

    dataset = CVFConfigForAnalysisDataset("cuda", "star_graph_n7")
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for batch in loader:
        x = batch[0]
        break
