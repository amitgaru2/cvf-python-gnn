import os
import sys
import ast
import json
import random

import torch
import pandas as pd
import networkx as nx

from functools import cached_property

from sklearn.manifold import SpectralEmbedding
from torch_geometric.utils import to_dense_adj
from torch.utils.data import Dataset, DataLoader


sys.path.append(os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "v2"))

from cvf_fa_helpers import get_graph
from graph_coloring import GraphColoringCVFAnalysisV2

from custom_logger import logger


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
    def __init__(
        self, device, graph_name, pt_dataset_file, D, program="coloring"
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
        row = self.data.loc[idx].reset_index(drop=True)
        is_na = row.isna()
        if is_na.any():
            first_na_index = is_na.idxmax()
        else:
            first_na_index = len(row)  # if no na in the data
        na_mask = torch.tensor(list(is_na))
        result = torch.FloatTensor([self.cvf_analysis.indx_to_config(i) for i in row])
        result[na_mask] = -1
        attention_mask = ~na_mask
        return result, attention_mask, first_na_index


class CVFConfigForTransformerDataset(Dataset):
    def __init__(
        self,
        device,
        graph_name,
        pt_dataset_file,
        config_rank_dataset,
        D,
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
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "v2",
            "datasets",
            program,
        )
        self.data = pd.read_csv(os.path.join(dataset_dir, pt_dataset_file))
        self.cr_data = pd.read_csv(os.path.join(dataset_dir, config_rank_dataset))
        self.sequence_length = len(self.data.loc[0])
        self.D = D

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx].reset_index(drop=True)
        is_na = row.isna()
        na_mask = torch.tensor(list(is_na)).to(self.device)
        result = torch.FloatTensor(
            [self.cvf_analysis.indx_to_config(i) for i in row]
        ).to(self.device)
        result[na_mask] = -1
        attention_mask = ~na_mask
        labels = [
            self.cr_data.loc[int(i)]["rank"] if not pd.isna(i) else -1 for i in row
        ]
        return (
            (
                result,
                attention_mask,
            ),
            torch.FloatTensor(labels).to(self.device),
        )


def get_A_of_graph(graph_path):
    edges = []
    with open(graph_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            nodes = list(map(int, line.strip().split()))
            src = nodes[0]
            for dst in nodes[1:]:
                edges.append((src, dst))

    # Create undirected graph
    G = nx.Graph()
    G.add_edges_from(edges)

    # Ensure nodes are sorted for consistent adjacency matrix
    nodes = sorted(G.nodes())

    # Convert to adjacency matrix
    adj_matrix = nx.to_numpy_array(G, nodelist=nodes)

    # print("Adjacency Matrix:")
    # print(adj_matrix)
    return adj_matrix


class CVFConfigForTransformerMDataset(Dataset):
    # control tokens
    eo_sp_dim_full_value = -10

    def __init__(
        self,
        device,
        graph_name,
        pt_dataset_file,
        config_rank_dataset,
        D,
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
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "v2",
            "datasets",
            program,
        )
        self.data = pd.read_csv(os.path.join(dataset_dir, pt_dataset_file))
        self.cr_data = pd.read_csv(os.path.join(dataset_dir, config_rank_dataset))
        self.sp_emb_dim = 1
        self.prepend_size = self.sp_emb_dim
        self.sequence_length = self.prepend_size + len(self.data.loc[0])
        self.D = D
        self.A = torch.FloatTensor(get_A_of_graph(graph_path))

    @cached_property
    def spectral_embedding(self):
        embedding_model = SpectralEmbedding(
            n_components=self.sp_emb_dim, affinity="precomputed"
        )
        embedding = embedding_model.fit_transform(self.A).T
        return torch.FloatTensor(embedding).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx].reset_index(drop=True)
        is_na = row.isna()
        na_mask = torch.tensor(list(is_na))
        result = torch.FloatTensor(
            [self.cvf_analysis.indx_to_config(i) for i in row]
        ).to(self.device)
        result[na_mask] = -1
        result = torch.cat(
            [
                self.spectral_embedding,
                result,
            ]
        )  # add the graph info here at indx 0
        padding_mask = torch.cat(
            [
                torch.Tensor([False for _ in range(self.prepend_size)]),
                na_mask,
            ]
        ).to(
            self.device
        )  # padding  mask for the graph info at indx 0
        labels = [
            -1 for _ in range(self.prepend_size)
        ]  # spectral dimension + separator
        labels.extend(
            [self.cr_data.loc[int(i)]["rank"] if not pd.isna(i) else -1 for i in row]
        )
        # label = torch.FloatTensor([self.cr_data.loc[row[0]]["rank"]]).to(self.device)
        return (
            (
                result,
                padding_mask,
            ),
            torch.FloatTensor(labels).to(self.device),
        )
    

class CVFConfigForTransformerDecoderDataset(Dataset):

    def __init__(
        self,
        device,
        graph_name,
        pt_dataset_file,
        config_rank_dataset,
        D,
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
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "v2",
            "datasets",
            program,
        )
        self.data = pd.read_csv(os.path.join(dataset_dir, pt_dataset_file))
        self.cr_data = pd.read_csv(os.path.join(dataset_dir, config_rank_dataset))
        self.sp_emb_dim = 1
        self.prepend_size = self.sp_emb_dim
        self.sequence_length = self.prepend_size + len(self.data.loc[0])
        self.D = D
        self.A = torch.FloatTensor(get_A_of_graph(graph_path))

    @cached_property
    def spectral_embedding(self):
        embedding_model = SpectralEmbedding(
            n_components=self.sp_emb_dim, affinity="precomputed"
        )
        embedding = embedding_model.fit_transform(self.A).T
        return torch.FloatTensor(embedding).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx].reset_index(drop=True)
        is_na = row.isna()
        len_row_non_na = len(row.dropna())
        na_mask = torch.tensor(list(is_na))
        result = torch.FloatTensor(
            [self.cvf_analysis.indx_to_config(i) for i in row]
        ).to(self.device)
        result[na_mask] = -1
        result = torch.cat(
            [
                self.spectral_embedding,
                result,
            ]
        )  # add the graph info here at indx 0
        padding_mask = torch.cat(
            [
                torch.Tensor([False for _ in range(self.prepend_size)]),
                na_mask,
            ]
        ).to(
            self.device
        )  # padding  mask for the graph info at indx 0
        labels = [
            1 for _ in range(self.prepend_size)
        ]  # 1 for valid sequence, 0 for invalid
        labels.extend(
            [len_row_non_na - (i - 1) if (i - 1) <= len_row_non_na else -1 for i in range(len(row))]
        )
        return (
            (
                result,
                padding_mask,
            ),
            torch.FloatTensor(labels).to(self.device),
        )


class CVFConfigForTransformerTestDataset(Dataset):
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
        self.data = pd.read_csv(os.path.join(dataset_dir, config_rank_dataset))
        self.D = D

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = [i for i in ast.literal_eval(row["config"])]
        config = torch.FloatTensor([config]).to(self.device)
        labels = torch.FloatTensor([row["rank"]]).to(self.device)
        return config, labels


class CVFConfigForTransformerTestDatasetWName(Dataset):
    eo_sp_dim_full_value = -10

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
        self.data = pd.read_csv(os.path.join(dataset_dir, config_rank_dataset))
        self.D = D
        self.A = torch.FloatTensor(get_A_of_graph(graph_path))
        self.sp_emb_dim = 1
        self.prefix_embedding = 1
        self.sequence_length = self.prefix_embedding + 44

    @cached_property
    def spectral_embedding(self):
        embedding_model = SpectralEmbedding(
            n_components=self.sp_emb_dim, affinity="precomputed"
        )
        embedding = embedding_model.fit_transform(self.A).T
        return torch.FloatTensor(embedding).to(self.device)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = [i for i in ast.literal_eval(row["config"])]
        config = torch.FloatTensor([config]).to(self.device)
        config = torch.cat(
            [
                self.spectral_embedding,
                config,
            ]
        )
        labels = torch.FloatTensor([row["rank"]]).to(self.device)
        return config, labels, self.dataset_name


class CVFConfigForBertFTDataset(Dataset):
    def __init__(self, device, graph_name, dataset_file, D, program="coloring") -> None:
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
        self.data = pd.read_csv(os.path.join(dataset_dir, dataset_file))
        self.sequence_length = len(self.data.loc[0])
        self.D = D

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = [i for i in ast.literal_eval(row["config"])]
        config = torch.FloatTensor([config]).to(self.device)
        labels = torch.FloatTensor([row["rank"]]).to(self.device)
        return config, labels


if __name__ == "__main__":
    device = "cuda"
    # dataset = CVFConfigForGCNWSuccLSTMDataset(
    #     device,
    #     "star_graph_n7_config_rank_dataset.csv",
    #     "star_graph_n7_edge_index.json",
    # )

    # loader = DataLoader(dataset, batch_size=2, shuffle=False)

    # dataset = CVFConfigForBertDataset(
    #     device,
    #     "graph_random_regular_graph_n4_d3",
    #     "graph_random_regular_graph_n4_d3_pt_adj_list.txt",
    #     D=4,
    # )

    # dataset = CVFConfigForBertDataset(
    #     device,
    #     "implicit_graph_n5",
    #     "implicit_graph_n5_pt_adj_list.txt",
    #     D=5,
    #     program="dijkstra",
    # )

    # dataset = CVFConfigForBertFTDataset(
    #     device,
    #     "implicit_graph_n5",
    #     "implicit_graph_n5_config_rank_dataset.csv",
    #     D=5,
    #     program="dijkstra",
    # )

    dataset = CVFConfigForTransformerDecoderDataset(
        device,
        "implicit_graph_n5",
        "implicit_graph_n5_pt_adj_list.txt",
        "implicit_graph_n5_config_rank_dataset.csv",
        D=5,
        program="dijkstra",
    )

    print(dataset.sequence_length)

    # dataset = CVFConfigForTransformerTestDatasetWName(
    #     device,
    #     "implicit_graph_n5",
    #     "implicit_graph_n5_config_rank_dataset.csv",
    #     D=5,
    #     program="dijkstra",
    # )

    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for batch in loader:
        # x = batch[0]
        print(batch[0])
        print(batch[1])
        # print(batch[2])
        break
