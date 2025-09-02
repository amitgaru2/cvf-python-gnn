import os
import sys
import ast
import json

import torch
import pandas as pd
import torch.nn.functional as F

from functools import wraps, lru_cache

from torch.utils.data import Dataset, DataLoader

from custom_logger import logger

sys.path.append(os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis"))

from cvf_fa_helpers import get_graph
from graph_coloring import GraphColoringCVFAnalysisV2
from dijkstra import DijkstraTokenRingCVFAnalysisV2
from maximal_matching import MaximalMatchingCVFAnalysisV2

device = "cuda"

LSTM_PAD_UPTO_LENGTH = 15
LSTM_PAD_VALUE = -1


def profile_peak_gpu_memory(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.reset_peak_memory_stats(device)

        start_mem = torch.cuda.memory_allocated(device)
        result = func(*args, **kwargs)
        end_mem = torch.cuda.memory_allocated(device)
        peak_mem = torch.cuda.max_memory_allocated(device)

        logger.info(f"[{func.__name__}] GPU {device} memory usage:")
        logger.info(f"  Start : {start_mem / 1024**2:.2f} MB")
        logger.info(f"  End   : {end_mem / 1024**2:.2f} MB")
        logger.info(f"  Peak  : {peak_mem / 1024**2:.2f} MB")

        return result

    return wrapper


def mean_relative_error(y_pred, y_true):
    """
    Mean Relative Error (MRE)

    Args:
        y_pred (Tensor): Predicted values
        y_true (Tensor): True values
        eps (float): Small constant to avoid division by zero

    Returns:
        Tensor: scalar MRE
    """
    denom = torch.abs(y_true)
    if torch.all(denom == 0):
        return torch.FloatTensor([0.0])
    return torch.mean(torch.abs(y_true - y_pred) / denom)


class CVFConfigForGCNWSuccWEIDataset(Dataset):
    def __init__(
        self,
        device,
        dataset_file,
        edge_index_file,
        program="graph_coloring",
    ) -> None:
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "datasets",
            program,
        )
        edge_index_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "graphs", "edge_indexes"
        )
        self.data = pd.read_csv(os.path.join(dataset_dir, dataset_file))
        self.device = device
        self.dataset_name = dataset_file.split("_config_rank_dataset.csv")[0]
        self.edge_index = (
            torch.LongTensor(
                json.load(open(os.path.join(edge_index_dir, edge_index_file), "r")),
            )
            .t()
            .to(self.device)
        )
        # self.A = to_dense_adj(self.edge_index).squeeze(0)
        self.D = 2

    def __len__(self):
        return len(self.data)

    def get_encoded_config(self, config):
        return [i[0] for i in config]

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = self.get_encoded_config(ast.literal_eval(row["config"]))
        succ = [self.get_encoded_config(s) for s in ast.literal_eval(row["succ"])]
        if succ:
            succ = torch.FloatTensor(succ).to(self.device)
            succ1 = torch.mean(succ, dim=0).unsqueeze(0)  # column wise
        else:
            succ1 = torch.zeros(1, len(config)).to(self.device)

        config = torch.FloatTensor([config]).to(self.device)
        result = (
            torch.cat((config, succ1), dim=0).t(),
            self.edge_index,
            self.dataset_name,
        ), torch.FloatTensor([row["rank"]]).to(self.device)

        return result

    def __repr__(self):
        return f"{self.__class__.__name__} {self.dataset_name}"


class CVFConfigForGCNWSuccWEIDatasetForMM(Dataset):
    """only for Max matching"""

    def __init__(
        self,
        device,
        dataset_file,
        edge_index_file,
        program="maximal_matching",
    ) -> None:
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "datasets",
            program,
        )
        edge_index_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "graphs", "edge_indexes"
        )
        self.data = pd.read_csv(os.path.join(dataset_dir, dataset_file))
        self.device = device
        self.dataset_name = dataset_file.split("_config_rank_dataset.csv")[0]
        self.edge_index = (
            torch.LongTensor(
                json.load(open(os.path.join(edge_index_dir, edge_index_file), "r")),
            )
            .t()
            .to(self.device)
        )
        # self.A = to_dense_adj(self.edge_index).squeeze(0)
        self.highest_p_value = 15
        self.D = 2

    def __len__(self):
        return len(self.data)

    def get_p_encoding(self, p_value):
        if p_value is None:
            p_value = self.highest_p_value + 1

        p_value = torch.LongTensor([p_value])
        return (
            F.one_hot(p_value, num_classes=self.highest_p_value + 2)
            .squeeze()
            .type(torch.float32)
        )

    def get_m_encoding(self, m_value):
        return (torch.LongTensor([1]) if m_value else torch.LongTensor([0])).type(
            torch.float32
        )

    @lru_cache(maxsize=None)
    def get_p_m_encoding(self, p_value, m_value):
        return torch.cat([self.get_p_encoding(p_value), self.get_m_encoding(m_value)])

    def get_encoded_config(self, config):
        return torch.stack([self.get_p_m_encoding(v[0], v[1]) for v in config])

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        succ = [i for i in ast.literal_eval(row["succ"])]
        config = self.get_encoded_config(ast.literal_eval(row["config"])).to(
            self.device
        )
        if succ:
            _succ = [self.get_encoded_config(s) for s in succ]
            succ = torch.stack(_succ).to(self.device)
            succ1 = torch.mean(succ, dim=0)
        else:
            succ1 = torch.zeros(config.shape[0], config.shape[1]).to(self.device)

        result = (
            torch.stack([config, succ1]).reshape(self.D, -1).t(),
            self.edge_index,
            self.dataset_name,
        ), torch.FloatTensor([row["rank"]]).to(self.device)

        return result

    def __repr__(self):
        return f"{self.__class__.__name__} {self.dataset_name}"


class CVFConfigForGCNWSuccLSTMDataset(Dataset):
    def __init__(
        self,
        device,
        dataset,
        program="graph_coloring",
    ) -> None:
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "datasets",
            program,
        )
        self.data = pd.read_csv(
            os.path.join(dataset_dir, f"{dataset}_config_rank_dataset.csv")
        )
        self.device = device
        self.dataset_name = dataset
        self.D = 2  # input dimension

    def __len__(self):
        return len(self.data)

    def get_encoded_config(self, config):
        return [i[0] for i in config]

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = self.get_encoded_config(ast.literal_eval(row["config"]))
        succ = [self.get_encoded_config(s) for s in ast.literal_eval(row["succ"])]
        if succ:
            succ = torch.FloatTensor(succ).to(self.device)
            succ1 = torch.mean(succ, dim=0).unsqueeze(0)  # column wise
        else:
            succ1 = torch.zeros(1, len(config)).to(self.device)

        config = torch.FloatTensor([config]).to(self.device)
        # padding
        X_wo_pad = torch.cat((config, succ1), dim=0)
        pad_length = LSTM_PAD_UPTO_LENGTH - X_wo_pad.shape[1]
        X_w_pad = F.pad(X_wo_pad, (0, pad_length), value=LSTM_PAD_VALUE)
        #
        result = (X_w_pad.t(), self.dataset_name), torch.FloatTensor([row["rank"]]).to(
            self.device
        )

        return result

    def __repr__(self):
        return f"{self.__class__.__name__} {self.dataset_name}"


class CVFConfigForGCNWSuccLSTMDatasetForMM(Dataset):
    """only for mm"""

    def __init__(self, device, dataset, program="maximal_matching") -> None:
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "datasets",
            program,
        )
        self.data = pd.read_csv(
            os.path.join(dataset_dir, f"{dataset}_config_rank_dataset.csv")
        )
        self.device = device
        self.dataset_name = dataset
        self.D = 2  # input dimension
        self.highest_p_value = 15

    def __len__(self):
        return len(self.data)

    def get_p_encoding(self, p_value):
        if p_value is None:
            p_value = self.highest_p_value + 1

        p_value = torch.LongTensor([p_value])
        return (
            F.one_hot(p_value, num_classes=self.highest_p_value + 2)
            .squeeze()
            .type(torch.float32)
        )

    def get_m_encoding(self, m_value):
        return (torch.LongTensor([1]) if m_value else torch.LongTensor([0])).type(
            torch.float32
        )

    @lru_cache(maxsize=None)
    def get_p_m_encoding(self, p_value, m_value):
        return torch.cat([self.get_p_encoding(p_value), self.get_m_encoding(m_value)])

    def get_encoded_config(self, config):
        return torch.stack([self.get_p_m_encoding(v[0], v[1]) for v in config])

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        succ = [i for i in ast.literal_eval(row["succ"])]
        config = self.get_encoded_config(ast.literal_eval(row["config"])).to(
            self.device
        )
        if succ:
            _succ = [self.get_encoded_config(s) for s in succ]
            succ = torch.stack(_succ).to(self.device)
            succ1 = torch.mean(succ, dim=0)
            # succ2 = torch.sum(torch.mean(succ, dim=1), dim=0)
            # succ2 = succ2.unsqueeze(0).repeat(succ1.shape[0], 1)
        else:
            succ1 = torch.zeros(config.shape[0], config.shape[1]).to(self.device)
            # succ2 = succ1.clone()

        result = (
            torch.stack([config, succ1]).reshape(self.D, -1).t(),
            self.dataset_name,
        ), torch.FloatTensor([row["rank"]]).to(self.device)
        # result = (
        #     torch.stack([config, succ1, succ2]).reshape(3, -1).t(),
        #     self.dataset_name,
        # ), torch.FloatTensor([row["rank"]]).to(self.device)

        return result

    def __repr__(self):
        return f"{self.__class__.__name__} {self.dataset_name}"


class CVFConfigForAnalysisDataset(Dataset):
    def __init__(
        self,
        device,
        graph_name,
        program="graph_coloring",
    ) -> None:
        graphs_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "graphs"
        )
        graph_path = os.path.join(graphs_dir, f"{graph_name}.txt")
        graph = get_graph(graph_path)
        program_class_map = {
            "graph_coloring": GraphColoringCVFAnalysisV2,
            "dijkstra_token_ring": DijkstraTokenRingCVFAnalysisV2,
            "maximal_matching": MaximalMatchingCVFAnalysisV2,
        }
        self.cvf_analysis = program_class_map[program](
            graph_name,
            graph,
            generate_data_ml=False,
            generate_data_embedding=False,
            generate_test_data_ml=True,
        )

        self.device = device
        self.dataset_name = graph_name
        self.cache = {}
        self.default_succ1 = torch.zeros(1, len(graph)).to(self.device)

    def __len__(self):
        return self.cvf_analysis.total_configs

    def _get_succ_encoding(self, idx, config):
        succ = list(
            i[1] for i in self.cvf_analysis._get_program_transitions_as_configs(config)
        )
        if succ:
            succ = torch.FloatTensor(succ).to(self.device)
            succ1 = torch.mean(succ, dim=0).unsqueeze(0)  # column wise
        else:
            succ1 = self.default_succ1.clone()

        return succ1

    def __getitem__(self, idx):
        config = self.cvf_analysis.indx_to_config(idx)
        succ1 = self._get_succ_encoding(idx, config)
        config = torch.FloatTensor([config]).to(self.device)
        # padding
        X_wo_pad = torch.cat((config, succ1), dim=0)
        pad_length = LSTM_PAD_UPTO_LENGTH - X_wo_pad.shape[1]
        X_w_pad = F.pad(X_wo_pad, (0, pad_length), value=LSTM_PAD_VALUE)
        #
        result = (X_w_pad.t(), idx)
        return result


class CVFConfigForAnalysisDatasetV2(Dataset):
    def __init__(
        self,
        device,
        graph_name,
        program="graph_coloring",
    ) -> None:
        graphs_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "graphs"
        )
        graph_path = os.path.join(graphs_dir, f"{graph_name}.txt")
        graph = get_graph(graph_path)
        program_class_map = {
            "graph_coloring": GraphColoringCVFAnalysisV2,
            "dijkstra_token_ring": DijkstraTokenRingCVFAnalysisV2,
            "maximal_matching": MaximalMatchingCVFAnalysisV2,
        }
        self.cvf_analysis = program_class_map[program](
            graph_name,
            graph,
            generate_data_ml=False,
            generate_data_embedding=False,
            generate_test_data_ml=True,
        )

        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "datasets",
            program,
        )

        self.data = torch.load(
            os.path.join(dataset_dir, f"{graph_name}_config_rank_dataset.pt")
        )

        self.device = device
        self.dataset_name = graph_name
        # self.default_succ1 = torch.zeros(1, len(graph)).to(self.device)

    def __len__(self):
        return self.data["y"].size(0)

    def __getitem__(self, idx):
        X = self.data["X"][idx].to(self.device)
        return (X, idx)


class CVFConfigForAnalysisDatasetMM(Dataset):
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
        program_class_map = {
            "coloring": GraphColoringCVFAnalysisV2,
            "dijkstra": DijkstraTokenRingCVFAnalysisV2,
            "maximal_matching": MaximalMatchingCVFAnalysisV2,
        }
        self.cvf_analysis = program_class_map[program](
            graph_name,
            graph,
            generate_data_ml=False,
            generate_data_embedding=False,
            generate_test_data_ml=True,
        )

        self.device = device
        self.dataset_name = graph_name
        self.default_succ1 = torch.zeros(1, len(graph)).to(self.device)
        self.highest_p_value = 15

    def __len__(self):
        return self.cvf_analysis.total_configs

    def get_p_encoding(self, p_value):
        if p_value is None:
            p_value = self.highest_p_value + 1

        p_value = torch.LongTensor([p_value])
        return (
            F.one_hot(p_value, num_classes=self.highest_p_value + 2)
            .squeeze()
            .to(torch.float32)
        )

    def get_m_encoding(self, m_value):
        return (torch.LongTensor([1]) if m_value else torch.LongTensor([0])).to(
            torch.float32
        )

    @lru_cache(maxsize=None)
    def get_p_m_encoding(self, p_value, m_value):
        return torch.cat([self.get_p_encoding(p_value), self.get_m_encoding(m_value)])

    def get_succ1_succ2(self, succ):
        succ1 = torch.mean(succ, dim=0)
        succ2 = torch.sum(torch.mean(succ, dim=1), dim=0)
        succ2 = succ2.unsqueeze(0).repeat(succ1.shape[0], 1)
        return succ1, succ2

    def get_encoded_config(self, config):
        return torch.stack(
            [
                self.get_p_m_encoding(
                    self.cvf_analysis.possible_node_values[i][v].p,
                    self.cvf_analysis.possible_node_values[i][v].m,
                )
                for i, v in enumerate(config)
            ]
        )

    def cvf_analysis_indx_to_config(self, idx):
        return self.cvf_analysis.indx_to_config(idx)

    def cvf_analysis_get_transitions_as_configs(self, config):
        return self.cvf_analysis._get_program_transitions_as_configs(config)

    def get_x(self, config, succ1, succ2):
        return torch.stack([config, succ1]).reshape(2, -1).t()

    def get_default_succs(self, config):
        succ1 = torch.zeros(config.shape[0], config.shape[1]).to(self.device)
        succ2 = succ1.clone()
        return succ1, succ2

    def move_to_device(self, tensor):
        return tensor.to(self.device)

    def __getitem__(self, idx):
        config = self.cvf_analysis_indx_to_config(idx)
        succ = [i[1] for i in self.cvf_analysis_get_transitions_as_configs(config)]
        config = self.move_to_device(self.get_encoded_config(config))

        if succ:
            _succ = [self.get_encoded_config(s) for s in succ]
            succ = self.move_to_device(torch.stack(_succ))
            succ1, succ2 = self.get_succ1_succ2(succ)
        else:
            succ1, succ2 = self.get_default_succs(config)

        result = (
            self.get_x(config, succ1, succ2),
            idx,
        )

        return result


class CVFConfigForAnalysisDatasetMMV2(Dataset):
    def __init__(
        self,
        device,
        graph_name,
        program="maximal_matching",
    ) -> None:
        graphs_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "graphs"
        )
        graph_path = os.path.join(graphs_dir, f"{graph_name}.txt")
        graph = get_graph(graph_path)
        program_class_map = {
            "coloring": GraphColoringCVFAnalysisV2,
            "dijkstra": DijkstraTokenRingCVFAnalysisV2,
            "maximal_matching": MaximalMatchingCVFAnalysisV2,
        }
        self.cvf_analysis = program_class_map[program](
            graph_name,
            graph,
            generate_data_ml=False,
            generate_data_embedding=False,
            generate_test_data_ml=True,
        )

        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "datasets",
            program,
        )

        self.data = torch.load(
            os.path.join(dataset_dir, f"{graph_name}_config_rank_dataset.pt")
        )

        self.device = device
        self.dataset_name = graph_name

    def __len__(self):
        return self.data["y"].size(0)

    def __getitem__(self, idx):
        X = self.data["X"][idx].to(self.device)
        result = (X, idx)
        return result


class CVFConfigForAnalysisDatasetForGCN(Dataset):
    def __init__(
        self,
        device,
        graph_name,
        program="graph_coloring",
    ) -> None:
        graphs_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "graphs"
        )
        graph_path = os.path.join(graphs_dir, f"{graph_name}.txt")
        graph = get_graph(graph_path)
        program_class_map = {
            "graph_coloring": GraphColoringCVFAnalysisV2,
            "dijkstra_token_ring": DijkstraTokenRingCVFAnalysisV2,
            "maximal_matching": MaximalMatchingCVFAnalysisV2,
        }
        self.cvf_analysis = program_class_map[program](
            graph_name,
            graph,
            generate_data_ml=False,
            generate_data_embedding=False,
            generate_test_data_ml=True,
        )
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
        self.cache = {}
        self.default_succ1 = torch.zeros(1, len(graph)).to(self.device)

    def __len__(self):
        return self.cvf_analysis.total_configs

    def _get_succ_encoding(self, idx, config):
        succ = list(
            i[1] for i in self.cvf_analysis._get_program_transitions_as_configs(config)
        )
        if succ:
            succ = torch.FloatTensor(succ).to(self.device)
            succ1 = torch.mean(succ, dim=0).unsqueeze(0)  # column wise
        else:
            succ1 = self.default_succ1.clone()

        return succ1

    def __getitem__(self, idx):
        config = self.cvf_analysis.indx_to_config(idx)
        succ1 = self._get_succ_encoding(idx, config)
        config = torch.FloatTensor([config]).to(self.device)
        result = (torch.cat((config, succ1), dim=0).t(), idx, self.edge_index)
        return result


class CVFConfigForAnalysisDatasetForGCNMM(Dataset):
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
        program_class_map = {
            "coloring": GraphColoringCVFAnalysisV2,
            "dijkstra": DijkstraTokenRingCVFAnalysisV2,
            "maximal_matching": MaximalMatchingCVFAnalysisV2,
        }
        self.cvf_analysis = program_class_map[program](
            graph_name,
            graph,
            generate_data_ml=False,
            generate_data_embedding=False,
            generate_test_data_ml=True,
        )

        self.device = device
        self.dataset_name = graph_name
        self.default_succ1 = torch.zeros(1, len(graph)).to(self.device)
        self.highest_p_value = 15

        edge_index_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "graphs", "edge_indexes"
        )
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

    def __len__(self):
        return self.cvf_analysis.total_configs

    def get_p_encoding(self, p_value):
        if p_value is None:
            p_value = self.highest_p_value + 1

        p_value = torch.LongTensor([p_value])
        return (
            F.one_hot(p_value, num_classes=self.highest_p_value + 2)
            .squeeze()
            .to(torch.float32)
        )

    def get_m_encoding(self, m_value):
        return (torch.LongTensor([1]) if m_value else torch.LongTensor([0])).to(
            torch.float32
        )

    @lru_cache(maxsize=None)
    def get_p_m_encoding(self, p_value, m_value):
        return torch.cat([self.get_p_encoding(p_value), self.get_m_encoding(m_value)])

    def get_succ1_succ2(self, succ):
        succ1 = torch.mean(succ, dim=0)
        succ2 = torch.sum(torch.mean(succ, dim=1), dim=0)
        succ2 = succ2.unsqueeze(0).repeat(succ1.shape[0], 1)
        return succ1, succ2

    def get_encoded_config(self, config):
        return torch.stack(
            [
                self.get_p_m_encoding(
                    self.cvf_analysis.possible_node_values[i][v].p,
                    self.cvf_analysis.possible_node_values[i][v].m,
                )
                for i, v in enumerate(config)
            ]
        )

    def cvf_analysis_indx_to_config(self, idx):
        return self.cvf_analysis.indx_to_config(idx)

    def cvf_analysis_get_transitions_as_configs(self, config):
        return self.cvf_analysis._get_program_transitions_as_configs(config)

    def get_x(self, config, succ1, succ2):
        return torch.stack([config, succ1]).reshape(2, -1).t()

    def get_default_succs(self, config):
        succ1 = torch.zeros(config.shape[0], config.shape[1]).to(self.device)
        succ2 = succ1.clone()
        return succ1, succ2

    def move_to_device(self, tensor):
        return tensor.to(self.device)

    def __getitem__(self, idx):
        config = self.cvf_analysis_indx_to_config(idx)
        succ = [i[1] for i in self.cvf_analysis_get_transitions_as_configs(config)]
        config = self.move_to_device(self.get_encoded_config(config))

        if succ:
            _succ = [self.get_encoded_config(s) for s in succ]
            succ = self.move_to_device(torch.stack(_succ))
            succ1, succ2 = self.get_succ1_succ2(succ)
        else:
            succ1, succ2 = self.get_default_succs(config)

        result = (self.get_x(config, succ1, succ2), idx, self.edge_index)

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
    device = "cpu"

    dataset = CVFConfigForGCNWSuccLSTMDataset(
        device,
        "star_graph_n4",
        program="graph_coloring",
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for batch in loader:
        # x = batch[0]
        # y = batch[1]
        # print(x[0])
        # print("y", y)
        print(batch)
        break
