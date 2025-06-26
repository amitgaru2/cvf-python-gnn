import os
import sys
import time

import torch
import numpy as np
import pandas as pd

from functools import wraps
from collections import defaultdict

from torch.utils.data import DataLoader, Sampler

from custom_logger import logger
from lstm_scratch import SimpleLSTM
from arg_parser_helper import generate_parser
from helpers import CVFConfigForAnalysisDataset, CVFConfigForAnalysisDatasetMM

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)

from common_helpers import create_dir_if_not_exists

# from mpi_helpers.custom_logger_mpi import logger
from mpi_helpers.custom_mpi import comm, program_node_rank

logger.info("MPI comm size: %s", comm.size)

args = generate_parser(takes_model=True)

model_name = args.model
program = args.program
graph_names = args.graph_names

ONLY_FA = model_name == "fa"

device = "cuda"

function_runtimes = defaultdict(float)


def track_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start
        function_runtimes[func.__name__] += duration
        return result

    return wrapper


class CVFConfigForAnalysisDatasetWithTT(CVFConfigForAnalysisDataset):
    @track_runtime
    def __getitem__(self, idx):
        return super().__getitem__(idx)


class CVFConfigForAnalysisDatasetMMWithTT(CVFConfigForAnalysisDatasetMM):
    @track_runtime
    def __getitem__(self, idx):
        return super().__getitem__(idx)


# Optional utility to print final report
def print_runtime_report():
    logger.info("\n=== Runtime Report : %s ===", program_node_rank)
    for func_name, total_time in function_runtimes.items():
        logger.info(f"Node: %s | {func_name}: {total_time:.6f}s", program_node_rank)
    logger.info("=== End Report : %s ===\n\n", program_node_rank)


@track_runtime
def get_perturbed_states(dataset, frm_idx):
    return dataset.cvf_analysis.possible_perturbed_state_frm(frm_idx)


@track_runtime
def get_model():
    model = torch.load(f"trained_models/{model_name}.pt", weights_only=False)
    model.eval()
    return model


@track_runtime
def get_rank(model, x):
    return model(x)


@track_runtime
def group_data(df, grp_by: list):
    return df.groupby(grp_by)


@track_runtime
def get_dataset(graph_name):
    dataset = (
        CVFConfigForAnalysisDatasetMMWithTT(device, graph_name, program)
        if program == "maximal_matching"
        else CVFConfigForAnalysisDatasetWithTT(device, graph_name, program=program)
    )
    return dataset


@track_runtime
def aggregation(result_df, result_rank_df):
    ml_grp_by_node_re = (
        group_data(result_df, ["node", "rank effect"])
        .size()
        .reset_index(name="ml_count")
    )

    ml_grp_by_re = (
        group_data(result_df, ["rank effect"]).size().reset_index(name="ml_count")
    )

    ml_grp_by_r = (
        group_data(result_rank_df, ["rank"]).size().reset_index(name="ml_count")
    ).astype(int)

    return ml_grp_by_r, ml_grp_by_re, ml_grp_by_node_re


@track_runtime
def gather_aggregate(ml_grp_by_r, ml_grp_by_re, ml_grp_by_node_re):
    all_ml_grp_by_r = comm.gather(ml_grp_by_r, root=0)
    all_ml_grp_by_re = comm.gather(ml_grp_by_re, root=0)
    all_ml_grp_by_node_re = comm.gather(ml_grp_by_node_re, root=0)

    if program_node_rank == 0:
        all_ml_grp_by_r = pd.concat(all_ml_grp_by_r, ignore_index=True)
        ml_grp_by_r = group_data(all_ml_grp_by_r, ["rank"]).sum()

        all_ml_grp_by_re = pd.concat(all_ml_grp_by_re, ignore_index=True)
        ml_grp_by_re = group_data(all_ml_grp_by_re, ["rank effect"]).sum()

        all_ml_grp_by_node_re = pd.concat(all_ml_grp_by_node_re, ignore_index=True)
        ml_grp_by_node_re = group_data(
            all_ml_grp_by_node_re, ["node", "rank effect"]
        ).sum()

    return ml_grp_by_r, ml_grp_by_re, ml_grp_by_node_re


@track_runtime
def aggregation_n_save(result_df, result_rank_df):
    ml_grp_by_r, ml_grp_by_re, ml_grp_by_node_re = aggregation(
        result_df, result_rank_df
    )

    ml_grp_by_r, ml_grp_by_re, ml_grp_by_node_re = gather_aggregate(
        ml_grp_by_r, ml_grp_by_re, ml_grp_by_node_re
    )

    if program_node_rank == 0:
        save_to_dir = os.path.join("ml_predictions", program)
        create_dir_if_not_exists(save_to_dir)

        ml_grp_by_node_re.to_csv(
            os.path.join(
                save_to_dir, f"{model_name}__{program}__{graph_name}__cvf_by_node.csv"
            )
        )

        ml_grp_by_re.to_csv(
            os.path.join(save_to_dir, f"{model_name}__{program}__{graph_name}__cvf.csv")
        )

        ml_grp_by_r.to_csv(
            os.path.join(
                save_to_dir, f"{model_name}__{program}__{graph_name}__rank.csv"
            )
        )

    return ml_grp_by_r, ml_grp_by_re, ml_grp_by_node_re


class SimpleMPISampler(Sampler):
    def __init__(self, data_source, frm_indx, to_indx):
        self.data_source = data_source
        self.frm_indx = frm_indx
        self.to_indx = to_indx

    def __iter__(self):
        return iter(range(self.frm_indx, self.to_indx))

    def __len__(self):
        return self.to_indx - self.frm_indx


@track_runtime
def get_ml_rank_effect(dataset, sampler):
    model = get_model()
    result_df = pd.DataFrame(
        {"node": pd.Series(dtype="int"), "rank effect": pd.Series(dtype="float")}
    )

    data = []
    rank_data = []
    with torch.no_grad():
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=1)
        for n, batch in enumerate(dataloader, 1):
            frm_idx = batch[1].item()
            perturbed_states = [
                (i, indx) for (i, indx) in get_perturbed_states(dataset, frm_idx)
            ]
            perturbed_states_x = [dataset[i[1]][0] for i in perturbed_states]
            x = torch.stack([batch[0][0], *perturbed_states_x])
            ranks = get_rank(model, x)
            rank_data.append(np.round(ranks[0].item()))
            rank_effects = ranks[0] - ranks
            for i, rank_effect in enumerate(rank_effects[1:]):
                data.append((perturbed_states[i][0], rank_effect.item()))

            if n % 5000 == 0:
                temp_df = pd.DataFrame(data, columns=["node", "rank effect"])
                data = []
                result_df = pd.concat([result_df, temp_df], ignore_index=True)

    if data:
        temp_df = pd.DataFrame(data, columns=["node", "rank effect"])
        data = []
        result_df = pd.concat([result_df, temp_df], ignore_index=True)

    return result_df, rank_data


@track_runtime
def ml_cvf_analysis(graph_name):
    dataset = get_dataset(graph_name)
    eq_sample_size = len(dataset) // comm.size
    tot_sample_size = eq_sample_size

    # MPI config
    if program_node_rank == comm.size - 1:
        tot_sample_size += len(dataset) - (eq_sample_size * comm.size)

    frm_indx = eq_sample_size * program_node_rank
    to_indx = tot_sample_size + eq_sample_size * program_node_rank

    logger.info(
        f"Data indices range: {frm_indx:,} to {to_indx:,}.",
    )

    sampler = SimpleMPISampler(dataset, frm_indx, to_indx)
    ##

    result_df, rank_data = get_ml_rank_effect(dataset, sampler)

    result_rank_df = pd.DataFrame(rank_data, columns=["rank"])
    result_df["rank effect"] = np.floor(result_df["rank effect"] + 0.5)
    logger.info("Done ML CVF Analysis!")

    return aggregation_n_save(result_df, result_rank_df)


def get_file_df(dir, file_name):
    file_path = os.path.join(dir, file_name)
    if not os.path.exists(file_path):
        logger.warning("FA results not found for %s.", graph_name)
        return None

    return pd.read_csv(file_path)


@track_runtime
def get_fa_results(graph_name, ml_grp_by_r, ml_grp_by_re, ml_grp_by_node_re):
    results_dir = os.path.join(
        os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "v2", "results", program
    )

    results_file = f"ranks_avg__{graph_name}.csv"
    f_grp_by_r = get_file_df(results_dir, results_file)
    if f_grp_by_r is None:
        return

    f_grp_by_r = f_grp_by_r.drop(f_grp_by_r.columns[0], axis=1)
    f_grp_by_r.rename(columns={"count": "fa_count"}, inplace=True)

    df_grp_by_r = (
        pd.merge(f_grp_by_r, ml_grp_by_r, on="rank", how="outer").fillna(0).astype(int)
    )

    save_to_dir = os.path.join("ml_predictions", program)
    create_dir_if_not_exists(save_to_dir)

    filepath = os.path.join(
        save_to_dir, f"{model_name}__{program}__{graph_name}__rank.csv"
    )

    df_grp_by_r.to_csv(filepath)

    results_file = f"rank_effects_avg__{graph_name}.csv"
    f_grp_by_re = get_file_df(results_dir, results_file)
    if f_grp_by_re is None:
        return

    f_grp_by_re = f_grp_by_re.drop(f_grp_by_re.columns[0], axis=1)
    f_grp_by_re.rename(columns={"count": "fa_count"}, inplace=True)

    df_grp_by_re = (
        pd.merge(f_grp_by_re, ml_grp_by_re, on="rank effect", how="outer")
        .fillna(0)
        .astype(int)
    )

    save_to_dir = os.path.join("ml_predictions", program)
    create_dir_if_not_exists(save_to_dir)

    filepath = os.path.join(
        save_to_dir, f"{model_name}__{program}__{graph_name}__cvf.csv"
    )

    df_grp_by_re.to_csv(filepath)

    results_file = f"rank_effects_by_node_avg__{graph_name}.csv"

    f_grp_by_node_re = get_file_df(results_dir, results_file)
    if f_grp_by_node_re is None:
        return

    f_grp_by_node_re = f_grp_by_node_re.melt(
        id_vars="node",
        value_vars=set(f_grp_by_node_re.columns) - {"node"},
        var_name="rank effect",
        value_name="fa_count",
    )
    f_grp_by_node_re["rank effect"] = f_grp_by_node_re["rank effect"].astype(float)

    df_grp_by_node_re = (
        pd.merge(
            f_grp_by_node_re, ml_grp_by_node_re, on=["node", "rank effect"], how="outer"
        )
        .fillna(0)
        .astype(int)
    )

    filepath = os.path.join(
        save_to_dir, f"{model_name}__{program}__{graph_name}__cvf_by_node.csv"
    )

    df_grp_by_node_re.to_csv(filepath)


@track_runtime
def main(graph_name, has_fa_analysis=True):
    logger.info("Starting for %s.", graph_name)
    if ONLY_FA:
        ml_grp_by_r = pd.DataFrame(columns=["rank"])
        ml_grp_by_re = pd.DataFrame(columns=["rank effect"])
        ml_grp_by_node_re = pd.DataFrame(columns=["node", "rank effect"])
    else:
        ml_grp_by_r, ml_grp_by_re, ml_grp_by_node_re = ml_cvf_analysis(graph_name)

    if program_node_rank == 0 and has_fa_analysis:
        get_fa_results(graph_name, ml_grp_by_r, ml_grp_by_re, ml_grp_by_node_re)

    logger.info("Complete for %s.", graph_name)


if __name__ == "__main__":
    for graph_name in graph_names:
        main(graph_name)
        print_runtime_report()
