import os
import sys
import time

import torch
import numpy as np
import pandas as pd

from functools import wraps
from collections import defaultdict

from torch.utils.data import DataLoader

from custom_logger import logger
from lstm_scratch import SimpleLSTM
from helpers import CVFConfigForAnalysisDataset


model_name = "lstm_trained_at_2025_05_12_21_31"

program = "dijkstra"
# program = "maximal_matching"

# graph_name = "graph_random_regular_graph_n8_d4"


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
        # logger.info(
        #     f"[{func.__name__}] call took {duration:.6f}s, total: {function_runtimes[func.__name__]:.6f}s"
        # )
        return result

    return wrapper


# Optional utility to print final report
def print_runtime_report():
    logger.info("\n=== Runtime Report ===")
    for func_name, total_time in function_runtimes.items():
        logger.info(f"{func_name}: {total_time:.6f}s")


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
def ml_cvf_analysis():
    model = get_model()

    dataset = CVFConfigForAnalysisDataset(device, graph_name, program=program)

    result_df = pd.DataFrame(
        {"node": pd.Series(dtype="int"), "rank effect": pd.Series(dtype="float")}
    )

    data = []
    with torch.no_grad():
        test_dataloader = DataLoader(dataset, batch_size=1)
        for batch in test_dataloader:
            frm_idx = batch[1].item()
            perturbed_states = [
                (p, indx) for (p, indx) in get_perturbed_states(dataset, frm_idx)
            ]
            perturbed_states_x = [dataset[i[1]][0] for i in perturbed_states]
            x = torch.stack([batch[0][0], *perturbed_states_x])
            ranks = get_rank(model, x)
            frm_rank = ranks[0]
            for i, to_rank in enumerate(ranks[1:]):
                rank_effect = (frm_rank - to_rank).item()
                data.append(
                    {"node": perturbed_states[i][0], "rank effect": rank_effect}
                )

            temp_df = pd.DataFrame(data, columns=["node", "rank effect"])
            data = []
            result_df = pd.concat([result_df, temp_df], ignore_index=True)

    result_df["rank effect"] = np.floor(result_df["rank effect"] + 0.5)

    logger.info("Done ML CVF Analysis!")

    ml_grp_by_node_re = (
        group_data(result_df, ["node", "rank effect"])
        .size()
        .reset_index(name="ml_count")
    )

    ml_grp_by_node_re.to_csv(
        f"ml_predictions/{model_name}__{graph_name}__cvf_by_node.csv"
    )

    ml_grp_by_re = (
        group_data(result_df, ["rank effect"]).size().reset_index(name="ml_count")
    )

    ml_grp_by_re.to_csv(f"ml_predictions/{model_name}__{graph_name}__cvf.csv")

    return ml_grp_by_re, ml_grp_by_node_re


@track_runtime
def get_fa_results(graph_name, ml_grp_by_re, ml_grp_by_node_re):
    results_dir = os.path.join(
        os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "v2", "results", program
    )

    results_file = f"rank_effects_avg__{graph_name}.csv"
    file_path = os.path.join(results_dir, results_file)
    if not os.path.exists(file_path):
        logger.warning("FA results not found for %s.", graph_name)
        return

    f_grp_by_re = pd.read_csv(file_path)
    f_grp_by_re = f_grp_by_re.drop(f_grp_by_re.columns[0], axis=1)
    f_grp_by_re.rename(columns={"count": "fa_count"}, inplace=True)

    df_grp_by_re = pd.merge(
        f_grp_by_re, ml_grp_by_re, on="rank effect", how="outer"
    ).fillna(0)

    df_grp_by_re.to_csv(f"ml_predictions/{model_name}__{graph_name}__cvf.csv")

    results_file = f"rank_effects_by_node_avg__{graph_name}.csv"

    file_path = os.path.join(results_dir, results_file)
    f_grp_by_node_re = pd.read_csv(file_path)
    f_grp_by_node_re = f_grp_by_node_re.melt(
        id_vars="node",
        value_vars=set(f_grp_by_node_re.columns) - {"node"},
        var_name="rank effect",
        value_name="fa_count",
    )
    f_grp_by_node_re["rank effect"] = f_grp_by_node_re["rank effect"].astype(float)

    df_grp_by_node_re = pd.merge(
        f_grp_by_node_re, ml_grp_by_node_re, on=["node", "rank effect"], how="outer"
    ).fillna(0)

    df_grp_by_node_re.to_csv(
        f"ml_predictions/{model_name}__{graph_name}__cvf_by_node.csv"
    )


def main(graph_name, has_fa_analysis=True):
    logger.info("Starting for %s.", graph_name)
    ml_grp_by_re, ml_grp_by_node_re = ml_cvf_analysis()
    if has_fa_analysis:
        get_fa_results(graph_name, ml_grp_by_re, ml_grp_by_node_re)

    logger.info("Complete for %s.", graph_name)
    print_runtime_report()


if __name__ == "__main__":
    graph_name = sys.argv[1]
    main(graph_name)
