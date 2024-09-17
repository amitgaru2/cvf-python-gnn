import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


COLORING_PROGRAM = "coloring"
DIJKSTRA_PROGRAM = "dijkstra_token_ring"
MAX_MATCHING_PROGRAM = "maximal_matching"
MAX_INDEPENDENT_SET_PROGRAM = "maximal_independent_set"
LINEAR_REGRESSION_PROGRAM = "linear_regression"


results_dir = "results"
program = "dijkstra_token_ring"  # coloring, dijkstra_token_ring, maximal_matching, maximal_independent_set
program = sys.argv[1]
programs = {
    DIJKSTRA_PROGRAM,
    COLORING_PROGRAM,
    MAX_MATCHING_PROGRAM,
    MAX_INDEPENDENT_SET_PROGRAM,
    LINEAR_REGRESSION_PROGRAM,
}
if program not in programs:
    print(f"Program {program} not found.")
    exit(1)

program_label_map = {
    "dijkstra_token_ring": "dijkstra",
    "coloring": "graph coloring",
    "maximal_matching": "maximal matching",
    "maximal_independent_set": "maximal indp. set",
    "linear_regression": "linear regression",
}
program_label = program_label_map.get(program, program)

analysis_type = "full"  # full, partial

fontsize = 20

graph_names_map = {
    COLORING_PROGRAM: {
        "graph_1",
        "graph_2",
        "graph_3",
        "graph_6",
        "graph_6b",
        "graph_7",
    },
    DIJKSTRA_PROGRAM: {
        "implicit_graph_n10",
        "implicit_graph_n11",
        "implicit_graph_n12",
        "implicit_graph_n13",
    },
    MAX_MATCHING_PROGRAM: {
        "graph_1",
        "graph_2",
        "graph_3",
        "graph_6",
        "graph_6b",
    },
    MAX_INDEPENDENT_SET_PROGRAM: {
        "graph_1",
        "graph_2",
        "graph_3",
        "graph_4",
        "graph_5",
        "graph_6",
        "graph_6b",
        "graph_7",
        "graph_8",
    },
    LINEAR_REGRESSION_PROGRAM: {
        # "0.8_1.9__0.025__test_lr_graph_1",
        # "0.8_1.9__0.025__2__test_lr_graph_1",
        "0.8_1.9__0.025__3__test_lr_graph_1",
        # "0.7_1.9__0.025__test_lr_graph_1",
        # "0.9_1.9__0.025__test_lr_graph_1",
    },
}

graph_names = graph_names_map[program]

plots_dir = os.path.join("plots", program)


def get_df(graph_name):
    full_path = os.path.join(
        results_dir,
        program,
        f"rank_effect_by_node__{analysis_type}__{program}__{graph_name}.csv",
    )
    if not os.path.exists(full_path):
        print("File not found:", full_path)
        return None

    df = pd.read_csv(full_path)
    df["CVF (Avg)"] = df["CVF In (Avg)"] + df["CVF Out (Avg)"]
    df["CVF (Max)"] = df["CVF In (Max)"] + df["CVF Out (Max)"]
    return df


def plot_node_rank_effect(node, df, ax):
    df = df.loc[df["CVF (Avg)"] > 0]
    sns.lineplot(data=df, x="Rank Effect", y="CVF (Avg)", ax=ax, marker="o")
    ax.set(xlabel=f"Rank Effect of Node: {node}", ylabel="Count")
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.set_title("CVF Avg")
    if df.shape[0] > 0:
        ax.set_yscale("log")


def plot_node_rank_effect_max(node, df, ax):
    df = df.loc[df["CVF (Max)"] > 0]
    sns.lineplot(data=df, x="Rank Effect", y="CVF (Max)", ax=ax)
    ax.set(xlabel=f"Rank Effect of Node: {node}", ylabel="Count")
    ax.set_title("CVF Max")
    if df.shape[0] > 0:
        ax.set_yscale("log")


def create_plots_dir_if_not_exists():
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)


create_plots_dir_if_not_exists()
plt.figure(figsize=(18, 10))

for graph_name in graph_names:
    df = get_df(graph_name)
    if df is None:
        continue

    rank_effects = df["Rank Effect"].unique()
    rank_effects.sort()
    df_preproc = pd.DataFrame({"Rank Effect": rank_effects})
    nodes = df["Node"].unique()
    nodes.sort()
    for node in nodes:
        col = f"Node {node}"
        node_data = df.loc[(df["Node"] == node)]["CVF (Avg)"]
        node_data = node_data.reset_index(drop=True)
        df_preproc.loc[:, col] = node_data

    df_preproc.set_index("Rank Effect", inplace=True)

    ax = sns.lineplot(data=df_preproc[["Node 0", "Node 1", "Node 2", "Node 3"]], marker='.')
    ax.set_xlabel("Rank Effect")

    # Set custom ticks and labels
    ax.set_yscale("log")
    ax.set_ylabel("Count", rotation=0, labelpad=20)
    file_name = f"rank_effect_by_node__{analysis_type}__{program}__{graph_name}"
    # fig_title = (
    #     f"Distribution of rank effects at node {index[0]} in {program_label} problem"
    # )
    # fig.suptitle(fig_title, fontsize=fontsize)
    plt.rc("font", size=20)
    plt.savefig(
        os.path.join(
            plots_dir,
            f"{file_name}.png",
        )
    )
    plt.close()

    print(f"Saved plot(s) for {graph_name}.")
