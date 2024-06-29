import math
import os
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

results_dir = "results"
program = "coloring"  # coloring, dijkstra_token_ring, maximal_matching, maximal_independent_set
analysis_type = "full"  # full, partial
graph_names = [
    "graph_1",
    "graph_2",
    "graph_3",
    "graph_4",
    "graph_5",
    "graph_6",
    "graph_6b",
    "graph_7",
    "graph_8",
]
# graph_names = [
#     "implicit_graph_n10",
#     "implicit_graph_n11",
#     "implicit_graph_n12",
#     "implicit_graph_n13",
# ]
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
    sns.lineplot(data=df, x="Rank Effect", y="CVF (Avg)", ax=ax)
    ax.set(xlabel=f"Rank Effect of Node: {node}", ylabel="Count")
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

for graph_name in graph_names:
    df = get_df(graph_name)
    if df is None:
        continue
    node_grps = df.groupby(["Node"])
    # fig, axs = plt.subplots(
    #     node_grps.ngroups, 2, figsize=(12, 20), constrained_layout=True
    # )
    no_of_cols = 5
    no_of_rows = math.ceil(node_grps.ngroups / no_of_cols)
    fig, axs = plt.subplots(
        no_of_rows,
        no_of_cols,
        figsize=(
            4 * no_of_cols,
            4 * no_of_rows,
        ),
        constrained_layout=True,
    )
    # fig_title = f"rank_effect_by_node__{analysis_type}__{program}__{graph_name}"
    # fig.suptitle(fig_title, fontsize=16)

    # for i, (index, grp) in enumerate(node_grps):
    #     plot_node_rank_effect(index[0], grp, axs[i][0])
    #     plot_node_rank_effect_max(index[0], grp, axs[i][1])
    for i, (index, grp) in enumerate(node_grps):
        fig, axs = plt.subplots(
            1,
            1,
            figsize=(10, 5),
            constrained_layout=True,
        )
        node_id = index[0]
        if node_id < 10:
            node_id = f"0{node_id}"
        else:
            node_id = f"{node_id}"
        fig_title = f"rank_effect_by_node__{analysis_type}__{program}__{graph_name}__node_{node_id}"
        fig.suptitle(fig_title, fontsize=10)
        plot_node_rank_effect(index[0], grp, axs)
        fig.savefig(
            os.path.join(
                plots_dir,
                f"{fig_title}.png",
            )
        )
        plt.close()

    # if node_grps.ngroups % no_of_cols != 0:
    #     r, c = node_grps.ngroups // no_of_cols, node_grps.ngroups % no_of_cols
    #     while c % no_of_cols != 0:
    #         axs[r, c].set_axis_off()
    #         c += 1

    # fig.savefig(
    #     os.path.join(
    #         plots_dir,
    #         f"{fig_title}.png",
    #     )
    # )
