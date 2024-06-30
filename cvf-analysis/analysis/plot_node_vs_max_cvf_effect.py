import os
import math
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


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
    return df


def create_plots_dir_if_not_exists():
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)


def plot_node_vs_max_rank_effect(df, ax, y_max):
    sns.barplot(data=df, x="Node", y="Rank Effect", ax=ax)
    ax.set_ylim(bottom=0, top=math.ceil(y_max * 1.1))
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)


if __name__ == "__main__":
    results_dir = os.path.join(os.pardir, "results")
    graphs_dir = os.path.join(os.pardir, "graphs")
    program = "dijkstra_token_ring"  # coloring, dijkstra_token_ring, maximal_matching, maximal_independent_set
    program_label_map = {"dijkstra_token_ring": "dijkstra_tr"}
    program_label = program_label_map.get(program, program)
    analysis_type = "full"  # full, partial
    fontsize = 15
    graph_names = [
        "graph_1",
        "graph_2",
        "graph_3",
        "graph_6",
        "graph_6b",
    ]
    graph_names = [
        "implicit_graph_n10",
        "implicit_graph_n11",
        "implicit_graph_n12",
        "implicit_graph_n13",
    ]
    plots_dir = os.path.join("plots", program, "node_vs_max_cvf_effect")

    create_plots_dir_if_not_exists()

    for graph_name in graph_names:
        df = get_df(graph_name)
        if df is None:
            continue
        node_vs_max_rank_effect = (
            df[df["CVF (Avg)"] > 0]
            .groupby(["Node"])
            .agg({"Rank Effect": ["max"]})
            .droplevel(1, axis=1)
        )
        fig, ax = plt.subplots(1, figsize=(10, 5), constrained_layout=True)
        file_name = f"node_vs_max_rank_effect__{analysis_type}__{program}__{graph_name}"
        fig_title = f"node_vs_max_rank_effect__{program_label}__{graph_name}"
        fig.suptitle(fig_title, fontsize=fontsize)
        plot_node_vs_max_rank_effect(
            node_vs_max_rank_effect, ax, node_vs_max_rank_effect["Rank Effect"].max()
        )

        fig.savefig(
            os.path.join(
                plots_dir,
                f"{file_name}.png",
            )
        )
