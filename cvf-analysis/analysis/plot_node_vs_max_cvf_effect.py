import os
import math
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


from plot_config import *


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
    sns.barplot(data=df, x="Node", y="Rank Effect", ax=ax, width=0.4)
    ax.set_ylim(bottom=0, top=math.ceil(y_max * 1.1))
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.yaxis.offsetText.set_fontsize(fontsize)
    ax.tick_params(axis="x", labelsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_xlabel("Node ID")


if __name__ == "__main__":

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
        fig_title = f"Node vs Max CVF effect for program {program_label}"
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

        print(f"Saved plot for {graph_name}")
