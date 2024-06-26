# TODO: Accumulate cvf new graphs for decreasing cut off like 40, 20, 10, 0
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


def plot_node_vs_accumulated_cvf_effect(df, ax, y_max):
    sns.barplot(data=df, x="Node", y="Accumulated Severe CVF Effect (Avg)", ax=ax)
    ax.set_ylim(bottom=0, top=math.ceil(y_max * 1.1))


if __name__ == "__main__":
    results_dir = os.path.join(os.pardir, "results")
    graphs_dir = os.path.join(os.pardir, "graphs")
    program = "dijkstra_token_ring"  # coloring, dijkstra_token_ring, maximal_matching, maximal_independent_set
    analysis_type = "full"  # full, partial
    # cut_off = [40, 40, 50, 60]
    cut_off = [20, 10, 15, 10]
    graph_names = [
        "graph_1",
        "graph_2",
        "graph_3",
        "graph_6",
    ]
    cut_off = [40, 40, 50, 60]
    graph_names = [
        "implicit_graph_n10",
        "implicit_graph_n11",
        "implicit_graph_n12",
        "implicit_graph_n13",
    ]
    plots_dir = os.path.join("plots", program, "node_vs_accumulated_cvf_effect")

    create_plots_dir_if_not_exists()

    for indx, graph_name in enumerate(graph_names):
        df = get_df(graph_name)
        if df is None:
            continue
        node_id_max = df.agg({"Node": ["max"]})["Node"]["max"]
        df = df[(df["CVF (Avg)"] > 0) & (df["Rank Effect"] > cut_off[indx])]
        df["Accumulated Severe CVF Effect (Avg)"] = df.apply(
            lambda x: x["Rank Effect"] * x["CVF (Avg)"], axis=1
        )
        node_vs_accumulated_cvf_effect = (
            df.groupby(["Node"])
            .agg({"Accumulated Severe CVF Effect (Avg)": ["sum"]})
            .droplevel(1, axis=1)
        )
        fig, ax = plt.subplots(
            1,
            figsize=(12, 5),
        )
        any_grps_filtered_out = set(range(node_id_max + 1)) - set(
            node_vs_accumulated_cvf_effect.index
        )
        any_grps_filtered_out = list(any_grps_filtered_out)
        any_grps_filtered_out.sort()
        for grp in any_grps_filtered_out:
            node_vs_accumulated_cvf_effect.loc[grp] = 0
        fig_title = f"node__vs__accumulated_severe_cvf_effect_gte_{cut_off[indx]}__{analysis_type}__{program}__{graph_name}"
        fig.suptitle(fig_title, fontsize=16)
        plot_node_vs_accumulated_cvf_effect(
            node_vs_accumulated_cvf_effect,
            ax,
            node_vs_accumulated_cvf_effect["Accumulated Severe CVF Effect (Avg)"].max(),
        )

        fig.savefig(
            os.path.join(
                plots_dir,
                f"{fig_title}.png",
            )
        )
