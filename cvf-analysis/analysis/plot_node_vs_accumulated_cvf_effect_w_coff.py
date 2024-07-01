# TODO: Accumulate cvf new graphs for decreasing cut off like 40, 20, 10, 0
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


def plot_node_vs_accumulated_cvf_effect(df, ax, y_max):
    sns.barplot(data=df, x="Node", y="Accumulated Severe CVF Effect (Avg)", ax=ax, width=.4)
    ax.set_ylabel("Accumulated Severe CVF Effect")
    ax.set_ylim(bottom=0, top=math.ceil(y_max * 1.1))
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)
    ax.yaxis.offsetText.set_fontsize(fontsize)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_xlabel("Node ID")


if __name__ == "__main__":

    plots_dir = os.path.join("plots", program, "node_vs_accumulated_cvf_effect")

    create_plots_dir_if_not_exists()

    for indx, graph_name in enumerate(graph_names):
        df = get_df(graph_name)
        if df is None:
            continue
        cut_off = graph_names[graph_name]["cut_off"]
        node_id_max = df.agg({"Node": ["max"]})["Node"]["max"]
        for c_off in [cut_off, cut_off // 2, 0]:
            df_copy = df.copy()
            df_copy = df_copy[
                (df_copy["CVF (Avg)"] > 0) & (df_copy["Rank Effect"] >= c_off)
            ]
            df_copy["Accumulated Severe CVF Effect (Avg)"] = df.apply(
                lambda x: x["Rank Effect"] * x["CVF (Avg)"], axis=1
            )
            node_vs_accumulated_cvf_effect = (
                df_copy.groupby(["Node"])
                .agg({"Accumulated Severe CVF Effect (Avg)": ["sum"]})
                .droplevel(1, axis=1)
            )
            fig, ax = plt.subplots(1, figsize=(10, 5), constrained_layout=True)
            any_grps_filtered_out = set(range(node_id_max + 1)) - set(
                node_vs_accumulated_cvf_effect.index
            )
            any_grps_filtered_out = list(any_grps_filtered_out)
            any_grps_filtered_out.sort()
            for grp in any_grps_filtered_out:
                node_vs_accumulated_cvf_effect.loc[grp] = 0

            file_name_substr = f"{c_off}" if len(str(c_off)) > 1 else f"0{c_off}"
            file_name = f"node__vs__accumulated_severe_cvf_effect_gte_{file_name_substr}__{analysis_type}__{program}__{graph_name}"
            fig_title = f"Node vs Accumulated CVF >= {c_off} for {program_label} program"
            fig.suptitle(fig_title, fontsize=fontsize)
            plot_node_vs_accumulated_cvf_effect(
                node_vs_accumulated_cvf_effect,
                ax,
                node_vs_accumulated_cvf_effect[
                    "Accumulated Severe CVF Effect (Avg)"
                ].max(),
            )
            fig.savefig(
                os.path.join(
                    plots_dir,
                    f"{file_name}.png",
                )
            )

        print(f"Saved plot for {graph_name}")
