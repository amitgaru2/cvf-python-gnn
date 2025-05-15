import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines

from matplotlib import pyplot as plt

from simulation import SimulationMixin


fontsize = 20

plots_dir = "plots"

selected_nodes = [0, 3, 6]

program = sys.argv[1]
graph_name = sys.argv[2]
sched = 0
no_simulations = 500000
me = False
fault_interval = sys.argv[3]


def get_filename(
    graph_name, sched, simulation_type, args, no_simulations, me, fault_interval
):
    return f"{graph_name}__{sched}__{simulation_type}_args_{args}__{no_simulations}__{me}__{fault_interval}"


def get_title():
    return f"Simulation - {program} | {graph_name} | Sched: {sched} | N: {no_simulations:,} | FI: {fault_interval}"


def get_filename():
    return f"{graph_name}__{sched}__{no_simulations:,}__{fault_interval}__{''.join([str(i) for i in selected_nodes])}"


def main():
    filenames = [
        get_filename(
            graph_name,
            sched,
            SimulationMixin.RANDOM_FAULT_SIMULATION_TYPE,
            "",
            no_simulations,
            me,
            fault_interval,
        )
    ]
    filenames.extend(
        [
            get_filename(
                graph_name,
                sched,
                SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE,
                arg,
                no_simulations,
                me,
                fault_interval,
            )
            for arg in selected_nodes
        ]
    )

    dfs = [
        pd.read_csv(os.path.join("results", program, f"{fn}.csv")) for fn in filenames
    ]

    max_steps = max(df["Steps"].max() for df in dfs)

    bins = np.linspace(0, max_steps, 20)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist_data = [np.histogram(df["Steps"], bins=bins) for df in dfs]
    hist_df = []
    for hd in hist_data:
        hist_df.append(pd.DataFrame({"Steps": bin_centers, "Count": hd[0]}))

    df_merged = hist_df[0]
    for i in range(1, len(hist_df)):
        df_merged = pd.merge(df_merged, hist_df[i], on=["Steps"], suffixes=(i - 1, i))

    df_merged.set_index("Steps", drop=True, inplace=True)
    plot_data(df_merged)


def plot_data(df_merged):
    plt.figure(figsize=(16, 8))
    ax = sns.lineplot(data=df_merged, marker="o", linewidth=3)
    ax.set_title(get_title(), fontdict={"fontsize": fontsize})

    # ax.set_yscale("log")
    for i, line in enumerate(ax.lines):
        if i >= 1:
            line_style = "solid"
        else:
            line_style = "dashed"
        line.set_linestyle(line_style)

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Count")

    labels = ["Random Fault"]
    labels.extend([f"Controlled at node {n}" for n in selected_nodes])
    custom_lines = [
        mlines.Line2D(
            [],
            [],
            color=line.get_color(),
            # marker=marker,
            label=cat,
            linewidth=1,
            linestyle=line.get_linestyle(),
        )
        for line, cat in zip(ax.lines, labels)
    ]
    plt.rc("font", size=fontsize)
    plt.legend(handles=custom_lines, fontsize=fontsize * 0.9)
    plt.savefig(
        os.path.join(
            plots_dir,
            program,
            f"{get_filename()}.png",
        ),
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
