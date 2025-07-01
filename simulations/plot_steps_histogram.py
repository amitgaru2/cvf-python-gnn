"""Plot and save the aggregated data of simulations for the given Graph, Program and Nodes."""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines

from itertools import cycle
from matplotlib import pyplot as plt

from simulation import SimulationMixin


utils_path = os.path.join(
    os.getenv("CVF_PROJECT_DIR", "/home/agaru/research/cvf-python-gnn"), "utils"
)

sys.path.append(utils_path)

from command_line_helpers import PROGRAM_CHOICES
from common_helpers import create_dir_if_not_exists


fontsize = 20

plots_dir = "plots"


colors = [("red", "red"), ("green", "green"), ("blue", "blue"), ("orange", "orange")]
color_cycle = cycle(colors)

sched = 0
me = False


def generate_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--program",
        choices=PROGRAM_CHOICES,
        required=True,
    )

    parser.add_argument(
        "--graph-name",
        type=str,
        help="graph name in the 'graphs_dir' or list of number of nodes for implict graphs (if implicit program)",
        required=True,
    )

    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        help="Nodes",
        required=True,
    )

    parser.add_argument(
        "--no-simulations",
        type=int,
        help="No. of simulations",
        required=True,
    )

    parser.add_argument(
        "--fault-interval",
        type=int,
        help="fault interval",
        default=1,
        required=True,
    )

    parser.add_argument("--duong-mode", action="store_true")

    args = parser.parse_args()

    return args


def get_sim_data_filename(
    graph_name, sched, simulation_type, args, no_simulations, me, fault_interval
):
    return f"{graph_name}__{sched}__{simulation_type}_args_{args}__{no_simulations}__{me}__{fault_interval}"


def get_filenames(
    graph_name, selected_nodes, no_simulations, fault_interval, duong_mode
):
    filenames = [
        get_sim_data_filename(
            graph_name,
            sched,
            SimulationMixin.RANDOM_FAULT_SIMULATION_TYPE,
            "",
            no_simulations,
            me,
            fault_interval,
        )
    ]
    if duong_mode:
        filenames.extend(
            [
                get_sim_data_filename(
                    graph_name,
                    sched,
                    SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_DUONG,
                    arg,
                    no_simulations,
                    me,
                    fault_interval,
                )
                for arg in selected_nodes
            ]
        )
    else:
        filenames.extend(
            [
                get_sim_data_filename(
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

    return filenames


def get_title(program, graph_name, no_simulations, fault_interval):
    return f"Simulation - {program} | {graph_name} | Sched: {sched} | N: {no_simulations:,} | FI: {fault_interval}"


def get_filename(
    program, graph_name, selected_nodes, no_simulations, fault_interval, duong_mode
):
    return f"{program}__{graph_name}__{sched}__{no_simulations}__{fault_interval}__{''.join([str(i) for i in selected_nodes])}{'__duong' if duong_mode else ''}"


def plot_save_fig(
    df, program, graph_name, selected_nodes, no_simulations, fault_interval, duong_mode
):
    plt.figure(figsize=(16, 8))
    ax = sns.lineplot(data=df, linewidth=3)
    ax.set_title(
        get_title(program, graph_name, no_simulations, fault_interval),
        fontdict={"fontsize": fontsize},
    )

    # ax.set_yscale("log")
    for i, line in enumerate(ax.lines):
        if i >= 1:
            line_style = "solid"
            line.set_color(next(color_cycle)[0])
        else:
            line_style = "dashed"
            line.set_color("goldenrod")
        line.set_linestyle(line_style)

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Count")

    labels = ["Random Fault"]
    labels.extend(
        [
            f'Controlled {"(duong)" if duong_mode else ""} at node {n}'
            for n in selected_nodes
        ]
    )
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
    plots_dir = os.path.join("plots", program)
    create_dir_if_not_exists(plots_dir)
    file_path = os.path.join(
        plots_dir,
        f"{get_filename(program, graph_name, selected_nodes, no_simulations, fault_interval, duong_mode)}.png",
    )
    plt.savefig(
        file_path,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved plot(s) for {file_path}")


def save_agg_data(
    df, program, graph_name, selected_nodes, no_simulations, fault_interval, duong_mode
):
    # Save to file
    df.columns = [
        "Random",
        *[
            f'Controlled {"(duong)" if duong_mode else ""} at node {n}'
            for n in selected_nodes
        ],
    ]
    df.index = df.index.astype(int)
    results_dir = os.path.join("results", program)
    create_dir_if_not_exists(results_dir)
    file_path = os.path.join(
        results_dir,
        f"agg_{get_filename(program, graph_name, selected_nodes, no_simulations, fault_interval, duong_mode)}.csv",
    )
    df.to_csv(file_path)

    print(f"Saved agg data for {file_path}")


def main(
    program, graph_name, selected_nodes, no_simulations, fault_interval, duong_mode
):
    dfs = [
        pd.read_csv(os.path.join("results", program, f"{fn}.csv"))
        for fn in get_filenames(
            graph_name, selected_nodes, no_simulations, fault_interval, duong_mode
        )
    ]
    max_steps = max(df["Steps"].max() for df in dfs)
    bins = np.linspace(0, max_steps, max_steps + 1)
    bin_centers = bins
    hist_data = [np.histogram(df["Steps"], bins=bins) for df in dfs]

    hist_df = []
    for hd in hist_data:
        hist_df.append(pd.DataFrame({"Steps": bin_centers[:-1], "Count": hd[0]}))

    df_merged = hist_df[0]
    for i in range(1, len(hist_df)):
        df_merged = pd.merge(df_merged, hist_df[i], on=["Steps"], suffixes=(i - 1, i))

    df_merged.set_index("Steps", drop=True, inplace=True)

    plot_save_fig(
        df_merged,
        program,
        graph_name,
        selected_nodes,
        no_simulations,
        fault_interval,
        duong_mode,
    )
    save_agg_data(
        df_merged,
        program,
        graph_name,
        selected_nodes,
        no_simulations,
        fault_interval,
        duong_mode,
    )


if __name__ == "__main__":
    args = generate_parser()
    main(
        args.program,
        args.graph_name,
        args.nodes,
        args.no_simulations,
        args.fault_interval,
        args.duong_mode,
    )
