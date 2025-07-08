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

    parser.add_argument(
        "--limit-steps",
        type=int,
        help="limit steps",
        default=None,
    )

    parser.add_argument(
        "--simulation-type", choices=SimulationMixin.SIMULATION_TYPES, required=True
    )

    parser.add_argument("--include-random", action="store_true")

    args = parser.parse_args()

    return args


def get_sim_data_filename(
    graph_name,
    sched,
    simulation_type,
    args,
    no_simulations,
    me,
    fault_interval,
    limit_steps,
):
    limits_text = f"__limits_{limit_steps}" if limit_steps else ""
    return f"{graph_name}__{sched}__{simulation_type}_args_{args}__{no_simulations}__{me}__{fault_interval}{limits_text}"


def get_filenames(
    graph_name,
    selected_nodes,
    simulation_type,
    no_simulations,
    fault_interval,
    include_random,
    limit_steps,
):
    filenames = (
        [
            get_sim_data_filename(
                graph_name,
                sched,
                SimulationMixin.RANDOM_FAULT_SIMULATION_TYPE,
                "",
                no_simulations,
                me,
                fault_interval,
                limit_steps,
            )
        ]
        if include_random
        else []
    )

    filenames.extend(
        [
            get_sim_data_filename(
                graph_name,
                sched,
                simulation_type,
                arg,
                no_simulations,
                me,
                fault_interval,
                limit_steps,
            )
            for arg in selected_nodes
        ]
    )

    return filenames


def get_title(program, graph_name, no_simulations, fault_interval):
    return f"Simulation - {program} | {graph_name} | Sched: {sched} | N: {no_simulations:,} | FI: {fault_interval}"


def get_save_filename(
    program,
    graph_name,
    selected_nodes,
    simulation_type,
    no_simulations,
    fault_interval,
):
    return f"{program}__{graph_name}__{sched}__{simulation_type}__{no_simulations}__{fault_interval}__{''.join([str(i) for i in selected_nodes])}"


def get_label(simulation_type, node):
    return {
        SimulationMixin.RANDOM_FAULT_SIMULATION_TYPE: "Random Fault",
        SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE: "Controlled (amit v1) at node %s",
        SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_AMIT_V2: "Controlled (amit v2) at node %s",
        SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_DUONG: "Controlled (duong) at node %s",
        SimulationMixin.RANDOM_FAULT_START_AT_NODE_SIMULATION_TYPE: "Random started at node %s",
    }.get(simulation_type, simulation_type) % (node)


def plot_save_fig(
    df,
    program,
    graph_name,
    selected_nodes,
    simulation_type,
    no_simulations,
    fault_interval,
    include_random,
):
    plt.figure(figsize=(16, 8))
    ax = sns.lineplot(data=df, linewidth=1)
    ax.set_title(
        get_title(program, graph_name, no_simulations, fault_interval),
        fontdict={"fontsize": fontsize},
    )

    # ax.set_yscale("log")
    for i, line in enumerate(ax.lines):
        if i == 0 and include_random:
            line_style = "dashed"
            line.set_color("goldenrod")
        else:
            line_style = "solid"
            line.set_color(next(color_cycle)[0])
        line.set_linestyle(line_style)

    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)

    ax.set_xlabel("Steps")
    ax.set_ylabel("Count")

    labels = ["Random Fault"] if include_random else []
    labels.extend([get_label(simulation_type, n) for n in selected_nodes])
    custom_lines = [
        mlines.Line2D(
            [],
            [],
            color=line.get_color(),
            # marker=marker,
            label=cat,
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
        f"{get_save_filename(program, graph_name, selected_nodes, simulation_type, no_simulations, fault_interval)}.png",
    )
    plt.savefig(
        file_path,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved plot(s) for {file_path}")


def save_agg_data(
    df,
    program,
    graph_name,
    selected_nodes,
    simulation_type,
    no_simulations,
    fault_interval,
    include_random,
):
    cols = ["Random Fault"] if include_random else []
    cols.extend([get_label(simulation_type, n) for n in selected_nodes])
    df.columns = cols
    df.index = df.index.astype(int)
    results_dir = os.path.join("results", program)
    create_dir_if_not_exists(results_dir)
    file_path = os.path.join(
        results_dir,
        f"agg_{get_save_filename(program, graph_name, selected_nodes, simulation_type, no_simulations, fault_interval)}.csv",
    )
    df.to_csv(file_path)

    print(f"Saved agg data for {file_path}")


def main(
    program,
    graph_name,
    selected_nodes,
    simulation_type,
    no_simulations,
    fault_interval,
    include_random,
    limit_steps,
):
    dfs = [
        pd.read_csv(os.path.join("results", program, f"{fn}.csv"))
        for fn in get_filenames(
            graph_name,
            selected_nodes,
            simulation_type,
            no_simulations,
            fault_interval,
            include_random,
            limit_steps,
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
        simulation_type,
        no_simulations,
        fault_interval,
        include_random,
    )
    save_agg_data(
        df_merged,
        program,
        graph_name,
        selected_nodes,
        simulation_type,
        no_simulations,
        fault_interval,
        include_random,
    )


if __name__ == "__main__":
    args = generate_parser()
    main(
        args.program,
        args.graph_name,
        args.nodes,
        args.simulation_type,
        args.no_simulations,
        args.fault_interval,
        args.include_random,
        args.limit_steps,
    )
