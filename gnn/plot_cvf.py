import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines

from itertools import cycle
from matplotlib import pyplot as plt

from arg_parser_helper import generate_parser

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)

from common_helpers import create_dir_if_not_exists
from command_line_helpers import ColoringProgram, DijkstraProgram, MaxMatchingProgram

args = generate_parser(takes_model=True)
model = args.model
program = args.program
graph_names = args.graph_names

plots_dir = os.path.join("plots", program)
create_dir_if_not_exists(plots_dir)

fontsize = 20
markers = ["*", "o", "h", "v", "P", "s", "p", "x", "D", "8"]

ONLY_FA = model == "fa"


TITLE_PROGRAM_MAP = {
    ColoringProgram: "Graph Coloring",
    DijkstraProgram: "Dijkstra Token Ring",
    MaxMatchingProgram: "Maximal Matching",
}

result_type = "cvf"


def get_title():
    result = " ".join(graph_name.split("_")).title()
    return f"{TITLE_PROGRAM_MAP[program]} - {result}"


def main(graph_name, marker_cycle):
    filepath = os.path.join(
        "ml_predictions",
        program,
        f"{model}__{program}__{graph_name}__{result_type}.csv",
    )
    df = pd.read_csv(filepath, index_col=0)
    if ONLY_FA:
        df = df[["rank effect", "fa_count"]]
        df = df.rename(columns={"fa_count": "FA count"})
        legends = ["FA count"]
    elif "fa_count" in df.columns:
        df = df[["rank effect", "ml_count", "fa_count"]]
        df = df.rename(columns={"ml_count": "ML Count", "fa_count": "FA count"})
        legends = ["ML count", "FA count"]
    else:
        df = df[["rank effect", "ml_count"]]
        df = df.rename(columns={"ml_count": "ML Count"})
        legends = ["ML count"]

    df = df.set_index("rank effect", drop=True)

    plot_df(df, legends, marker_cycle)


def plot_df(df, legends, marker_cycle):
    plt.figure(figsize=(16, 8))
    ax = sns.lineplot(data=df, linewidth=1, markersize=10)

    for line in ax.lines:
        line.set_marker(next(marker_cycle))

    ax.set_xlabel("Rank Effect")

    ax.set_yscale("log")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)

    ax.set_title(get_title(), fontdict={"fontsize": fontsize})

    file_name = f"RE__{model}__{program}__{graph_name}__{model}.png"
    custom_lines = [
        mlines.Line2D(
            [],
            [],
            color=line.get_color(),
            marker=marker,
            label=cat,
            linewidth=1,
            markersize=10,
            linestyle=line.get_linestyle(),
        )
        for line, marker, cat in zip(ax.lines, markers, legends)
    ]

    plt.rc("font", size=fontsize)
    plt.legend(handles=custom_lines, fontsize=fontsize * 0.9)  # using a size in points
    filepath = os.path.join(
        plots_dir,
        f"{file_name}",
    )
    plt.savefig(
        filepath,
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved plot(s) at %s" % filepath)


if __name__ == "__main__":
    for graph_name in graph_names:
        marker_cycle = cycle(markers)
        main(graph_name, marker_cycle)
