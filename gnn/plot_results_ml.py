import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines

from itertools import cycle
from matplotlib import pyplot as plt

model = sys.argv[1]
program = sys.argv[2]

plots_dir = "plots"


fontsize = 20
markers = ["*", "o", "h", "v", "P", "s", "p", "x", "D", "8"]

ONLY_FA = model == "fa"

marker_cycle = cycle(markers)


COLORING_PROGRAM = "coloring"
DIJKSTRA_PROGRAM = "dijkstra_token_ring"
MAX_MATCHING_PROGRAM = "maximal_matching"

TITLE_PROGRAM_MAP = {
    COLORING_PROGRAM: "Graph Coloring",
    DIJKSTRA_PROGRAM: "Dijkstra Token Ring",
    MAX_MATCHING_PROGRAM: "Maximal Matching",
}

# program = COLORING_PROGRAM
# program = DIJKSTRA_PROGRAM
# program = MAX_MATCHING_PROGRAM


graphs = [
    # "star_graph_n7",
    #     # "star_graph_n15",
    "graph_powerlaw_cluster_graph_n7",
    #     # "graph_random_regular_graph_n7_d4",
    #     # "star_graph_n13",
    #     # "graph_powerlaw_cluster_graph_n8",
    #     # "graph_powerlaw_cluster_graph_n9",
    #     # "graph_random_regular_graph_n8_d4",
    #     # "graph_random_regular_graph_n9_d4",
]

# graphs = [
#     "implicit_graph_n6",
#     "implicit_graph_n7",
#     # "implicit_graph_n8",
#     # "implicit_graph_n9",
#     # "implicit_graph_n10",
#     # "implicit_graph_n11",
#     "implicit_graph_n12",
# ]


result_type = "cvf"


def get_title():
    result = " ".join(graph_name.split("_")).title()
    return f"{TITLE_PROGRAM_MAP[program]} - {result}"


def main(graph_name):
    filepath = os.path.join(
        "ml_predictions", f"{model}__{graph_name}__{result_type}.csv"
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

    plot_df(df, legends)


def plot_df(df, legends):
    plt.figure(figsize=(16, 8))
    ax = sns.lineplot(data=df, linewidth=3, markersize=10)

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

    file_name = f"RE__{program}__{graph_name}__{model}.png"
    custom_lines = [
        mlines.Line2D(
            [],
            [],
            color=line.get_color(),
            marker=marker,
            label=cat,
            linewidth=3,
            markersize=10,
        )
        for line, marker, cat in zip(ax.lines, markers, legends)
    ]

    plt.rc("font", size=fontsize)
    plt.legend(handles=custom_lines, fontsize=fontsize * 0.9)  # using a size in points
    plt.savefig(
        os.path.join(
            plots_dir,
            f"{file_name}",
        ),
        bbox_inches="tight",
    )
    plt.close()

    print(f"Saved plot(s) for plots/{file_name}")


if __name__ == "__main__":
    for graph_name in graphs:
        main(graph_name)
