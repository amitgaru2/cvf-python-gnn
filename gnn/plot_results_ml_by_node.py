import os
import sys
import pandas as pd
import seaborn as sns

from itertools import cycle
import matplotlib.lines as mlines
from matplotlib import pyplot as plt


model = sys.argv[1]
program = sys.argv[2]

markers = ["*", "o", "h", "v", "P", "s", "p", "x", "D", "8"]
marker_cycle = cycle(markers)

colors = [sns.cubehelix_palette(5, start=i, rot=0)[2:4] for i in range(5)]
colors = [("red", "red"), ("green", "green"), ("blue", "blue"), ("orange", "orange")]
color_cycle = cycle(colors)

plots_dir = "plots"

fontsize = 20


COLORING_PROGRAM = "coloring"
DIJKSTRA_PROGRAM = "dijkstra_token_ring"
MAX_MATCHING_PROGRAM = "maximal_matching"

TITLE_PROGRAM_MAP = {
    COLORING_PROGRAM: "Graph Coloring",
    DIJKSTRA_PROGRAM: "Dijkstra Token Ring",
    MAX_MATCHING_PROGRAM: "Maximal Matching",
}


graphs = [
    "star_graph_n7",
    #     # "star_graph_n15",
    #     # "graph_powerlaw_cluster_graph_n7",
    #     # "graph_random_regular_graph_n7_d4",
    #     # "star_graph_n13",
    #     # "graph_powerlaw_cluster_graph_n8",
    #     "graph_powerlaw_cluster_graph_n9",
    #     # "graph_random_regular_graph_n8_d4",
    #     # "graph_random_regular_graph_n9_d4",
]

graphs = [
# "implicit_graph_n6",
# "implicit_graph_n7",
# "implicit_graph_n8",
# "implicit_graph_n9",
# "implicit_graph_n10",
# "implicit_graph_n11",
    "implicit_graph_n12",
]


selected_nodes = [0, 2, 11]


result_type = "cvf_by_node"


def get_title(graph_name):
    result = " ".join(graph_name.split("_")).title()
    return f"{TITLE_PROGRAM_MAP[program]} - {result}"


def main(graph_name):
    filepath = os.path.join(
        "ml_predictions", f"{model}__{graph_name}__{result_type}.csv"
    )
    df = pd.read_csv(filepath, index_col=0)

    if "fa_count" in df.columns:
        df = df[["rank effect", "node", "ml_count", "fa_count"]]
        df = df.rename(columns={"ml_count": "ML count", "fa_count": "FA count"})
        lines_in_pair = 2
    else:
        df = df[["rank effect", "node", "ml_count"]]
        df = df.rename(columns={"ml_count": "ML count"})
        lines_in_pair = 1

    rank_effects = df["rank effect"].unique()
    rank_effects.sort()
    df_preproc = pd.DataFrame({"Rank Effect": rank_effects})

    nodes = df["node"].unique()
    nodes.sort()
    for node in nodes:
        col = f"Node {node} ML count"
        node_data = df.loc[(df["node"] == node)]["ML count"]
        node_data = node_data.reset_index(drop=True)
        df_preproc.loc[:, col] = node_data

        if "FA count" in df.columns:
            col = f"Node {node} FA count"
            node_data = df.loc[(df["node"] == node)]["FA count"]
            node_data = node_data.reset_index(drop=True)
            df_preproc.loc[:, col] = node_data

    selected_cols = ["Rank Effect"]
    for i in selected_nodes:
        temp = [f"Node {i} ML count"]
        if "FA count" in df.columns:
            temp.append(f"Node {i} FA count")
        selected_cols.extend(temp)

    df_preproc = df_preproc[selected_cols]
    df_preproc.set_index("Rank Effect", inplace=True)

    plot_df(df_preproc, selected_cols, graph_name, lines_in_pair)


def plot_df(df_preproc, selected_cols, graph_name, lines_in_pair):
    plt.figure(figsize=(16, 8))
    ax = sns.lineplot(data=df_preproc, linewidth=1, markersize=10)

    i = 0
    no_of_lines = len(ax.lines) // 2
    while i < no_of_lines:
        lines = ax.lines[i : i + lines_in_pair]
        colors = next(color_cycle)
        line_styles = ("solid", "dashed")
        for j, line in enumerate(lines):
            line.set_color(colors[j])
            line.set_linestyle(line_styles[j])
        i += lines_in_pair

    # prev_marker = None
    # next_color = None
    # for i, line in enumerate(ax.lines):
    #     if prev_marker is None:
    #         marker = next(marker_cycle)
    #         color, next_color = next(color_cycle)
    #         prev_marker = marker
    #         line_style = "solid"
    #     else:
    #         marker = prev_marker
    #         color = next_color
    #         prev_marker = None
    #         line_style = "dashed"
    #     # line.set_marker(marker)
    #     line.set_color(color)
    #     line.set_linestyle(line_style)

    ax.set_xlabel("Rank Effect")

    ax.set_yscale("log")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)

    ax.set_title(get_title(graph_name), fontdict={"fontsize": fontsize})

    file_name = f"RE_Node__{program}__{graph_name}__{''.join([str(i) for i in selected_nodes])}__{model}.png"

    dup_markers = []
    for marker in markers:
        dup_markers.extend([marker, marker])

    custom_lines = [
        mlines.Line2D(
            [],
            [],
            color=line.get_color(),
            # marker=marker,
            label=cat,
            linewidth=1,
            markersize=10,
            linestyle=line.get_linestyle(),
        )
        for line, marker, cat in zip(ax.lines, dup_markers, selected_cols[1:])
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
