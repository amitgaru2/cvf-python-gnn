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


args = generate_parser(takes_model=True, takes_nodes=True)

model = args.model
program = args.program
graph_names = args.graph_names
selected_nodes = args.nodes
selected_nodes.sort()

plots_dir = os.path.join("plots", program)
create_dir_if_not_exists(plots_dir)

ONLY_FA = model == "fa"


markers = ["*", "o", "h", "v", "P", "s", "p", "x", "D", "8"]
marker_cycle = cycle(markers)

# colors = [sns.cubehelix_palette(5, start=i, rot=0)[2:4] for i in range(5)]
colors = [("red", "red"), ("green", "green"), ("blue", "blue"), ("orange", "orange")]
color_cycle = cycle(colors)


fontsize = 20

COLORING_PROGRAM = "coloring"
DIJKSTRA_PROGRAM = "dijkstra_token_ring"
MAX_MATCHING_PROGRAM = "maximal_matching"

TITLE_PROGRAM_MAP = {
    COLORING_PROGRAM: "Graph Coloring",
    DIJKSTRA_PROGRAM: "Dijkstra Token Ring",
    MAX_MATCHING_PROGRAM: "Maximal Matching",
}


result_type = "cvf_by_node"


def get_title(graph_name):
    result = " ".join(graph_name.split("_")).title()
    return f"{TITLE_PROGRAM_MAP[program]} - {result}"


def main(graph_name):
    filepath = os.path.join(
        "ml_predictions", f"{model}__{program}__{graph_name}__{result_type}.csv"
    )
    df = pd.read_csv(filepath, index_col=0)

    if ONLY_FA:
        df = df[["rank effect", "node", "fa_count"]]
        df = df.rename(columns={"fa_count": "FA count"})
        lines_in_pair = 1
    elif "fa_count" in df.columns:
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
        if not ONLY_FA:
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
        temp = [f"Node {i} ML count"] if not ONLY_FA else []
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

    ax.set_xlabel("Rank Effect")

    ax.set_yscale("log")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", labelsize=fontsize)
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.xaxis.label.set_size(fontsize)
    ax.yaxis.label.set_size(fontsize)

    ax.set_title(get_title(graph_name), fontdict={"fontsize": fontsize})

    file_name = f"RE_Node__{model}__{program}__{graph_name}__{''.join([str(i) for i in selected_nodes])}__{model}.png"

    custom_lines = [
        mlines.Line2D(
            [],
            [],
            color=line.get_color(),
            label=cat,
            linewidth=1,
            markersize=10,
            linestyle=line.get_linestyle(),
        )
        for line, cat in zip(ax.lines, selected_cols[1:])
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

    print(f"Saved plot(s) for %s" % filepath)


if __name__ == "__main__":
    for graph_name in graph_names:
        main(graph_name)
