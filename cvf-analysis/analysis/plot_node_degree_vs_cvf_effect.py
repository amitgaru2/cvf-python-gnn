# TODO: Node id in X axis as well
import os
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


def get_graph(graph_name):
    full_path = os.path.join(graphs_dir, f"{graph_name}.txt")
    graph = {}
    with open(full_path, "r") as f:
        line = f.readline()
        while line:
            node_edges = [int(i) for i in line.split()]
            node = node_edges[0]
            edges = node_edges[1:]
            graph[node] = set(edges)
            line = f.readline()
    return graph


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
    graph = get_graph(graph_name)
    degree_of_nodes = {n: len(graph[n]) for n in graph}
    df["Node Degree"] = df.apply(lambda x: degree_of_nodes[x["Node"]], axis=1)
    return df


def create_plots_dir_if_not_exists():
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)


def plot_node_degree_vs_rank_effect(df, ax):
    sns.scatterplot(data=df, x="Node Degree", y="Rank Effect", ax=ax, s=500)
    rank_effect_max = df["Rank Effect"].max()
    ax.set_yticks([i for i in range(0, rank_effect_max + 1)])

if __name__ == "__main__":
    results_dir = os.path.join(os.pardir, "results")
    graphs_dir = os.path.join(os.pardir, "graphs")
    program = "coloring"  # coloring, dijkstra_token_ring, maximal_matching, maximal_independent_set
    analysis_type = "full"  # full, partial
    graph_names = [
        "graph_1",
        "graph_2",
        "graph_3",
        "graph_6",
        "graph_6b",
        "graph_7",
    ]
    plots_dir = os.path.join("plots", program, "node_degree_vs_cvf_effect")

    create_plots_dir_if_not_exists()

    for graph_name in graph_names:
        df = get_df(graph_name)
        if df is None:
            continue
        grps = df[(df["CVF (Avg)"] > 0) & (df["Rank Effect"] > 0)].groupby(
            ["Node Degree", "Rank Effect"]
        )
        data = grps.groups.keys()
        df = pd.DataFrame(data, columns=["Node Degree", "Rank Effect"])
        df["Node Degree"] = df["Node Degree"].astype("str")
        fig, ax = plt.subplots(1, figsize=(10, 5), constrained_layout=True)
        fig_title = (
            f"node_degree_vs_rank_effect__{analysis_type}__{program}__{graph_name}"
        )
        fig.suptitle(fig_title, fontsize=10)
        plot_node_degree_vs_rank_effect(df, ax)

        fig.savefig(
            os.path.join(
                plots_dir,
                f"{fig_title}.png",
            )
        )
