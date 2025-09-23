import os
import sys

import pandas as pd
import networkx as nx

CVF_PROJECT_DIR = os.getenv("CVF_PROJECT_DIR", "")
utils_path = os.path.join(CVF_PROJECT_DIR, "utils")
sys.path.append(utils_path)

from custom_logger import logger
from command_line_helpers import GRAPHS_DIR


PAGE_RANK_ALPHA = 0.85
KATZ_ALPHA = 0.1
KATZ_BETA = 1.0


def get_nx_graph(graph_name):
    G = nx.Graph()

    # open file and read line by line
    with open(os.path.join(GRAPHS_DIR, f"{graph_name}.txt"), "r") as f:
        for line in f:
            numbers = list(map(int, line.strip().split()))
            node = numbers[0]  # first number is the node
            neighbors = numbers[1:]  # rest are neighbors

            for neighbor in neighbors:
                G.add_edge(node, neighbor)

    return G


def save_to_file(graph_name, results):
    filename = os.path.join(
        CVF_PROJECT_DIR,
        "cvf-analysis",
        "graphs",
        "centrality_scores",
        f"{graph_name}_centrality_scores.csv",
    )

    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)


def main(graph_name):
    G = get_nx_graph(graph_name)
    nodes = list(nx.nodes(G))

    degree = dict(nx.degree(G))

    degree_centr = nx.degree_centrality(G)

    closeness_centr = nx.closeness_centrality(G)

    betweenness_centr = nx.betweenness_centrality(G)

    eigenvector_centr = nx.eigenvector_centrality(G)

    try:
        pagerank_centr = nx.pagerank(G, alpha=PAGE_RANK_ALPHA)
    except Exception:
        pagerank_centr = {i: "N/A" for i in nodes}

    katz_centr = nx.katz_centrality(G, alpha=KATZ_ALPHA, beta=KATZ_BETA)

    harmonic_centr = nx.harmonic_centrality(G)

    subgraph_centr = nx.subgraph_centrality(G)

    communicability_betweenness_centr = nx.communicability_betweenness_centrality(G)

    results = {
        "degree": degree,
        "degree_centr": degree_centr,
        f"page_rank_alpha{PAGE_RANK_ALPHA}_centr": pagerank_centr,
        "betweenness_centr": betweenness_centr,
        "closeness_centr": closeness_centr,
        "eigenvector_centr": eigenvector_centr,
        f"katz_alpha{KATZ_ALPHA}_beta{KATZ_BETA}_centr": katz_centr,
        "harmonic_centr": harmonic_centr,
        "subgraph_centr": subgraph_centr,
        "communicability_betweenness_centr": communicability_betweenness_centr,
    }

    data = {"nodes": sorted(nodes)}
    data.update({k: [iv for ik, iv in sorted(v.items())] for k, v in results.items()})
    save_to_file(graph_name, data)


if __name__ == "__main__":
    main(sys.argv[1])
