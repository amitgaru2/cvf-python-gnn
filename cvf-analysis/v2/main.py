import os
import logging
import argparse

from custom_logger import logger
from dijkstra import DijkstraTokenRingCVFAnalysisV2
from graph_coloring import GraphColoringCVFAnalysisV2

ColoringProgram = "graph_coloring"
DijkstraProgram = "dijkstra_token_ring"
MaxMatchingProgram = "maximal_matching"
MaxIndependentSetProgram = "maximal_independent_set"
LinearRegressionProgram = "linear_regression"

AnalysisMap = {
    ColoringProgram: GraphColoringCVFAnalysisV2,
    DijkstraProgram: DijkstraTokenRingCVFAnalysisV2,
}

graphs_dir = os.path.join(
    os.getenv("CVF_PROJECT_DIR", "/home"), "cvf-analysis", "graphs"
)


def start(graphs_dir, graph_names):
    for graph_name in graph_names:
        logger.info('Locating Graph: "%s".', graph_name)
        full_path = os.path.join(graphs_dir, f"{graph_name}.txt")
        if not os.path.exists(full_path):
            logger.warning("Graph file: %s not found! Skipping the graph.", full_path)
            continue

        graph = {}
        with open(full_path, "r") as f:
            line = f.readline()
            while line:
                node_edges = [int(i) for i in line.split()]
                node = node_edges[0]
                edges = node_edges[1:]
                graph[node] = set(edges)
                line = f.readline()

        yield graph_name, graph


def main(
    graph_name,
    graph,
    program,
):
    CVFAnalysisKlass = AnalysisMap[program]
    cvf_analysis = CVFAnalysisKlass(graph_name, graph)
    cvf_analysis.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--program",
        choices=[
            ColoringProgram,
            DijkstraProgram,
            # MaxMatchingProgram,
            # MaxIndependentSetProgram,
            # LinearRegressionProgram,
        ],
        required=True,
    )
    parser.add_argument(
        "--graph-names",
        type=str,
        nargs="+",
        help="list of graph names in the 'graphs_dir' or list of number of nodes for implict graphs (if implicit program)",
        required=True,
    )
    parser.add_argument(
        "--logging",
        choices=[
            "INFO",
            "DEBUG",
        ],
        required=False,
    )
    args = parser.parse_args()
    if args.logging:
        logger.setLevel(getattr(logging, args.logging, "INFO"))

    for graph_name, graph in start(graphs_dir, args.graph_names):
        main(
            graph_name,
            graph,
            args.program,
        )


"""
python main.py --program graph_coloring --graph-names graph_1
python main.py --program dijkstra_token_ring --graph-names implicit_graph_n10
"""
