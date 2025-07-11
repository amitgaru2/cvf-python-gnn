"""
CLI program to execute the CVF full analysis.
"""

import os
import sys
import logging
import argparse

from custom_logger import logger
from dijkstra import DijkstraTokenRingCVFAnalysisV2
from graph_coloring import GraphColoringCVFAnalysisV2
from maximal_matching import MaximalMatchingCVFAnalysisV2
from linear_regression import LinearRegressionCVFAnalysisV2

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)

from command_line_helpers import (
    get_graph,
    ColoringProgram,
    DijkstraProgram,
    MaxMatchingProgram,
    LinearRegressionProgram,
)


AnalysisMap = {
    ColoringProgram: GraphColoringCVFAnalysisV2,
    DijkstraProgram: DijkstraTokenRingCVFAnalysisV2,
    MaxMatchingProgram: MaximalMatchingCVFAnalysisV2,
    LinearRegressionProgram: LinearRegressionCVFAnalysisV2,
}

graphs_dir = os.path.join(
    os.getenv("CVF_PROJECT_DIR", "/home"), "cvf-analysis", "graphs"
)


def main(
    graph_name,
    graph,
    program,
    generate_data_ml,
    generate_data_emb,
    generate_test_data_ml,
):
    CVFAnalysisKlass = AnalysisMap[program]
    cvf_analysis = CVFAnalysisKlass(
        graph_name, graph, generate_data_ml, generate_data_emb, generate_test_data_ml
    )
    cvf_analysis.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--program",
        choices=[
            ColoringProgram,
            DijkstraProgram,
            MaxMatchingProgram,
            LinearRegressionProgram,
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
    parser.add_argument("-ml", "--generate-data-ml", action="store_true")
    parser.add_argument("-emb", "--generate-data-emb", action="store_true")
    parser.add_argument("-test-ml", "--generate-test-data-ml", action="store_true")
    args = parser.parse_args()
    if args.logging:
        logger.setLevel(getattr(logging, args.logging, "INFO"))

    for graph_name, graph in get_graph(logger, graphs_dir, args.graph_names):
        main(
            graph_name,
            graph,
            args.program,
            args.generate_data_ml,
            args.generate_data_emb,
            args.generate_test_data_ml,
        )


"""
python main.py --program graph_coloring --graph-names graph_1
python main.py --program dijkstra_token_ring --graph-names implicit_graph_n10
"""
