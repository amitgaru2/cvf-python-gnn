import os
import logging
import argparse

from cvf_analysis import (
    CVFAnalysis,
    logger,
    PartialAnalysisType,
    FullAnalysisType,
    PartialCVFAnalysisMixin,
)
from graph_coloring import GraphColoringFullAnalysis, GraphColoringPartialAnalysis
from maximal_matching import MaximalMatchingFullAnalysis, MaximalMatchingPartialAnalysis
from dijkstra_token_ring import (
    DijkstraTokenRingFullAnalysis,
    DijkstraTokenRingPartialAnalysis,
)
from maximal_independent_set import (
    MaximalSetIndependenceFullAnalysis,
    MaximalSetIndependencePartialAnalysis,
)
from linear_regression import (
    LinearRegressionFullAnalysis,
    LinearRegressionPartialAnalysis,
)


ColoringProgram = "graph_coloring"
DijkstraProgram = "dijkstra_token_ring"
MaxMatchingProgram = "maximal_matching"
MaxIndependentSetProgram = "maximal_independent_set"
LinearRegressionProgram = "linear_regression"

AnalysisMap = {
    ColoringProgram: {
        FullAnalysisType: GraphColoringFullAnalysis,
        PartialAnalysisType: GraphColoringPartialAnalysis,
    },
    DijkstraProgram: {
        FullAnalysisType: DijkstraTokenRingFullAnalysis,
        PartialAnalysisType: DijkstraTokenRingPartialAnalysis,
    },
    MaxMatchingProgram: {
        FullAnalysisType: MaximalMatchingFullAnalysis,
        PartialAnalysisType: MaximalMatchingPartialAnalysis,
    },
    MaxIndependentSetProgram: {
        FullAnalysisType: MaximalSetIndependenceFullAnalysis,
        PartialAnalysisType: MaximalSetIndependencePartialAnalysis,
    },
    LinearRegressionProgram: {
        FullAnalysisType: LinearRegressionFullAnalysis,
        PartialAnalysisType: LinearRegressionPartialAnalysis,
    },
}


def start(graphs_dir, graph_names):
    for graph_name in graph_names:
        logger.info('Started for Graph: "%s".', graph_name)
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


def set_sample_size(analysis: PartialCVFAnalysisMixin, sample_size: int):
    analysis.sample_size = sample_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--program",
        choices=[
            ColoringProgram,
            DijkstraProgram,
            MaxMatchingProgram,
            MaxIndependentSetProgram,
            LinearRegressionProgram,
        ],
        required=True,
    )  # coloring, dijkstra, max_matching
    parser.add_argument("-f", "--full-analysis", action="store_true")
    parser.add_argument("--sample-size", type=int, required=False)
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
    parser.add_argument(
        "--config-file",
        required=False,
    )
    args = parser.parse_args()
    if args.logging:
        logger.setLevel(getattr(logging, args.logging, "INFO"))
    # print(args.program, args.full_analysis, args.graph_names, args.sample_size)

    analysis_type = FullAnalysisType if args.full_analysis else PartialAnalysisType
    CVFAnalysisKlass: CVFAnalysis = AnalysisMap[args.program][analysis_type]
    logger.info("Analysis program : %s.", CVFAnalysisKlass.__name__)
    if args.program == DijkstraProgram:
        for no_nodes in args.graph_names:
            no_nodes = int(no_nodes)
            graph_name = f"implicit_graph_n{no_nodes}"
            logger.info('Started for Graph: "%s".', graph_name)
            graph = CVFAnalysisKlass.gen_implicit_graph(no_nodes)
            analysis = CVFAnalysisKlass(graph_name, graph)
            if analysis_type == PartialAnalysisType and args.sample_size:
                set_sample_size(analysis, args.sample_size)
            analysis.start()
    elif args.program == LinearRegressionProgram:
        if not args.config_file:
            logger.error("You need to provide config file for Linear Regression!")
            exit(1)
        for graph_name, graph in start(CVFAnalysisKlass.graphs_dir, args.graph_names):
            analysis = CVFAnalysisKlass(graph_name, graph, args.config_file)
            if analysis_type == PartialAnalysisType and args.sample_size:
                set_sample_size(analysis, args.sample_size)
            analysis.start()
    else:
        for graph_name, graph in start(CVFAnalysisKlass.graphs_dir, args.graph_names):
            analysis = CVFAnalysisKlass(graph_name, graph)
            if analysis_type == PartialAnalysisType and args.sample_size:
                set_sample_size(analysis, args.sample_size)
            analysis.start()


if __name__ == "__main__":
    main()
