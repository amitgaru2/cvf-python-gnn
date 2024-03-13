import os
import argparse

from graph_coloring import GraphColoringFullAnalysis, GraphColoringPartialAnalysis
from cvf_analysis import CVFAnalysis, logger, PartialAnalysisType, FullAnalysisType

ColoringProgram = "coloring"
DijkstraProgram = "dijkstra"
MaxMatchingProgram = "max_matching"

AnalysisMap = {
    ColoringProgram: {
        FullAnalysisType: GraphColoringFullAnalysis,
        PartialAnalysisType: GraphColoringPartialAnalysis,
    }
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
                node_edges = line.split()
                node = node_edges[0]
                edges = node_edges[1:]
                graph[node] = set(edges)
                line = f.readline()

        yield graph_name, graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--program", choices=[ColoringProgram, DijkstraProgram, MaxMatchingProgram]
    )  # coloring, dijkstra, max_matching
    parser.add_argument("-f", "--full-analysis", action="store_true")
    parser.add_argument(
        "--graph_names",
        type=str,
        nargs="+",
        help="list of graph names in the 'graphs_dir'",
    )
    args = parser.parse_args()
    print(args.program, args.full_analysis, args.graph_names)

    analysis_type = FullAnalysisType if args.full_analysis else PartialAnalysisType
    CVFAnalysisKlass: CVFAnalysis = AnalysisMap[args.program][analysis_type]
    logger.info("Analysis program : %s.", CVFAnalysisKlass.__name__)
    for graph_name, graph in start(CVFAnalysisKlass.graphs_dir, args.graph_names):
        analysis = CVFAnalysisKlass(graph_name, graph)
        analysis.start()


if __name__ == "__main__":
    main()
