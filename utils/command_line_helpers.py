import os


ColoringProgram = "graph_coloring"
DijkstraProgram = "dijkstra_token_ring"
MaxMatchingProgram = "maximal_matching"
MaxIndependentSetProgram = "maximal_independent_set"
LinearRegressionProgram = "linear_regression"

PROGRAM_CHOICES = [
    ColoringProgram,
    DijkstraProgram,
    MaxMatchingProgram,
    MaxIndependentSetProgram,
    LinearRegressionProgram,
]


def get_graph(logger, graphs_dir, graph_names):
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
