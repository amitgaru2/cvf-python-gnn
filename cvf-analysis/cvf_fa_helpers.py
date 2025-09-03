import numpy as np


def get_graph(graph_full_path):
    graph = {}
    with open(graph_full_path, "r") as f:
        line = f.readline()
        while line:
            node_edges = [int(i) for i in line.split()]
            node = node_edges[0]
            edges = node_edges[1:]
            graph[node] = set(edges)
            line = f.readline()

    return graph


def get_graph_stats(graph):
    no_nodes = len(graph)
    no_edges = sum([len(e) for k, e in graph.items()])
    return no_nodes, no_edges
