import os

from custom_logger import logger

GRAPHS_DIR = "graphs"


class Graph:
    def __init__(self, adjacency_dict, name):
        """
        Initializes the graph.

        Parameters:
        adjacency_dict (dict): A dictionary where keys are nodes and values are sets of neighboring nodes
        """
        if not isinstance(adjacency_dict, dict):
            raise TypeError("Input must be a dictionary")
        self.adj = adjacency_dict
        self.name = name

    def nodes(self):
        """Returns a list of nodes in the graph"""
        return list(self.adj.keys())

    def degree(self, node):
        """Returns the degree of a given node"""
        if node not in self.adj:
            raise ValueError(f"Node {node} does not exist in the graph")
        return len(self.adj[node])

    def number_of_nodes(self):
        """Returns the total number of nodes"""
        return len(self.adj)

    def neighbors(self, node):
        """
        Returns the neighbors of a given node

        Parameters:
        node: the node whose neighbors are requested

        Returns:
        set of neighboring nodes
        """
        if node not in self.adj:
            raise ValueError(f"Node {node} does not exist in the graph")
        return self.adj[node]

    def __str__(self):
        return f"Graph(name={self.name}, N={self.number_of_nodes()})"


def get_graph(graph_name) -> Graph | None:
    logger.debug('Locating Graph: "%s".', graph_name)
    full_path = os.path.join(GRAPHS_DIR, f"{graph_name}.txt")
    if not os.path.exists(full_path):
        logger.warning("Graph file: %s not found! Skipping the graph.", full_path)
        return None

    graph_dict = {}
    with open(full_path, "r") as f:
        line = f.readline()
        while line:
            node_edges = [int(i) for i in line.split()]
            node = node_edges[0]
            edges = node_edges[1:]
            graph_dict[node] = set(edges)
            line = f.readline()

    return Graph(graph_dict, name=graph_name)
