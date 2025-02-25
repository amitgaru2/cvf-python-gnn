import pydot
import json
import networkx as nx
import matplotlib.pyplot as plt

from networkx.drawing.nx_pydot import graphviz_layout


def get_color_map(G):
    color_map = []
    for node in G:
        if node in invariants:
            color_map.append("green")
        else:
            color_map.append("blue")
    return color_map


def draw_graph(G):
    nx.draw(
        G,
        with_labels=True,
        node_color=get_color_map(G),
        font_color="white",
        font_weight="bold",
        node_size=3000,
    )


invariants = {
    (0.9, 0.9, 0.9),
}

paths = json.load(open("output.json", "r"))

G = nx.Graph()
G.add_nodes_from(paths)


for k, v in paths.items():
    for iv in v:
        G.add_edge(k, tuple(iv))

draw_graph(G)

plt.show()
