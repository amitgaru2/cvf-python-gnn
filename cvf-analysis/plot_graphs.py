import networkx as nx

from matplotlib import pyplot as plt

graph_names = [
    # "graph_1",
    # "graph_2",
    # "graph_3",
    
    # "graph_6",
    # "graph_6b",
    # "graph_7",
    # "graph_8",

    "graph_4",
    "graph_5",
]

planar = True

for gname in graph_names:
    G = nx.read_adjlist(f"graphs/{gname}.txt")
    fig = plt.figure(figsize=(12, 5))
    if planar:
        pos = nx.planar_layout(G)
        nx.draw_networkx(G, pos=pos, ax=fig.add_subplot())
    else:
        nx.draw_networkx(G, ax=fig.add_subplot())

    fig.savefig(f"graph_images/{gname}.png")
    # plt.show()
