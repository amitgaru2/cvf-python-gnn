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
    # "graph_4",
    # "graph_5",
    # "test_lr_graph_6",
    # "implicit_graph_n5",
    # "star_graph_n10"
    # "graph_powerlaw_cluster_graph_n7"
    "graph_15"
]

planar = False
fontsize = 25

for gname in graph_names:
    G = nx.read_adjlist(f"graphs/{gname}.txt")
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    if planar:
        pos = nx.planar_layout(G)
        nx.draw_networkx(
            G,
            pos=pos,
            node_color="white",
            font_color="black",
            ax=fig.add_subplot(),
            font_size=60,
            node_size=5000,
        )
    else:
        nx.draw_networkx(
            G,
            node_color="white",
            font_color="black",
            ax=fig.add_subplot(),
            font_size=60,
            node_size=5000,
        )
    ax = plt.gca()  # to get the current axis
    ax.collections[0].set_edgecolor("#000000")

    fig.suptitle(f"Graph {" ".join(gname.split("_"))}", fontsize=fontsize)
    fig.savefig(f"graph_images/{gname}.png")
    # plt.show()
