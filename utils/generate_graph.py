import networkx as nx


def main():
    G = nx.powerlaw_cluster_graph(10)
    print(G)


if __name__ == "__main__":
    main()
