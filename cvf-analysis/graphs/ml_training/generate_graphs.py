import json
import networkx as nx


N = 7

graph_type = "cycle_graph"

G = getattr(nx, graph_type)(N, nx.Graph())

graph_name = f"{graph_type}" + f"_n{N}"

nx.write_adjlist(G, f"{graph_name}.txt", comments="#")
# print(G.edges)

json.dump(list(G.edges), open(f"{graph_name}_edge_index.json", "w"))
