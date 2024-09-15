import os
import csv
import copy
import math

import pandas as pd

from itertools import combinations

graphs_dir = "graphs"
results_dir = "results"


graph_name = "graph_powerlaw_cluster_graph_n9"

graph = {}
with open(os.path.join(graphs_dir, f"{graph_name}.txt"), "r") as f:
    line = f.readline()
    while line:
        node_edges = line.split()
        node = node_edges[0]        
        edges = node_edges[1:]
        graph[node] = set(edges)
        line = f.readline()



print(graph)


nodes = list(graph.keys())
node_positions = {v: i for i, v in enumerate(nodes)}

# max_degree = max(len(graph[n]) for n in nodes)

# degree_of_nodes = {n: max_degree for n in nodes}

degree_of_nodes = {n: len(graph[n]) for n in nodes}

print("Degree of all nodes (starting from 0):")
degree_of_nodes # start from 0

configurations = {
    tuple([0 for i in range(len(nodes))])
}
# perturb each state at a time for all states in configurations and accumulate the same in the configurations for next state to perturb
for n in nodes:
    node_pos = node_positions[n]
    config_copy = copy.deepcopy(configurations)
    for i in range(1, degree_of_nodes[n]+1):
        for cc in config_copy:
            cc = list(cc)
            cc[node_pos] = i
            configurations.add(tuple(cc))

print("No. of Configurations:", len(configurations))


invariants = set()
for state in configurations:
    all_paths = combinations(range(len(state)), 2)
    for src, dest in all_paths:
        src_node, dest_node = nodes[src], nodes[dest]
        src_color, dest_color = state[src], state[dest]
        if dest_node in graph[src_node] and src_color == dest_color:
            # found same color node between neighbors
            break
    else:
        invariants.add(state)

print("Invariants and Count of Invariants:")
len(invariants)


program_transitions_rank = {}
for inv in invariants:
    program_transitions_rank[inv] = {"L": 0, "C": 1, "A": 0, "Ar": 0, "M": 0}

def find_min_possible_color(colors):
    for i in range(len(colors)+1):
        if i not in colors:
            return i
        

def is_different_color(color, other_colors):
    """
    return True if "color" is different from all "other_colors"
    """
    for c in other_colors:
        if color == c:
            return False
    return True


def is_program_transition(perturb_pos, start_state, dest_state):
    if start_state in invariants and dest_state in invariants:
        return False

    node = nodes[perturb_pos]
    neighbor_pos = [node_positions[n] for n in graph[node]]
    neighbor_colors = set(dest_state[i] for i in neighbor_pos)
    min_color = find_min_possible_color(neighbor_colors)
    return dest_state[perturb_pos] == min_color


def get_program_transitions(start_state):
    program_transitions = set()
    for position, val in enumerate(start_state):
        # check if node already has different color among the neighbors => If yes => no need to perturb that node's value
        node = nodes[position]
        neighbor_pos = [node_positions[n] for n in graph[node]]
        neighbor_colors = set(start_state[i] for i in neighbor_pos)
        if is_different_color(val, neighbor_colors):
            continue
        
        # if the current node's color is not different among the neighbors => search for the program transitions possible
        possible_node_colors = set(range(degree_of_nodes[nodes[position]]+1))
        for perturb_val in possible_node_colors:
            perturb_state = list(start_state)
            perturb_state[position] = perturb_val
            perturb_state = tuple(perturb_state)
            if perturb_state != start_state:
                if is_program_transition(position, start_state, perturb_state):
                    program_transitions.add(perturb_state)
    return {"program_transitions": program_transitions}



def get_cvfs(start_state):
    cvfs_in = dict()
    cvfs_out = dict()
    for position, _ in enumerate(start_state):
        possible_node_colors = set(range(degree_of_nodes[nodes[position]]+1))
        for perturb_val in possible_node_colors:
            perturb_state = list(start_state)
            perturb_state[position] = perturb_val
            perturb_state = tuple(perturb_state)
            if perturb_state != start_state:
                if start_state in invariants:
                    cvfs_in[perturb_state] = position # track the nodes to calculate its overall rank effect
                else:
                    cvfs_out[perturb_state] = position
    return {"cvfs_in": cvfs_in, "cvfs_out": cvfs_out}


program_transitions_n_cvf = {}

for state in configurations:
    program_transitions_n_cvf[state] = {**get_program_transitions(state), **get_cvfs(state)}


unranked_states = set(program_transitions_n_cvf.keys()) - set(program_transitions_rank.keys())
print("Unranked states for Program transitions:", len(unranked_states))


# rank the states that has all the paths to the ranked one
while unranked_states:
    ranked_states = set(program_transitions_rank.keys())
    remove_from_unranked_states = set()
    for state in unranked_states:
        dests = program_transitions_n_cvf[state]['program_transitions']
        if dests - ranked_states:       # some desitnations states are yet to be ranked
            pass
        else:                           # all the destination has been ranked
            total_path_length = 0
            path_count = 0
            _max = 0
            for succ in dests:
                path_count += program_transitions_rank[succ]["C"]
                total_path_length += program_transitions_rank[succ]["L"] + program_transitions_rank[succ]["C"]
                _max = max(_max, program_transitions_rank[succ]["M"])
            program_transitions_rank[state] = {
                "L": total_path_length,
                "C": path_count,
                "A": total_path_length/path_count,
                "Ar": math.ceil(total_path_length/path_count),
                "M": _max + 1
            }
            remove_from_unranked_states.add(state)
    unranked_states -= remove_from_unranked_states


pt_rank_effect = {}

for state, pt_cvfs in program_transitions_n_cvf.items():
    for pt in pt_cvfs['program_transitions']:
        pt_rank_effect[(state, pt)] = {
            "Ar": program_transitions_rank[pt]["Ar"] - program_transitions_rank[state]["Ar"],
            "M": program_transitions_rank[pt]["M"] - program_transitions_rank[state]["M"]
        }


cvfs_in_rank_effect = {}
cvfs_out_rank_effect = {}

for state, pt_cvfs in program_transitions_n_cvf.items():
    for cvf, node in pt_cvfs['cvfs_in'].items():
        cvfs_in_rank_effect[(state, cvf)] = {
            "node": node,
            "Ar": program_transitions_rank[cvf]["Ar"] - program_transitions_rank[state]["Ar"],
            "M": program_transitions_rank[cvf]["M"] - program_transitions_rank[state]["M"]
        }
    for cvf, node in pt_cvfs['cvfs_out'].items():
        cvfs_out_rank_effect[(state, cvf)] = {
            "node": node,
            "Ar": program_transitions_rank[cvf]["Ar"] - program_transitions_rank[state]["Ar"],
            "M": program_transitions_rank[cvf]["M"] - program_transitions_rank[state]["M"]
        }


# Rank Effect count
        
pt_rank_effect_ = []
for state in pt_rank_effect:
    pt_rank_effect_.append({"state": state, **pt_rank_effect[state]})

pt_rank_effect_df = pd.DataFrame(pt_rank_effect_)


cvfs_in_rank_effect_ = []
for state in cvfs_in_rank_effect:
    cvfs_in_rank_effect_.append({"state": state, **cvfs_in_rank_effect[state]})
    
cvfs_in_rank_effect_df = pd.DataFrame(cvfs_in_rank_effect_)

cvfs_out_rank_effect_ = []
for state in cvfs_out_rank_effect:
    cvfs_out_rank_effect_.append({"state": state, **cvfs_out_rank_effect[state]})

cvfs_out_rank_effect_df = pd.DataFrame(cvfs_out_rank_effect_)


pt_avg_counts = pt_rank_effect_df['Ar'].value_counts()
pt_max_counts = pt_rank_effect_df['M'].value_counts()
cvf_in_avg_counts = cvfs_in_rank_effect_df['Ar'].value_counts()
cvf_in_max_counts = cvfs_in_rank_effect_df['M'].value_counts()
cvf_out_avg_counts = cvfs_out_rank_effect_df['Ar'].value_counts()
cvf_out_max_counts = cvfs_out_rank_effect_df['M'].value_counts()


fieldnames = ["Rank Effect", "PT (Max)", "PT (Avg)", "CVF In (Max)", "CVF In (Avg)", "CVF Out (Max)", "CVF Out (Avg)"]
with open(os.path.join(results_dir, f"rank_effect_{graph_name}.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for re in sorted(
        set(pt_avg_counts.index) |
        set(pt_max_counts.index) |
        set(cvf_in_avg_counts.index) |
        set(cvf_in_max_counts.index) |
        set(cvf_out_avg_counts.index) |
        set(cvf_out_max_counts.index)
    ):
        writer.writerow({
            "Rank Effect": re,
            "PT (Max)": pt_max_counts.get(re, 0),
            "PT (Avg)": pt_avg_counts.get(re, 0),
            "CVF In (Max)": cvf_in_max_counts.get(re, 0),
            "CVF In (Avg)": cvf_in_avg_counts.get(re, 0),
            "CVF Out (Max)": cvf_out_max_counts.get(re, 0),
            "CVF Out (Avg)": cvf_out_avg_counts.get(re, 0),
        })


# Rank Count
        
pt_rank_ = []
for state in program_transitions_rank:
    pt_rank_.append({"state": state, **program_transitions_rank[state]})

pt_rank_df = pd.DataFrame(pt_rank_)

pt_avg_counts = pt_rank_df['Ar'].value_counts()
pt_max_counts = pt_rank_df['M'].value_counts()

fieldnames = ["Rank", "Count (Max)", "Count (Avg)"]
with open(os.path.join(results_dir, f"rank_{graph_name}.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for rank in sorted(set(pt_avg_counts.index)|set(pt_max_counts.index)):
        writer.writerow({"Rank": rank, "Count (Max)": pt_max_counts.get(rank, 0), "Count (Avg)": pt_avg_counts.get(rank, 0)})


# Rank Effect of Individual Nodes

cvf_in_avg_counts_by_node = cvfs_in_rank_effect_df.groupby(['node', 'Ar'])['Ar'].count()
cvf_in_max_counts_by_node = cvfs_in_rank_effect_df.groupby(['node', 'M'])['M'].count()
cvf_out_avg_counts_by_node = cvfs_out_rank_effect_df.groupby(['node', 'Ar'])['Ar'].count()
cvf_out_max_counts_by_node = cvfs_out_rank_effect_df.groupby(['node', 'M'])['M'].count()


max_Ar = max(cvfs_in_rank_effect_df['Ar'].max(), cvfs_out_rank_effect_df['Ar'].max())
min_Ar = min(cvfs_in_rank_effect_df['Ar'].min(), cvfs_out_rank_effect_df['Ar'].min())

max_M = max(cvfs_in_rank_effect_df['M'].max(), cvfs_out_rank_effect_df['M'].max())
min_M = min(cvfs_in_rank_effect_df['M'].min(), cvfs_out_rank_effect_df['M'].min())

max_Ar_M = max(max_Ar, max_M)
min_Ar_M = min(min_Ar, min_M)

print(max_Ar, min_Ar, max_M, min_M, max_Ar_M, min_Ar_M)


# rank effect of individual node
fieldnames = ["Node", "Rank Effect", "CVF In (Max)", "CVF In (Avg)", "CVF Out (Max)", "CVF Out (Avg)"]
with open(os.path.join(results_dir, f"rank_effect_by_node_{graph_name}.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for node in nodes:
        for rank_effect in range(min_Ar_M, max_Ar_M+1):
            node_re = (node_positions[node], rank_effect)
            writer.writerow({
                "Node": node,
                "Rank Effect": rank_effect,
                "CVF In (Max)": cvf_in_max_counts_by_node.get(node_re, 0),
                "CVF In (Avg)": cvf_in_avg_counts_by_node.get(node_re, 0),
                "CVF Out (Max)": cvf_out_max_counts_by_node.get(node_re, 0),
                "CVF Out (Avg)": cvf_out_avg_counts_by_node.get(node_re, 0),
            })

