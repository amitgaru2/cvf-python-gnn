import os
import sys
import csv
import copy
import math

import pandas as pd

graphs_dir = "graphs"
results_dir = "results"

graph_name = sys.argv[1]

graph = {}
with open(os.path.join(graphs_dir, f"{graph_name}.txt"), "r") as f:
    line = f.readline()
    while line:
        node_edges = line.split()
        node = node_edges[0]
        edges = node_edges[1:]
        # graph[node] = set(edges)
        graph[node] = edges
        line = f.readline()


nodes = list(graph.keys())
node_positions = {v: i for i, v in enumerate(nodes)}

graph_based_on_indx = {}
for k, v in graph.items():
    graph_based_on_indx[node_positions[k]] = []
    for iv in v:
        graph_based_on_indx[node_positions[k]].append(node_positions[iv])

degree_of_nodes = {n: len(graph[n]) for n in nodes}


class Configuration:
    def __init__(self, p=None, m=False):
        self._p = p
        self._m = m

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, val):
        self._p = val

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, val):
        self._m = val

    def __eq__(self, other):
        return self.p == other.p and self.m == other.m

    def __hash__(self):
        return hash((self.p, self.m))
    
    def __repr__(self):
        return f"<p: {self.p}, m: {self.m}>"
    

def possible_values_in_node_inc_null(node_pos):
    return set([None]+[node_positions[nb] for nb in graph[nodes[node_pos]]])


configurations = {
    tuple([Configuration(p=None, m=False) for i in range(len(nodes))])
}
# perturb each state at a time for all states in configurations and accumulate the same in the configurations for next state to perturb
for i, n in enumerate(nodes):
    node_pos = node_positions[n]
    config_copy = copy.deepcopy(configurations)
    for val in possible_values_in_node_inc_null(node_pos):
        for cc in config_copy:
            cc = list(cc)
            cc[node_pos] = Configuration(p=val, m=False)
            configurations.add(tuple(cc))
            cc[node_pos] = Configuration(p=val, m=True)
            configurations.add(tuple(cc))


print("All possible configurations:", len(configurations))


def check_if_invariant(state):
    def _pr_married(j, config):
        for i in graph_based_on_indx[j]:
            if state[i].p == j and config.p == i:
                return True
        return False

    for j, config in enumerate(state):
        # update m.j
        if config.m != _pr_married(j, config):
            return False

        # accept a proposal
        if config.m == _pr_married(j, config):
            if config.p is None:
                for i in graph_based_on_indx[j]:
                    if state[i].p == j:
                        return False
    
                for k in graph_based_on_indx[j]:
                    if state[k].p is None and k < j and not state[k].m:
                        return False
            else:
                i = config.p
                if state[i].p != j and ( state[i].m or j <= i ):
                    return False

    return True


invariants = set()
for state in configurations:
    # mm specifilc
    if check_if_invariant(state):
        invariants.add(state)

print("Invariants and Count of Invariants:", len(invariants))

program_transitions_rank = {}
for inv in invariants:
    program_transitions_rank[inv] = {"L": 0, "C": 1, "A": 0, "Ar": 0, "M": 0}


def is_program_transition(perturb_pos, start_state, dest_state):
    j = perturb_pos
    state = start_state
    config = state[perturb_pos]
    dest_config = dest_state[perturb_pos]

    def _pr_married(j, config):
        for i in graph_based_on_indx[j]:
            if state[i].p == j and config.p == i:
                return True
        return False

    # update m.j
    if start_state[perturb_pos].m != _pr_married(j, config):
        if dest_config.m == _pr_married(j, config):
            return True

    # accept a proposal
    if config.m == _pr_married(j, config):
        if config.p is None:
            for i in graph_based_on_indx[j]:
                if state[i].p == j and dest_config.p == i:
                    return True

            # make a proposal
            for i in graph_based_on_indx[j]:
                if state[i].p == j:
                    break
            else:
                max_k = -1
                for k in graph_based_on_indx[j]:
                    if state[k].p is None and k < j and not state[k].m:
                        if k > max_k:
                            max_k = k
    
                if max_k >= 0 and dest_config.p == max_k:
                    return True
        else:
            # withdraw a proposal
            i = config.p
            if state[i].p != j and ( state[i].m or j <= i ):
                if dest_config.p is None:
                    return True

    return False


def get_program_transitions(start_state):
    # dijkstra specific
    program_transitions = set()
    for position, _ in enumerate(start_state):
        possible_config_p_val = possible_values_in_node_inc_null(position) - {start_state[position].p}
        for perturb_p_val in possible_config_p_val:
            perturb_state = list(copy.deepcopy(start_state))
            perturb_state[position].p = perturb_p_val
            perturb_state = tuple(perturb_state)
            if is_program_transition(position, start_state, perturb_state):
                program_transitions.add(perturb_state)

        possible_config_m_val = {True, False} - {start_state[position].m}
        for perturb_m_val in possible_config_m_val:
            perturb_state = list(copy.deepcopy(start_state))
            perturb_state[position].m = perturb_m_val
            perturb_state = tuple(perturb_state)
            if is_program_transition(position, start_state, perturb_state):
                program_transitions.add(perturb_state)

    return {"program_transitions": program_transitions}


def evaluate_perturbed_pr_married(position, state):
    results = [False]
    config = state[position]

    if config.p is None:
        return results

    for nbr in graph_based_on_indx[position]:
        if nbr == config.p:
            results.append(True)
            return results

    return results


def get_cvfs(start_state):
    cvfs_in = dict()
    cvfs_out = dict()

    def _add_to_cvf(perturb_state, position):
        if start_state in invariants:
            cvfs_in[perturb_state] = position
        else:
            cvfs_out[perturb_state] = position

    for position, _ in enumerate(start_state):
        config = start_state[position]
        for a_pr_married_value in evaluate_perturbed_pr_married(position, start_state):
            if config.m is not a_pr_married_value:
                perturb_state = copy.deepcopy(start_state)
                perturb_state[position].m = a_pr_married_value
                _add_to_cvf(perturb_state, position)
            else:
                if config.p is None:
                    for nbr in graph_based_on_indx[position]:
                        perturb_state = copy.deepcopy(start_state)
                        perturb_state[position].p = nbr
                        perturb_state[position].m = a_pr_married_value
                        _add_to_cvf(perturb_state, position)
                else:
                    perturb_state = copy.deepcopy(start_state)
                    perturb_state[position].p = None
                    perturb_state[position].m = a_pr_married_value
                    _add_to_cvf(perturb_state, position)

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

# Rank Effect Count
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

