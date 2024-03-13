import csv
import os
import math
import copy
import logging
import random

import pandas as pd

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

FullAnalysisType = "full"
PartialAnalysisType = "partial"


class Analysis:
    graphs_dir = "graphs"
    results_dir = "results"
    analysis_type = "full"
    results_prefix = ""

    def __init__(self, graph_name, graph) -> None:
        self.graph_name = graph_name
        self.graph = graph
        self.nodes = list(graph.keys())
        self.node_positions = {v: i for i, v in enumerate(self.nodes)}
        self.degree_of_nodes = {n: len(self.graph[n]) for n in self.nodes}
        self.configurations = set()
        self.invariants = set()
        self.pts_rank = dict()
        self.pts_n_cvfs = dict()
        self.pts_rank_effect = dict()
        self.cvfs_in_rank_effect = dict()
        self.cvfs_out_rank_effect = dict()
        self.cvfs_in_rank_effect_df = None
        self.cvfs_out_rank_effect_df = None

        self.create_results_dir_if_not_exists()

    def start(self):
        self._start()

    def _start(self):
        self._gen_configurations()
        self._find_invariants()
        self._init_pts_rank()
        self._find_program_transitions_n_cvf()
        self._rank_all_states()
        self._gen_save_rank_count()
        self._calculate_pts_rank_effect()
        self._calculate_cvfs_rank_effect()
        self._gen_save_rank_effect_count()
        self._gen_save_rank_effect_by_node_count()

    def create_results_dir_if_not_exists(self):
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def _gen_configurations(self):
        self.configurations = {tuple([0 for i in range(len(self.nodes))])}
        # perturb each state at a time for all states in configurations and accumulate the same in the configurations for next state to perturb
        for n in self.nodes:
            node_pos = self.node_positions[n]
            config_copy = copy.deepcopy(self.configurations)
            for i in range(1, self.degree_of_nodes[n] + 1):
                for cc in config_copy:
                    cc = list(cc)
                    cc[node_pos] = i
                    self.configurations.add(tuple(cc))

        logger.info("No. of Configurations: %s", len(self.configurations))

    def _find_invariants(self):
        raise NotImplemented

    def _init_pts_rank(self):
        for inv in self.invariants:
            self.pts_rank[inv] = {"L": 0, "C": 1, "A": 0, "Ar": 0, "M": 0}

    def _is_program_transition(self, perturb_pos, start_state, dest_state) -> bool:
        raise NotImplemented

    def _get_program_transitions(self, start_state):
        raise NotImplemented

    def _get_cvfs(self, start_state):
        raise NotImplemented

    def _find_program_transitions_n_cvf(self):
        for state in self.configurations:
            self.pts_n_cvfs[state] = {
                **self._get_program_transitions(state),
                **self._get_cvfs(state),
            }

    def _rank_all_states(self):
        unranked_states = set(self.pts_n_cvfs.keys()) - set(self.pts_rank.keys())
        logger.info("No. of Unranked states: %s", len(unranked_states))

        # rank the states that has all the paths to the ranked one
        while unranked_states:
            ranked_states = set(self.pts_rank.keys())
            remove_from_unranked_states = set()
            for state in unranked_states:
                dests = self.pts_n_cvfs[state]["program_transitions"]
                if (
                    dests - ranked_states
                ):  # some desitnations states are yet to be ranked
                    pass
                else:  # all the destination has been ranked
                    total_path_length = 0
                    path_count = 0
                    _max = 0
                    for succ in dests:
                        path_count += self.pts_rank[succ]["C"]
                        total_path_length += (
                            self.pts_rank[succ]["L"] + self.pts_rank[succ]["C"]
                        )
                        _max = max(_max, self.pts_rank[succ]["M"])
                    self.pts_rank[state] = {
                        "L": total_path_length,
                        "C": path_count,
                        "A": total_path_length / path_count,
                        "Ar": math.ceil(total_path_length / path_count),
                        "M": _max + 1,
                    }
                    remove_from_unranked_states.add(state)
            unranked_states -= remove_from_unranked_states

    def _calculate_pts_rank_effect(self):
        for state, pt_cvfs in self.pts_n_cvfs.items():
            for pt in pt_cvfs["program_transitions"]:
                self.pts_rank_effect[(state, pt)] = {
                    "Ar": self.pts_rank[pt]["Ar"] - self.pts_rank[state]["Ar"],
                    "M": self.pts_rank[pt]["M"] - self.pts_rank[state]["M"],
                }

    def _calculate_cvfs_rank_effect(self):
        for state, pt_cvfs in self.pts_n_cvfs.items():
            for cvf, node in pt_cvfs["cvfs_in"].items():
                self.cvfs_in_rank_effect[(state, cvf)] = {
                    "node": node,
                    "Ar": self.pts_rank[cvf]["Ar"] - self.pts_rank[state]["Ar"],
                    "M": self.pts_rank[cvf]["M"] - self.pts_rank[state]["M"],
                }
            for cvf, node in pt_cvfs["cvfs_out"].items():
                self.cvfs_out_rank_effect[(state, cvf)] = {
                    "node": node,
                    "Ar": self.pts_rank[cvf]["Ar"] - self.pts_rank[state]["Ar"],
                    "M": self.pts_rank[cvf]["M"] - self.pts_rank[state]["M"],
                }

    def _gen_save_rank_count(self):
        pt_rank_ = []
        for state in self.pts_rank:
            pt_rank_.append({"state": state, **self.pts_rank[state]})

        pt_rank_df = pd.DataFrame(pt_rank_)

        pt_avg_counts = pt_rank_df["Ar"].value_counts()
        pt_max_counts = pt_rank_df["M"].value_counts()

        fieldnames = ["Rank", "Count (Max)", "Count (Avg)"]
        with open(
            os.path.join(
                self.results_dir,
                f"rank__{self.analysis_type}__{self.results_prefix}__{self.graph_name}.csv",
            ),
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for rank in sorted(set(pt_avg_counts.index) | set(pt_max_counts.index)):
                writer.writerow(
                    {
                        "Rank": rank,
                        "Count (Max)": pt_max_counts.get(rank, 0),
                        "Count (Avg)": pt_avg_counts.get(rank, 0),
                    }
                )

    def _gen_save_rank_effect_count(self):
        pt_rank_effect_ = []
        for state in self.pts_rank_effect:
            pt_rank_effect_.append({"state": state, **self.pts_rank_effect[state]})

        pt_rank_effect_df = pd.DataFrame(pt_rank_effect_)

        cvfs_in_rank_effect_ = []
        for state in self.cvfs_in_rank_effect:
            cvfs_in_rank_effect_.append(
                {"state": state, **self.cvfs_in_rank_effect[state]}
            )

        self.cvfs_in_rank_effect_df = pd.DataFrame(cvfs_in_rank_effect_)

        cvfs_out_rank_effect_ = []
        for state in self.cvfs_out_rank_effect:
            cvfs_out_rank_effect_.append(
                {"state": state, **self.cvfs_out_rank_effect[state]}
            )

        self.cvfs_out_rank_effect_df = pd.DataFrame(cvfs_out_rank_effect_)

        pt_avg_counts = pt_rank_effect_df["Ar"].value_counts()
        pt_max_counts = pt_rank_effect_df["M"].value_counts()
        cvf_in_avg_counts = self.cvfs_in_rank_effect_df["Ar"].value_counts()
        cvf_in_max_counts = self.cvfs_in_rank_effect_df["M"].value_counts()
        cvf_out_avg_counts = self.cvfs_out_rank_effect_df["Ar"].value_counts()
        cvf_out_max_counts = self.cvfs_out_rank_effect_df["M"].value_counts()

        fieldnames = [
            "Rank Effect",
            "PT (Max)",
            "PT (Avg)",
            "CVF In (Max)",
            "CVF In (Avg)",
            "CVF Out (Max)",
            "CVF Out (Avg)",
        ]
        with open(
            os.path.join(
                self.results_dir,
                f"rank_effect__{self.analysis_type}__{self.results_prefix}__{self.graph_name}.csv",
            ),
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for re in sorted(
                set(pt_avg_counts.index)
                | set(pt_max_counts.index)
                | set(cvf_in_avg_counts.index)
                | set(cvf_in_max_counts.index)
                | set(cvf_out_avg_counts.index)
                | set(cvf_out_max_counts.index)
            ):
                writer.writerow(
                    {
                        "Rank Effect": re,
                        "PT (Max)": pt_max_counts.get(re, 0),
                        "PT (Avg)": pt_avg_counts.get(re, 0),
                        "CVF In (Max)": cvf_in_max_counts.get(re, 0),
                        "CVF In (Avg)": cvf_in_avg_counts.get(re, 0),
                        "CVF Out (Max)": cvf_out_max_counts.get(re, 0),
                        "CVF Out (Avg)": cvf_out_avg_counts.get(re, 0),
                    }
                )

    def _gen_save_rank_effect_by_node_count(self):
        cvf_in_avg_counts_by_node = self.cvfs_in_rank_effect_df.groupby(["node", "Ar"])[
            "Ar"
        ].count()
        cvf_in_max_counts_by_node = self.cvfs_in_rank_effect_df.groupby(["node", "M"])[
            "M"
        ].count()
        cvf_out_avg_counts_by_node = self.cvfs_out_rank_effect_df.groupby(
            ["node", "Ar"]
        )["Ar"].count()
        cvf_out_max_counts_by_node = self.cvfs_out_rank_effect_df.groupby(
            ["node", "M"]
        )["M"].count()

        max_Ar = max(
            self.cvfs_in_rank_effect_df["Ar"].max(),
            self.cvfs_out_rank_effect_df["Ar"].max(),
        )
        min_Ar = min(
            self.cvfs_in_rank_effect_df["Ar"].min(),
            self.cvfs_out_rank_effect_df["Ar"].min(),
        )

        max_M = max(
            self.cvfs_in_rank_effect_df["M"].max(),
            self.cvfs_out_rank_effect_df["M"].max(),
        )
        min_M = min(
            self.cvfs_in_rank_effect_df["M"].min(),
            self.cvfs_out_rank_effect_df["M"].min(),
        )

        max_Ar_M = max(max_Ar, max_M)
        min_Ar_M = min(min_Ar, min_M)

        # rank effect of individual node
        fieldnames = [
            "Node",
            "Rank Effect",
            "CVF In (Max)",
            "CVF In (Avg)",
            "CVF Out (Max)",
            "CVF Out (Avg)",
        ]
        with open(
            os.path.join(
                self.results_dir,
                f"rank_effect_by_node__{self.analysis_type}__{self.results_prefix}__{self.graph_name}.csv",
            ),
            "w",
            newline="",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for node in self.nodes:
                for rank_effect in range(min_Ar_M, max_Ar_M + 1):
                    node_re = (self.node_positions[node], rank_effect)
                    writer.writerow(
                        {
                            "Node": node,
                            "Rank Effect": rank_effect,
                            "CVF In (Max)": cvf_in_max_counts_by_node.get(node_re, 0),
                            "CVF In (Avg)": cvf_in_avg_counts_by_node.get(node_re, 0),
                            "CVF Out (Max)": cvf_out_max_counts_by_node.get(node_re, 0),
                            "CVF Out (Avg)": cvf_out_avg_counts_by_node.get(node_re, 0),
                        }
                    )


class PartialAnalysisMixin:
    analysis_type = "partial"
    K_sampling = 100

    @staticmethod
    def generate_random_samples(population, k):
        N = copy.deepcopy(population)
        random.shuffle(N)
        indx = 0
        samples = []
        while k > 0 and N:
            sampled_indx = random.randint(0, len(N[indx]) - 1)
            samples.append(N[indx].pop(sampled_indx))

            if not N[indx]:  # all elements popped for this list
                N.pop(indx)
                if N:
                    indx = indx % len(N)
            else:
                indx = (indx + 1) % len(N)

            if indx == 0 and len(N) > 1:
                random.shuffle(N)

            k -= 1

        return samples
