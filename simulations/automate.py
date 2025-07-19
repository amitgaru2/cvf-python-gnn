import os
import sys
import csv
import itertools
import subprocess

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)

from common_helpers import create_dir_if_not_exists
from command_line_helpers import (
    get_graph,
)

# PROGRAM = "dijkstra_token_ring"
# PROGRAM = "graph_coloring"
PROGRAM = "maximal_matching"

GRAPH_NAMES = ("graph_2_node",)
GRAPH = next(get_graph(GRAPH_NAMES))[1]

EDGES = []
for src, dests in GRAPH.items():
    EDGES.extend([(src, dest) for dest in dests])  # (src, dest) src being read by dest

max_size = len(EDGES) // 2

N = "1000000"
FI = ("5", "5")
LIMIT_STEPS = "100"
HIST_SIZE = "5"

results_dir = "automation_results"
agg_file = f"{PROGRAM}__{GRAPH_NAMES[0]}__N{N}__FI{"-".join(FI)}__L{LIMIT_STEPS}__H{HIST_SIZE}__steps__agg.csv"
agg_file = os.path.join(results_dir, agg_file)
create_dir_if_not_exists(results_dir)
if os.path.exists(agg_file):
    os.remove(agg_file)


def execute_command(command):
    # print("Executing ", cmd)
    print("Executing...", " ".join(command))

    result = subprocess.run(command, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return Code:", result.returncode)


def main():
    for i in range(1, max_size + 1):
        FAULTY_EDGES_COMB = itertools.combinations(EDGES, i)
        for FAULTY_EDGES in FAULTY_EDGES_COMB:
            FAULTY_EDGES = [f"{k[0]},{k[1]}" for k in FAULTY_EDGES]
            command = [
                "python",
                "simulate_v2.py",
                "--program",
                PROGRAM,
                "--faulty-edges",
                *FAULTY_EDGES,
                "--no-sim",
                N,
                "--fault-interval",
                *FI,
                "--graph-names",
                *GRAPH_NAMES,
                "--limit-steps",
                LIMIT_STEPS,
                "--hist-size",
                HIST_SIZE,
                "--extra-kwargs",
                "agg=1",
                f"agg_file={agg_file}",
            ]

            execute_command(command)

    # one with no faulty edges

    # FAULTY_EDGES = [f"{k[0]},{k[1]}" for k in EDGES]
    # FAULTY_EDGES = []
    # # HIST_SIZE = "5"

    # command = [
    #     "python",
    #     "simulate_v2.py",
    #     "--program",
    #     "maximal_matching",
    #     "--faulty-edges",
    #     *FAULTY_EDGES,
    #     "--no-sim",
    #     N,
    #     "--fault-interval",
    #     *FI,
    #     "--graph-names",
    #     *GRAPH_NAMES,
    #     "--limit-steps",
    #     LIMIT_STEPS,
    #     "--hist-size",
    #     HIST_SIZE,
    #     "--extra-kwargs",
    #     "agg=1",
    #     # "--logging",
    #     # "DEBUG",
    # ]

    # execute_command(command)


if __name__ == "__main__":
    main()
