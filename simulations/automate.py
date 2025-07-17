import os
import sys
import itertools
import subprocess

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)

from command_line_helpers import (
    get_graph,
)

GRAPH_NAMES = ("star_graph_n4",)

GRAPH = next(get_graph(GRAPH_NAMES))[1]

EDGES = []
for src, dests in GRAPH.items():
    EDGES.extend([(src, dest) for dest in dests])  # (src, dest) src being read by dest

max_size = len(EDGES)

N = "10000"
FI = ("5", "5")
LIMIT_STEPS = "100"
HIST_SIZE = "5"


def execute_command(cmd):
    # print("Executing ", cmd)
    print("Executing...", " ".join(cmd))

    result = subprocess.run(command, capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return Code:", result.returncode)


for i in range(1, max_size + 1):
    FAULTY_EDGES_COMB = itertools.combinations(EDGES, i)
    for FAULTY_EDGES in FAULTY_EDGES_COMB:
        FAULTY_EDGES = [f"{k[0]},{k[1]}" for k in FAULTY_EDGES]
        command = [
            "python",
            "simulate_v2.py",
            "--program",
            "maximal_matching",
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
        ]

        execute_command(command)

# one with no faulty edges

command = [
    "python",
    "simulate_v2.py",
    "--program",
    "maximal_matching",
    "--faulty-edges",
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
]

execute_command(command)
