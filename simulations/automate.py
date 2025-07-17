import itertools
import subprocess

EDGES = [(0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)]
max_size = len(EDGES)

N = "10000"
FI = ("5", "5")
GRAPH_NAMES = ("star_graph_n4",)
LIMIT_STEPS = "100"
HIST_SIZE = "5"


def execute_command(cmd):
    print("Executing ", cmd)

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
