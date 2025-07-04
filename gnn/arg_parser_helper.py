import os
import sys
import argparse

utils_path = os.path.join(
    os.getenv("CVF_PROJECT_DIR", "/home/agaru/research/cvf-python-gnn"), "utils"
)

sys.path.append(utils_path)

from command_line_helpers import PROGRAM_CHOICES


def generate_parser(
    takes_program=True, takes_graph_names=True, takes_nodes=False, takes_model=False
):
    parser = argparse.ArgumentParser()
    if takes_model:
        parser.add_argument(
            "--model",
            type=str,
            required=True,
        )

    if takes_program:
        parser.add_argument(
            "--program",
            choices=PROGRAM_CHOICES,
            required=True,
        )

    if takes_graph_names:
        parser.add_argument(
            "--graph-names",
            type=str,
            nargs="+",
            help="list of graph names in the 'graphs_dir' or list of number of nodes for implict graphs (if implicit program)",
            required=True,
        )

    if takes_nodes:
        parser.add_argument(
            "--nodes",
            type=int,
            nargs="+",
            help="Nodes",
            required=True,
        )

    parser.add_argument(
        "--logging",
        choices=[
            "INFO",
            "DEBUG",
        ],
        required=False,
    )

    args = parser.parse_args()

    return args
