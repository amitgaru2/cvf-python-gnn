import argparse


ColoringProgram = "coloring"
DijkstraProgram = "dijkstra_token_ring"
MaxMatchingProgram = "maximal_matching"


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
            choices=[
                ColoringProgram,
                DijkstraProgram,
                MaxMatchingProgram,
            ],
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
