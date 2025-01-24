import argparse

ColoringProgram = "graph_coloring"
DijkstraProgram = "dijkstra_token_ring"
MaxMatchingProgram = "maximal_matching"
MaxIndependentSetProgram = "maximal_independent_set"
LinearRegressionProgram = "linear_regression"



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--program",
        choices=[
            ColoringProgram,
            DijkstraProgram,
            MaxMatchingProgram,
            MaxIndependentSetProgram,
            LinearRegressionProgram,
        ],
        required=True,
    )  # coloring, dijkstra, max_matching
    parser.add_argument(
        "--scheduler",
        choices=[
           CENTRAL,
           DISTRIBUTED
        ],
        required=True,
    )  # coloring, dijkstra, max_matching
    # parser.add_argument("-f", "--full-analysis", action="store_true")
    parser.add_argument("--no-simulation", type=int, required=True)
    parser.add_argument("--no-simulation", type=int, required=True)
    parser.add_argument(
        "--graph-names",
        type=str,
        nargs="+",
        help="list of graph names in the 'graphs_dir' or list of number of nodes for implict graphs (if implicit program)",
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
    parser.add_argument(
        "--config-file",
        required=False,
    )
    args = parser.parse_args()
