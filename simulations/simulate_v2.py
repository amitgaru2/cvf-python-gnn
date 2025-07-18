import os
import sys
import logging
import argparse

from custom_logger import logger
from simulation_v2 import SimulationMixinV2
from dijkstra_simulation import DijkstraSimulationV2
from graph_coloring_simulation import GraphColoringSimulationV2
from maximal_matching_simulation import (
    MaximalMatchingSimulationV2,
    MaximalMatchingSimulationSepVarV2,
)

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)

from command_line_helpers import (
    get_graph,
    ColoringProgram,
    DijkstraProgram,
    MaxMatchingProgram,
)

AnalysisMap = {
    ColoringProgram: GraphColoringSimulationV2,
    DijkstraProgram: DijkstraSimulationV2,
    MaxMatchingProgram: MaximalMatchingSimulationSepVarV2,
}


def parse_extra_kwargs(extra_kwargs):
    result = {}
    for kwarg in extra_kwargs:
        kw_split = kwarg.split("=")
        result[kw_split[0]] = kw_split[1]

    return result


def parse_faulty_edges(faulty_edges):
    result = []
    for edge in faulty_edges:
        result.append(tuple(int(i) for i in edge.split(",")))
    return result


def main(
    program,
    graph_name,
    graph,
    extra_kwargs,
    no_simulations,
    faulty_edges,
    fault_interval,
    limit_steps,
    hist_size,
):
    logger.info(
        "Analysis program: %s | graph: %s | No. of Simulations: %s | Fault Interval: %s | Limit steps: %s | History size: %s",
        program,
        graph_name,
        no_simulations,
        fault_interval,
        limit_steps,
        hist_size,
    )
    SimulationCVFAnalysisKlass: SimulationMixinV2 = AnalysisMap[program]
    simulation = SimulationCVFAnalysisKlass(graph_name, graph, extra_kwargs)
    simulation.create_simulation_environment(
        no_of_simulations=no_simulations,
        limit_steps=limit_steps,
        fault_interval=fault_interval,
        faulty_edges=faulty_edges,
        hist_size=hist_size,
    )
    result = simulation.start_simulation()
    if extra_kwargs.get("agg", False):
        simulation.aggregate_result(result, extra_kwargs.get("agg_file", None))
    else:
        simulation.store_raw_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--program",
        choices=[ColoringProgram, DijkstraProgram, MaxMatchingProgram],
        required=True,
    )  # coloring, dijkstra, max_matching
    parser.add_argument("--faulty-edges", type=str, nargs="*")
    parser.add_argument("--no-sim", type=int, required=True)  # number of simulations
    parser.add_argument("--limit-steps", type=int, default=None)
    # parser.add_argument(
    #     "--fault-interval", type=int, required=True
    # )
    parser.add_argument("--fault-interval", type=int, nargs=2)
    parser.add_argument(
        "--graph-names",
        type=str,
        nargs="+",
        help="list of graph names in the 'graphs_dir' or list of number of nodes for implict graphs (if implicit program)",
        required=True,
    )
    parser.add_argument(
        "--hist-size",
        type=int,
        help="size of the history that each node can store upto",
        default=5,
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
        "--extra-kwargs",
        type=str,
        nargs="*",
        help="any extra kwargs for the given program",
    )
    args = parser.parse_args()

    if args.logging:
        logger.info("Setting logger level to %s.", args.logging)
        logger.setLevel(getattr(logging, args.logging, "INFO"))

    faulty_edges = parse_faulty_edges(args.faulty_edges)
    extra_kwargs = parse_extra_kwargs(args.extra_kwargs) if args.extra_kwargs else {}
    for graph_name, graph in get_graph(args.graph_names):
        main(
            args.program,
            graph_name,
            graph,
            extra_kwargs,
            args.no_sim,
            faulty_edges,
            args.fault_interval,
            args.limit_steps,
            args.hist_size,
        )
