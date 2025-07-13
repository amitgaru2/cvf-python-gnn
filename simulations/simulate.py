import os
import sys
import logging
import argparse

from custom_logger import logger
from dijkstra_simulation import DijkstraSimulation
from linear_regression_simulation import LinearRegressionSimulation
from graph_coloring_simulation import GraphColoringSimulation
from maximal_matching_simulation import MaximalMatchingSimulation
from simulation import CENTRAL_SCHEDULER, DISTRIBUTED_SCHEDULER, SimulationMixin

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)

from command_line_helpers import (
    get_graph,
    ColoringProgram,
    DijkstraProgram,
    MaxMatchingProgram,
    LinearRegressionProgram,
)

AnalysisMap = {
    ColoringProgram: GraphColoringSimulation,
    DijkstraProgram: DijkstraSimulation,
    MaxMatchingProgram: MaximalMatchingSimulation,
    LinearRegressionProgram: LinearRegressionSimulation,
}


def parse_extra_kwargs(extra_kwargs):
    result = {}
    for kwarg in extra_kwargs:
        kw_split = kwarg.split("=")
        result[kw_split[0]] = kw_split[1]

    return result


def parse_controlled_at_nodes_w_wt(controlled_at_nodes_w_wt):
    result = {}
    tot_prob = 0
    for kv in controlled_at_nodes_w_wt:
        kv_split = kv.split("=")
        k, v = int(kv_split[0]), float(kv_split[1])
        result[k] = v
        tot_prob += v

    if tot_prob > 1.0:
        raise Exception("Probabilties sum cannot be greater than 1.0.")

    return result


def main(
    program,
    graph_name,
    graph,
    extra_kwargs,
    simulation_type,
    no_simulations,
    fault_prob,
    fault_interval,
    limit_steps,
    simulation_type_kwargs,
):
    logger.info(
        "Analysis graph: %s | program: %s | Simulation Type: %s, Args: %s | No. of Simulations: %s | Scheduler: %s | Mutual Exclusion: %s | Fault Interval: %s",
        graph_name,
        program,
        simulation_type,
        simulation_type_kwargs,
        no_simulations,
        0,
        False,
        fault_interval,
    )
    SimulationCVFAnalysisKlass = AnalysisMap[program]
    simulation = SimulationCVFAnalysisKlass(graph_name, graph, extra_kwargs)
    simulation.create_simulation_environment(
        simulation_type=simulation_type,
        no_of_simulations=no_simulations,
        scheduler=0,
        me=False,
        limit_steps=limit_steps,
    )
    simulation.apply_fault_settings(
        fault_probability=fault_prob, fault_interval=fault_interval
    )
    result = simulation.start_simulation(simulation_type_kwargs)
    simulation.store_raw_result(result, simulation_type_kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--program",
        choices=[
            ColoringProgram,
            DijkstraProgram,
            MaxMatchingProgram,
            LinearRegressionProgram,
        ],
        required=True,
    )  # coloring, dijkstra, max_matching
    parser.add_argument(
        "--simulation-type",
        choices=[
            SimulationMixin.RANDOM_FAULT_SIMULATION_TYPE,
            SimulationMixin.RANDOM_FAULT_START_AT_NODE_SIMULATION_TYPE,
            SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE,
            SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_AMIT_V2,
            SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_DUONG,
        ],
        required=True,
    )
    parser.add_argument("--controlled-at-nodes-w-wt", type=str, nargs="*")
    # parser.add_argument(
    #     "--sched",
    #     choices=[CENTRAL_SCHEDULER, DISTRIBUTED_SCHEDULER],
    #     type=int,
    #     required=True,
    # )
    # parser.add_argument("-me", "--me", action="store_true")
    parser.add_argument("--no-sim", type=int, required=True)  # number of simulations
    parser.add_argument("--fault-prob", type=float, required=True)  # fault probability
    parser.add_argument("--limit-steps", type=int, default=None)
    parser.add_argument(
        "--fault-interval", type=int, required=True
    )  # fault probability
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
        "--extra-kwargs",
        type=str,
        nargs="*",
        help="any extra kwargs for the given program",
    )
    args = parser.parse_args()

    if args.logging:
        logger.info("Setting logger level to %s.", args.logging)
        logger.setLevel(getattr(logging, args.logging, "INFO"))

    simulation_type_args = []
    if args.simulation_type in {
        SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE,
        SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_AMIT_V2,
        SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_DUONG,
        SimulationMixin.RANDOM_FAULT_START_AT_NODE_SIMULATION_TYPE,
    }:
        if args.controlled_at_nodes_w_wt is None or not args.controlled_at_nodes_w_wt:
            raise Exception('Missing "--controlled-at-nodes-w-wt" argument.')
        else:
            controlled_at_nodes_w_wt = parse_controlled_at_nodes_w_wt(
                args.controlled_at_nodes_w_wt
            )
            simulation_type_kwargs = {
                "controlled_at_nodes_w_wt": controlled_at_nodes_w_wt
            }

    extra_kwargs = parse_extra_kwargs(args.extra_kwargs) if args.extra_kwargs else {}
    for graph_name, graph in get_graph(args.graph_names):
        main(
            args.program,
            graph_name,
            graph,
            extra_kwargs,
            args.simulation_type,
            args.no_sim,
            args.fault_prob,
            args.fault_interval,
            args.limit_steps,
            simulation_type_kwargs,
        )
