import logging
import os
import sys
import argparse

from custom_logger import logger
from simulation import CENTRAL_SCHEDULER, DISTRIBUTED_SCHEDULER
from graph_coloring_v2_simulation import GraphColoringSimulation

ColoringProgram = "graph_coloring"
DijkstraProgram = "dijkstra_token_ring"
MaxMatchingProgram = "maximal_matching"
MaxIndependentSetProgram = "maximal_independent_set"
LinearRegressionProgram = "linear_regression"

AnalysisMap = {
    ColoringProgram: GraphColoringSimulation,
}

graphs_dir = os.path.join(
    os.getenv("CVF_PROJECT_DIR", "/home"), "cvf-analysis", "graphs"
)


def start(graphs_dir, graph_names):
    for graph_name in graph_names:
        logger.info('Locating Graph: "%s".', graph_name)
        full_path = os.path.join(graphs_dir, f"{graph_name}.txt")
        if not os.path.exists(full_path):
            logger.warning("Graph file: %s not found! Skipping the graph.", full_path)
            continue

        graph = {}
        with open(full_path, "r") as f:
            line = f.readline()
            while line:
                node_edges = [int(i) for i in line.split()]
                node = node_edges[0]
                edges = node_edges[1:]
                graph[node] = set(edges)
                line = f.readline()

        yield graph_name, graph


def main(graph_name, graph, program, no_simulations, scheduler, me, fault_prob):
    if scheduler == CENTRAL_SCHEDULER:
        me = False

    logger.info(
        "Analysis graph: %s | program: %s | No. of Simulations: %s | Scheduler: %s | Mutual Exclusion: %s | Fault Probability: %s",
        graph_name,
        program,
        no_simulations,
        scheduler,
        me,
        fault_prob,
    )
    SimulationCVFAnalysisKlass = AnalysisMap[program]
    simulation = SimulationCVFAnalysisKlass(graph_name, graph)
    simulation.create_simulation_environment(
        no_of_simulations=no_simulations, scheduler=scheduler, me=me
    )
    simulation.apply_fault_settings(fault_probability=fault_prob)
    result = simulation.start_simulation()
    result = simulation.aggregate_result(result)
    logger.info("Result %s", result)
    simulation.store_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--program",
        choices=[
            ColoringProgram,
            # DijkstraProgram,
            # MaxMatchingProgram,
            # MaxIndependentSetProgram,
            # LinearRegressionProgram,
        ],
        required=True,
    )  # coloring, dijkstra, max_matching
    parser.add_argument(
        "--sched",
        choices=[CENTRAL_SCHEDULER, DISTRIBUTED_SCHEDULER],
        type=int,
        required=True,
    )
    parser.add_argument("-me", "--me", action="store_true")
    parser.add_argument("--no-sim", type=int, required=True)  # number of simulations
    parser.add_argument("--fault-prob", type=float, required=True)  # fault probability
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

    if args.logging:
        logger.setLevel(getattr(logging, args.logging, "INFO"))

    for graph_name, graph in start(graphs_dir, args.graph_names):
        main(
            graph_name,
            graph,
            args.program,
            args.no_sim,
            args.sched,
            args.me,
            args.fault_prob,
        )


"""
python main.py --program graph_coloring --sched 0 --no-sim 100 --fault-prob 0.5 --graph-names graph_1
python main.py --program graph_coloring --sched 1 --no-sim 100 --fault-prob 0.5 --graph-names graph_1
python main.py --program graph_coloring --sched 1 -me --no-sim 100 --fault-prob 0.5 --graph-names graph_1
"""
