import os
import logging
import argparse

from custom_logger import logger
from dijkstra_simulation import DijkstraSimulation
from graph_coloring_simulation import GraphColoringSimulation
from maximal_matching_simulation import MaximalMatchingSimulation
from simulation import CENTRAL_SCHEDULER, DISTRIBUTED_SCHEDULER, SimulationMixin
from maximal_independent_set_simulation import MaximalIndependentSetSimulation

ColoringProgram = "graph_coloring"
DijkstraProgram = "dijkstra_token_ring"
MaxMatchingProgram = "maximal_matching"
MaxIndependentSetProgram = "maximal_independent_set"
LinearRegressionProgram = "linear_regression"

AnalysisMap = {
    ColoringProgram: GraphColoringSimulation,
    DijkstraProgram: DijkstraSimulation,
    MaxMatchingProgram: MaximalMatchingSimulation,
    MaxIndependentSetProgram: MaximalIndependentSetSimulation,
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


def main(
    graph_name,
    graph,
    program,
    simulation_type,
    no_simulations,
    scheduler,
    me,
    fault_prob,
    fault_interval,
    simulation_type_args,
):
    if scheduler == CENTRAL_SCHEDULER:
        me = False

    logger.info(
        "Analysis graph: %s | program: %s | Simulation Type: %s, Args: %s | No. of Simulations: %s | Scheduler: %s | Mutual Exclusion: %s | Fault Interval: %s",
        graph_name,
        program,
        simulation_type,
        simulation_type_args,
        no_simulations,
        scheduler,
        me,
        fault_interval,
    )
    SimulationCVFAnalysisKlass = AnalysisMap[program]
    simulation = SimulationCVFAnalysisKlass(graph_name, graph)
    simulation.create_simulation_environment(
        simulation_type=simulation_type,
        no_of_simulations=no_simulations,
        scheduler=scheduler,
        me=me,
    )
    simulation.apply_fault_settings(
        fault_probability=fault_prob, fault_interval=fault_interval
    )
    result = simulation.start_simulation(*simulation_type_args)
    simulation.store_raw_result(result, *simulation_type_args)
    # hist, bin_edges = simulation.aggregate_result(result)
    # logger.info("Result %s", result)
    # simulation.store_result(hist, bin_edges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--program",
        choices=[
            ColoringProgram,
            DijkstraProgram,
            MaxMatchingProgram,
        ],
        required=True,
    )  # coloring, dijkstra, max_matching
    parser.add_argument(
        "--simulation-type",
        choices=[
            SimulationMixin.RANDOM_FAULT_SIMULATION_TYPE,
            SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE,
            SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_DUONG,
        ],
        required=True,
    )
    parser.add_argument("--controlled-at-node", type=int, required=False, default=None)
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
        "--config-file",
        required=False,
    )
    args = parser.parse_args()

    if args.logging:
        logger.info("Setting logger level to %s.", args.logging)
        logger.setLevel(getattr(logging, args.logging, "INFO"))

    simulation_type_args = []
    if args.simulation_type in {
        SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE,
        SimulationMixin.CONTROLLED_FAULT_AT_NODE_SIMULATION_TYPE_DUONG,
    }:
        if args.controlled_at_node is None:
            raise Exception('Missing "--controlled-at-node" argument.')
        else:
            simulation_type_args = [args.controlled_at_node]

    for graph_name, graph in start(graphs_dir, args.graph_names):
        main(
            graph_name,
            graph,
            args.program,
            args.simulation_type,
            args.no_sim,
            args.sched,
            args.me,
            args.fault_prob,
            args.fault_interval,
            simulation_type_args,
        )


"""
python main.py --program graph_coloring --sched 0 --no-sim 100 --fault-prob 0.5 --graph-names graph_1
python main.py --program graph_coloring --sched 1 --no-sim 100 --fault-prob 0.5 --graph-names graph_1
python main.py --program graph_coloring --sched 1 -me --no-sim 100 --fault-prob 0.5 --graph-names graph_1
"""
