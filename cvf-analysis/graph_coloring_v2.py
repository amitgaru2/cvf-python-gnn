import copy
import math
import os

from pprint import pprint


class Rank:
    def __init__(self, L, C, M):
        self.L = L
        self.C = C
        self.M = M

    def add_cost(self, val):
        self.L += val
        self.C += 1
        self.M = max(val, self.M)

    def __str__(self) -> str:
        return f"L: {self.L}, C: {self.C}, M: {self.M}"

    __repr__ = __str__


GlobalRankMap = {}  # config: Rank
GlobalAvgRank = {}


def create_record_in_global_rank(config):
    GlobalRankMap[config] = Rank(L=0, C=0, M=0)


class ConfigurationNode:
    def __init__(self, config, children: set["ConfigurationNode"] = set()) -> None:
        self.config = config
        self.children = children  # program transitions
        # self.cost = Rank(0, 0, 0)

    def traverse(self):
        queue = [self]
        while queue:
            operating_node = queue.pop(0)
            print(operating_node)
            for child in operating_node.children:
                queue.append(child)

    def __hash__(self) -> int:
        return self.config

    def __eq__(self, other):
        return self.config == other.config

    def __str__(self) -> str:
        return f"{self.config}"

    __repr__ = __str__


graphs_dir = "graphs"
graph_names = ["small_graph_test"]


def start(graphs_dir, graph_name):
    # logger.info('Started for Graph: "%s".', graph_name)
    full_path = os.path.join(graphs_dir, f"{graph_name}.txt")
    if not os.path.exists(full_path):
        # logger.warning("Graph file: %s not found! Skipping the graph.", full_path)
        exit()

    graph = {}
    with open(full_path, "r") as f:
        line = f.readline()
        while line:
            node_edges = [int(i) for i in line.split()]
            node = node_edges[0]
            edges = node_edges[1:]
            graph[node] = set(edges)
            line = f.readline()

    return graph


class GraphColoring:
    def __init__(self) -> None:
        self.graph = start(graphs_dir, graph_names[0])
        self.nodes = list(self.graph.keys())
        self.degree_of_nodes = {n: len(self.graph[n]) for n in self.nodes}

        self.possible_node_values = [
            set(range(self.degree_of_nodes[node] + 1)) for node in self.nodes
        ]
        self.possible_node_values_length = [len(i) for i in self.possible_node_values]
        self.possible_values = list(
            set([j for i in self.possible_node_values for j in i])
        )
        self.possible_values.sort()
        self.possible_values_indx_str = {
            v: str(i) for i, v in enumerate(self.possible_values)
        }  # mapping from value to index

        self.initialize_helpers()

    def initialize_helpers(self):
        possible_node_values_len_rev = self.possible_node_values_length[::-1]

        self.base_n_to_decimal_multiplier = [1]
        for x in possible_node_values_len_rev[:-1]:
            self.base_n_to_decimal_multiplier.append(
                self.base_n_to_decimal_multiplier[-1] * x
            )

        self.base_n_to_decimal_multiplier_rev = self.base_n_to_decimal_multiplier[::-1]

    def base_n_to_decimal(self, base_n_str):
        value = 0
        length = len(base_n_str)
        for i in range(length):
            digit = int(base_n_str[length - i - 1])
            value = value + self.base_n_to_decimal_multiplier[i] * digit
        return value  # base 10, not fractional value

    def config_to_indx(self, config):
        config_to_indx_str = "".join(self.possible_values_indx_str[i] for i in config)
        result = self.base_n_to_decimal(config_to_indx_str)
        # print(config, result)
        return result

    def indx_to_config(self, indx: int):
        s = []
        for multiplier in self.base_n_to_decimal_multiplier_rev:
            if indx >= multiplier:
                indx_val = indx // multiplier
                s.append(indx_val)
                indx -= indx_val * multiplier
            else:
                s.append(0)
        return tuple(s)

    def start(self):
        self.find_rank()

    def _find_min_possible_color(self, colors):
        for i in range(len(colors) + 1):
            if i not in colors:
                return i

    def _is_program_transition(self, perturb_pos, start_state, dest_state):
        # if start_state in self.invariants and dest_state in self.invariants:
        #     return False
        neighbor_pos = [*self.graph[perturb_pos]]
        neighbor_colors = set(dest_state[i] for i in neighbor_pos)
        min_color = self._find_min_possible_color(neighbor_colors)

        return dest_state[perturb_pos] == min_color

    def _get_program_transitions(self, start_state):
        program_transitions = set()
        for position, val in enumerate(start_state):
            # check if node already has different color among the neighbors => If yes => no need to perturb that node's value
            # neighbor_pos = [*self.graph[position]]
            # neighbor_colors = set(start_state[i] for i in neighbor_pos)
            # if self._is_different_color(val, neighbor_colors):
            #     continue

            # if the current node's color is not different among the neighbors => search for the program transitions possible
            possible_node_colors = self.possible_node_values[position] - {
                start_state[position]
            }
            for perturb_val in possible_node_colors:
                perturb_state = list(start_state)
                perturb_state[position] = perturb_val
                perturb_state = tuple(perturb_state)
                if self._is_program_transition(position, start_state, perturb_state):
                    indx = self.config_to_indx(perturb_state)
                    program_transitions.add(ConfigurationNode(indx))

        return program_transitions

    def is_invariant(self, config):
        for node, color in enumerate(config):
            for dest_node in self.graph[node]:
                if config[dest_node] == color:
                    return False
        return True

    def backtrack_path(self, path: list[ConfigurationNode]):
        for i, node in enumerate(path):
            if node not in GlobalRankMap:
                create_record_in_global_rank(node)
            GlobalRankMap[node].add_cost(i)

    def dfs(self, path):
        state = path[-1]
        config = self.indx_to_config(state.config)
        if self.is_invariant(config):
            self.backtrack_path(path[::-1])
            return

        state.children = self._get_program_transitions(config)
        for node in state.children:
            path_copy = path[:]
            path_copy.append(node)
            self.dfs(path_copy)

    def _generate_configurations(self):
        config = tuple([0 for _ in self.nodes])
        yield config

        configurations = {config}

        # perturb each state at a time for all states in configurations and accumulate the same in the configurations for next state to perturb
        for node_pos in self.nodes:
            config_copy = copy.deepcopy(configurations)
            for i in range(1, self.degree_of_nodes[node_pos] + 1):
                for cc in config_copy:
                    cc = list(cc)
                    cc[node_pos] = i
                    config = tuple(cc)
                    yield config
                    configurations.add(config)

    def find_rank(self):
        configurations = self._generate_configurations()
        for config in configurations:
            indx = self.config_to_indx(config)
            self.dfs([ConfigurationNode(indx)])

        for _, rank in GlobalRankMap.items():
            avg_rank = math.ceil(rank.L / rank.C)
            if avg_rank in GlobalAvgRank:
                GlobalAvgRank[avg_rank] += 1
            else:
                GlobalAvgRank[avg_rank] = 1


def main():
    coloring = GraphColoring()
    coloring.start()
    # pprint(GlobalRankMap)
    # print(len(GlobalRankMap))
    # coloring.initial_state.traverse()
    pprint(GlobalAvgRank)


if __name__ == "__main__":
    main()
