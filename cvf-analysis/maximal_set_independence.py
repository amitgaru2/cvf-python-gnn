import os
import copy
import string

from cvf_analysis import CVFAnalysis, PartialCVFAnalysisMixin, logger


class Configuration:
    def __init__(self, val=0):
        self._val = val

    @property
    def val(self):
        return self._val

    @val.setter
    def val(self, val):
        self._val = val

    def __eq__(self, other):
        return self.val == other.val

    def __hash__(self) -> int:
        return hash(self.val)

    def __repr__(self):
        return f"<val: {self.val}>"


class MaximalSetIndependenceFullAnalysis(CVFAnalysis):
    results_prefix = "maximal_set_independence"
    results_dir = os.path.join("results", results_prefix)

    def possible_values_of_node(self, position):
        return {0, 1}  # 0: out, 1: in

    def _gen_configurations(self):
        self.configurations = {
            tuple([Configuration(val=0) for i in range(len(self.nodes))])
        }
        # perturb each state at a time for all states in configurations and accumulate the same in the configurations for next state to perturb
        for _, n in enumerate(self.nodes):
            node_pos = self.node_positions[n]
            config_copy = copy.deepcopy(self.configurations)
            for val in self.possible_values_of_node(node_pos):
                for cc in config_copy:
                    cc = list(cc)
                    cc[node_pos] = Configuration(val=val)
                    self.configurations.add(tuple(cc))

        logger.info("No. of Configurations: %s", len(self.configurations))

    def _I_lte_v_null(self, position, state):
        for nbr in self.graph_based_on_indx[position]:
            if (
                self.degree_of_nodes_based_on_indx[nbr]
                <= self.degree_of_nodes_based_on_indx[position]
                and state[nbr].val == 1
            ):
                return False
        return True

    def _check_if_none_eligible_process(self, state):
        """check invariant"""
        for position, config in enumerate(state):
            if config.val == 0 and self._I_lte_v_null(position, state):
                return False
            if config.val == 1 and not self._I_lte_v_null(position, state):
                return False

        return True

    def _find_invariants(self):
        for state in self.configurations:
            if self._check_if_none_eligible_process(state):
                self.invariants.add(state)

        logger.info("No. of Invariants: %s", len(self.invariants))
        print(self.invariants)

    def _is_program_transition(self, perturb_pos, start_state, dest_state) -> bool:
        if start_state[perturb_pos].val == 0 and self._I_lte_v_null(
            perturb_pos, start_state
        ):
            return dest_state[perturb_pos].val == 1
        if start_state[perturb_pos].val == 1 and not self._I_lte_v_null(
            perturb_pos, start_state
        ):
            return dest_state[perturb_pos].val == 0
        return False

    def _get_program_transitions(self, start_state):
        program_transitions = set()
        for position, _ in enumerate(start_state):
            possible_config_val = self.possible_values_of_node(position) - {
                start_state[position].val
            }
            for perturb_val in possible_config_val:
                perturb_state = list(copy.deepcopy(start_state))
                perturb_state[position].val = perturb_val
                perturb_state = tuple(perturb_state)
                if self._is_program_transition(position, start_state, perturb_state):
                    program_transitions.add(perturb_state)

        return {"program_transitions": program_transitions}

    def _get_cvfs(self, start_state):
        """
        Case 1: If you are IN and a neighbor with degree > yours is OUT: then your neighbor turning IN is a CVF
        """
        cvfs_in = dict()
        cvfs_out = dict()

        def _add_to_cvf(perturb_state, position):
            if start_state in self.invariants:
                cvfs_in[perturb_state] = position
            else:
                cvfs_out[perturb_state] = position

        for position, _ in enumerate(start_state):
            if start_state[position].val == 1:
                for nbr in self.graph_based_on_indx[position]:
                    if self.degree_of_nodes_based_on_indx[nbr] >= self.degree_of_nodes_based_on_indx[position]:
                        perturb_state = copy.deepcopy(start_state)
                        perturb_state[position].val = 0
                        _add_to_cvf(perturb_state, position)
                        break
            else:
                for nbr in self.graph_based_on_indx[position]:
                    if not self._I_lte_v_null:
                        perturb_state = copy.deepcopy(start_state)
                        perturb_state[position].val = 1
                        _add_to_cvf(perturb_state, position)
                        break
            # if start_state[position].val == 1:                                                                                                                                                                                                                                                                   
            #     for nbr in self.graph_based_on_indx[position]:
            #         if (
            #             self.degree_of_nodes_based_on_indx[nbr]
            #             > self.degree_of_nodes_based_on_indx[position]
            #             and start_state[nbr].val == 0
            #         ):
            #             perturb_state = copy.deepcopy(start_state)
            #             perturb_state[nbr].val = 1
            #             _add_to_cvf(perturb_state, position)
        print(start_state, cvfs_in, cvfs_out)
        return {"cvfs_in": cvfs_in, "cvfs_out": cvfs_out}

    def _start(self):
        self._gen_configurations()
        self._find_invariants()
        self._init_pts_rank()
        self._find_program_transitions_n_cvf()
        # self._rank_all_states()
        # self._gen_save_rank_count()
        # self._calculate_pts_rank_effect()
        # self._calculate_cvfs_rank_effect()
        # self._gen_save_rank_effect_count()
        # self._gen_save_rank_effect_by_node_count()
