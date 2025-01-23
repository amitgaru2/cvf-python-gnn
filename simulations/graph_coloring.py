import random

CENTRAL_SCHEDULER = 0
DISTRIBUTED_SCHEDULER = 1


def run_simulations(scheduler, me):
    state = get_random_state()
    step = 0
    while not is_invariant(state):
        actions = get_actions(scheduler, me, state)
        execute(actions)
        step += 1

    return step


def execute(actions):
    pass


def get_actions(scheduler, me, state):
    eligible_actions = get_all_eligible_actions(state)
    if scheduler == CENTRAL_SCHEDULER:
        actions = get_one_random_action(eligible_actions)
    else:
        actions = get_subset_of_actions(eligible_actions)
        if me:
            actions = remove_conflicts(actions)
    return actions


def remove_conflicts(actions):
    pass


def get_one_random_action(actions):
    return random.sample(actions, 1)


def get_subset_of_actions(actions):
    count = len(actions)
    subset_size = random.randint(1, count + 1)
    return random.sample(actions, subset_size)


def get_all_eligible_actions(state):
    pass


def get_random_state():
    pass


def is_invariant(state):
    return False


def main(no_of_simulations, scheduler, me):
    results = []
    for i in range(no_of_simulations):
        results[i] = run_simulations(scheduler, me)


if __name__ == "__main__":
    main()
