import sys

from lstm_scratch import SimpleLSTM, test_model_for_new_graphs


model = sys.argv[1]
program = sys.argv[2]
graph_name = sys.argv[3]


if __name__ == "__main__":
    test_model_for_new_graphs(
        model,
        program,
        [graph_name],
    )
