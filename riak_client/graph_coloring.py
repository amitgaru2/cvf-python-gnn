import os
import sys

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)


from custom_logger import logger
from common_helpers import create_dir_if_not_exists
from command_line_helpers import (
    get_graph_v2,
    ColoringProgram,
    DijkstraProgram,
    MaxMatchingProgram,
    LinearRegressionProgram,
)


RIAK_BASE_URL = "http://localhost:8098"
RIAK_GRAPH_BUCKET_PREFIX = "graph_coloring"
RIAK_GRAPH_KEY_PREFIX = "graph_"


graph_name = sys.argv[1]
graph = get_graph_v2(graph_name)
logger.info(f"Loaded graph {graph}.")

init_config = tuple(0 for _ in graph.nodes())
logger.info(f"Writing initial config: {init_config}")
