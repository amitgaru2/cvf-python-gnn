import time

from functools import wraps
from collections import defaultdict


function_runtimes = defaultdict(float)


def track_runtime(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        duration = end - start
        function_runtimes[func.__name__] += duration
        return result

    return wrapper


def reset_runtime():
    global function_runtimes
    function_runtimes = defaultdict(float)


def print_runtime_report(logger):
    logger.info("\n=== Runtime Report ===")
    for func_name, total_time in function_runtimes.items():
        logger.info(f"{func_name}: {total_time:.6f}s")
    logger.info("=== End Report ===\n\n")
