import os
import json
import time
import random

import requests

from functools import wraps
from custom_logger import logger

# RING_SIZE = 8
# RIAK_BASE_URL = "http://localhost:8098"
RIAK_BASE_URLS = [
    f"http://{url}"
    for url in os.getenv("RIAK_SERVER_URLS", "localhost:8098").split(";")
]
RIAK_BUCKET_PREFIX = "graph_coloring"
RIAK_LCK_BUCKET_PREFIX = "graph_coloring_lck"
RIAK_NODE_KEY_PREFIX = "node_"
RIAK_PETERSON_LCK_FLAG_KEY_PREFIX = "L_FLAG_"
RIAK_PETERSON_LCK_TURN_KEY_PREFIX = "L_TURN_"

logger.info(f"Using RIAK_BASE_URLS: {RIAK_BASE_URLS}")


def retry(
    backoff_factor=1,
    exceptions=(requests.exceptions.RequestException,),
):
    """
    Retry decorator with exponential backoff.

    :param retries: Number of retries before giving up
    :param backoff_factor: Multiplier for wait time between retries
    :param exceptions: Tuple of exception classes to catch
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    wait = backoff_factor * (2 ** (attempt - 1))
                    logger.warning(
                        f"Attempt {attempt} failed: {e}. Retrying in {wait}s..."
                    )
                    time.sleep(wait)

        return wrapper

    return decorator


def get_random_riak_base_url():
    return random.choice(RIAK_BASE_URLS)


@retry()
def put_request_riak(bucket_name, key, value):
    """
    Implements the equivalent of:
      curl -XPUT \
        -H "Content-Type: application/json" \
        -d '{"name":"BAlice2"}' \
        http://127.0.0.1:8098/buckets/users/keys/user1
    """
    url = f"{get_random_riak_base_url()}/buckets/{bucket_name}/keys/{key}"
    headers = {"Content-Type": "application/json"}
    data = value

    try:
        response = requests.put(url, json=data, headers=headers)
        response.raise_for_status()
        logger.debug(f"Wrote to {url}.")
        logger.debug(f"Success: {response.status_code}")
        if response.text:
            logger.debug("Response body:", response.text)
        return True
    except requests.HTTPError as err:
        logger.error(f"HTTP error: {err}")
        logger.error("Status code:", err.response.status_code)
        logger.error("Response:", err.response.text)
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False


@retry()
def get_request_riak(bucket_name, key, params={}):
    if key:
        url = f"{get_random_riak_base_url()}/buckets/{bucket_name}/keys/{key}"
    else:
        url = f"{get_random_riak_base_url()}/buckets/{bucket_name}/keys"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        try:
            value = response.json()  # attempt to parse JSON
        except json.JSONDecodeError:
            value = response.text  # fallback to raw text
        logger.debug(f"Read from {url}.")
    elif response.status_code == 404:
        logger.error(f"Key '{key}' not found in bucket '{bucket_name}'.")
        value = None
    else:
        logger.error(f"Error {response.status_code}: {response.text}")
        value = None
    return value


@retry()
def delete_request_riak(bucket_name, key):
    url = f"{get_random_riak_base_url()}/buckets/{bucket_name}/keys/{key}"
    try:
        response = requests.delete(url)
        response.raise_for_status()
        logger.debug(f"Success deleting key '{key}' from bucket '{bucket_name}'.")
        if response.text:
            logger.debug("Response body:", response.text)
        return True
    except requests.HTTPError as err:
        logger.error(f"HTTP error: {err}")
        logger.error("Status code:", err.response.status_code)
        logger.error("Response:", err.response.text)
        return False
    except Exception as e:
        logger.error(f"Error: {e}")
        return False
