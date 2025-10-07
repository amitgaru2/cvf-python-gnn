import os
import sys
import json

import requests

utils_path = os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "utils")
sys.path.append(utils_path)


from custom_logger import logger

RING_SIZE = 8
RIAK_BASE_URL = "http://localhost:8098"
RIAK_BUCKET_PREFIX = "graph_coloring"
RIAK_NODE_KEY_PREFIX = "node_"
RIAK_PETERSON_LCK_FLAG_KEY_PREFIX = "L_FLAG_"
RIAK_PETERSON_LCK_TURN_KEY_PREFIX = "L_TURN_"


def put_request_riak(bucket_name, key, value):
    """
    Implements the equivalent of:
      curl -XPUT \
        -H "Content-Type: application/json" \
        -d '{"name":"BAlice2"}' \
        http://127.0.0.1:8098/buckets/users/keys/user1
    """
    url = f"{RIAK_BASE_URL}/buckets/{bucket_name}/keys/{key}"
    headers = {"Content-Type": "application/json"}
    data = value

    try:
        response = requests.put(url, json=data, headers=headers)
        response.raise_for_status()
        logger.info(f"Wrote {{ {key}: {value} }} to the Bucket: {bucket_name}.")
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


def get_request_riak(bucket_name, key, params={}):
    if key:
        url = f"{RIAK_BASE_URL}/buckets/{bucket_name}/keys/{key}"
    else:
        url = f"{RIAK_BASE_URL}/buckets/{bucket_name}/keys"
    headers = {"Content-Type": "application/json"}
    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        try:
            value = response.json()  # attempt to parse JSON
        except json.JSONDecodeError:
            value = response.text  # fallback to raw text
    elif response.status_code == 404:
        logger.error(f"Key '{key}' not found in bucket '{bucket_name}'.")
        value = None
    else:
        logger.error(f"Error {response.status_code}: {response.text}")
        value = None
    return value


def delete_request_riak(bucket_name, key):
    url = f"{RIAK_BASE_URL}/buckets/{bucket_name}/keys/{key}"
    try:
        response = requests.delete(url)
        response.raise_for_status()
        logger.info(f"Success deleting key '{key}' from bucket '{bucket_name}'.")
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
