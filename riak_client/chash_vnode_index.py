import hashlib


def chash_key(bucket_type: str, bucket: str, key: str) -> int:
    """
    Returns the SHA-1 hash of bucket<<>>key as an integer
    """
    data = (bucket + key).encode("utf-8")
    # sha1_hash_bytes = hashlib.sha1(data).digest()  # 20 bytes
    # print(len(sha1_hash_bytes))
    sha1_hash = hashlib.sha1(data).hexdigest()  # 40 hex characters
    return int(sha1_hash, 16)


def get_vnode_index(bucket: str, key: str, ring_size: int) -> int:
    chash = chash_key("default", bucket, key)
    return chash % ring_size


def main():
    bucket = "graph_coloring__star_graph_n6"
    keys = ["node_0", "node_1", "node_2", "node_3", "node_4", "node_5"]
    ring_size = 8  # check your Riak ring size
    for key in keys:
        index = get_vnode_index(bucket, key, ring_size)
        print(f"Key {key} in bucket {bucket} maps to vnode index {index}")


if __name__ == "__main__":
    main()
