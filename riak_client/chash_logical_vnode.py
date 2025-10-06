import hashlib


bucket = "users"
key = "user1"
ring_size = 64  # check your Riak ring size


def chash_key(bucket: str, key: str) -> int:
    """
    Returns the SHA-1 hash of bucket<<>>key as an integer
    """
    data = (bucket + key).encode("utf-8")
    # sha1_hash_bytes = hashlib.sha1(data).digest()  # 20 bytes
    # print(len(sha1_hash_bytes))
    sha1_hash = hashlib.sha1(data).hexdigest()  # 40 hex characters
    return int(sha1_hash, 16)


def vnode_index(bucket: str, key: str, ring_size: int) -> int:
    chash = chash_key(bucket, key)
    return chash % ring_size


index = vnode_index(bucket, key, ring_size)
print(f"Key {key} in bucket {bucket} maps to vnode index {index}")
