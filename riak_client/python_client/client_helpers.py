def consistent_partition(numbers, m):
    """
    Partition N numbers into M partitions consistently using a hash-based method.
    """
    partitions = [[] for _ in range(m)]
    for num in numbers:
        index = hash(num) % m
        partitions[index].append(num)
    return partitions


def get_partition_for_client(graph, client_id, m):
    """
    Get the partition of nodes assigned to a client based on its ID.
    m: total number of clients.
    client_id: ID of the client (0 to m-1).
    """
    nodes = list(graph.nodes())
    partitions = consistent_partition(nodes, m)
    return partitions[client_id]
