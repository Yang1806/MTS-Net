

def floyd(adj_matrix):
    num_nodes = len(adj_matrix)
    dist = [[float('inf') for _ in range(num_nodes)] for _ in range(num_nodes)]
    next_node = [[None for _ in range(num_nodes)] for _ in range(num_nodes)]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_matrix[i][j] == 1:
                dist[i][j] = 1
                next_node[i][j] = j

    for k in range(num_nodes):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_node[i][j] = next_node[i][k]

    return next_node

def shortest_path(next_node, start_node, end_node):
    path = [start_node]
    while start_node != end_node:
        start_node = next_node[start_node][end_node]
        if start_node == None:
            path = [-1]
            return path
        else:
            path.append(start_node)
    return path

