import numpy as np


def getEdgeListFromAdj(adj, threshold=0.5, is_undirected=True, edge_pairs=None):
    result = []
    node_num = adj.shape[0]
    if edge_pairs:
        for (st, ed) in edge_pairs:
            if adj[st, ed] >= threshold:
                result.append((st, ed, adj[st, ed]))
    else:
        for i in range(node_num):
            for j in range(i, node_num):
                if adj[i, j] > threshold:
                    result.append((i, j, adj[i, j]))
                    result.append((j, i, adj[i, j]))
    return result


def graphify(reconstruction):
    [n1, n2] = reconstruction.shape
    n = min(n1, n2)
    reconstruction = reconstruction.detach().numpy()
    reconstruction = (reconstruction + reconstruction.T) / 2
    reconstruction -= np.diag(np.diag(reconstruction))
    return reconstruction