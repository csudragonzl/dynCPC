import os


if __name__ == '__main__':
    edge_list_path = os.listdir('enron_large')
    edge_list_path.sort(key=lambda x: int(x[5:-6]))
    edges_list = []
    for i in range(len(edge_list_path)):
        file1 = open(os.path.join('enron_large', edge_list_path[i]), 'r')
        edges = list(y.split(' ')[:2] for y in file1.read().split('\n'))
        if i > 0:
            edges += edges_list[-1]
        file1.close()
        file2 = open(os.path.join('enron_large', edge_list_path[i]), 'w')
        file2.write('\n'.join(' '.join(edge) for edge in edges))
        file2.close()
        edges_list.append(edges)