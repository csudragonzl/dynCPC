import time
import networkx as nx


if __name__ == '__main__':
    file = open('C://Users//dragonzl//Desktop//动态网络数据集//ia-contacts_dublin//ia-contacts_dublin.edges')
    out = open('ia-contacts_dublin//ia-contacts_dublin.edges', 'w')
    edges = list(y.split(',') for y in file.read().split('\n'))[:-1]
    min_time, max_time = int(edges[0][-1]), 0
    time_series = {}
    G = nx.Graph()
    for edge in edges:
        min_time = min(min_time, int(edge[-1]))
        max_time = max(max_time, int(edge[-1]))
        edge[-1] = time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(int(edge[-1])))
        if edge[-1] in time_series.keys():
            time_series[edge[-1]] += 1
        else:
            time_series[edge[-1]] = 1
        out.write(' '.join(edge))
        out.write('\n')
        G.add_edge(edge[0], edge[1], name=edge[-1])
    out.close()
    file.close()
    largest_components = max(nx.connected_components(G), key=len)
    print(largest_components)  #找出最大联通成分，返回是一个set{0,1,2,3}
    print(len(largest_components))
    print(time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(int(min_time))), time.strftime("%Y--%m--%d %H:%M:%S", time.localtime(int(max_time))))
    # for time_stamp, count in time_series.items():
    #     print(time_stamp, str(count), '\n')