import os
import pandas as pd
import numpy as np

edge_list_path = os.listdir('data/enron')
edge_list_path.sort(key=lambda x: int(x[5:-6]))
node_num = 0
nodes_set = []
file = open(os.path.join('data/enron', edge_list_path[4]), 'r')
edges = list(y.split('\t')[:2] for y in file.read().split('\n'))[:-1]
for j in range(len(edges)):
    # 将字符的边转为int型
    edges[j] = list(int(z) - 1 for z in edges[j])

# 去除重复的边
edges = list(set([tuple(t) for t in edges]))
edges_temp = []
edges_all = []
for j in range(len(edges)):
    # 去除反向的边和自环
    if [edges[j][1], edges[j][0]] not in edges_temp and edges[j][1] != edges[j][0]:
        edge = list(edges[j])
        edges_temp.append(edge)
        edge.append(1)
        edges_all.append(edge)
        nodes_set.append(edges[j][0])
        nodes_set.append(edges[j][1])
edges_all = np.array(edges_all)
nodes_set = list(set(nodes_set))

df_edges = pd.DataFrame(data=np.array(edges_all), columns=['source', 'target', 'weight'])
df_edges.to_csv('edges_true.csv', index=False)
df_nodes = pd.DataFrame(data=np.array(nodes_set), columns=['nodes'])
df_nodes.to_csv('nodes.csv', index=False)

dysat = 0.5
vgrnn = 0.7
tmf = 0.55
ctlp = 0.8

index = np.arange(len(edges_all))
np.random.shuffle(index)
for i in range(len(index)):
    if i < int(len(index) * dysat):
        edges_all[index[i]][-1] = 1
    else:
        edges_all[index[i]][-1] = 0
df_edges = pd.DataFrame(data=np.array(edges_all), columns=['source', 'target', 'weight'])
df_edges.to_csv('edges_dysat.csv', index=False)

np.random.shuffle(index)
for i in range(len(index)):
    if i < int(len(index) * vgrnn):
        edges_all[index[i]][-1] = 1
    else:
        edges_all[index[i]][-1] = 0
df_edges = pd.DataFrame(data=np.array(edges_all), columns=['source', 'target', 'weight'])
df_edges.to_csv('edges_vgrnn.csv', index=False)

np.random.shuffle(index)
for i in range(len(index)):
    if i < int(len(index) * tmf):
        edges_all[index[i]][-1] = 1
    else:
        edges_all[index[i]][-1] = 0
df_edges = pd.DataFrame(data=np.array(edges_all), columns=['source', 'target', 'weight'])
df_edges.to_csv('edges_tmf.csv', index=False)

np.random.shuffle(index)
for i in range(len(index)):
    if i < int(len(index) * ctlp):
        edges_all[index[i]][-1] = 1
    else:
        edges_all[index[i]][-1] = 0
df_edges = pd.DataFrame(data=np.array(edges_all), columns=['source', 'target', 'weight'])
df_edges.to_csv('edges_ctlp.csv', index=False)