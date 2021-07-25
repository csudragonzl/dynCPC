import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from evaluation.metrics import computeMAP
from evaluation.evaluation_util import graphify
from evaluation.evaluation_util import getEdgeListFromAdj
import numpy as np
from sklearn.metrics import roc_auc_score


class InnerProductDecoder(nn.Module):
    def __init__(self, act=torch.sigmoid, dropout=0.):
        super(InnerProductDecoder, self).__init__()

        self.act = act
        self.dropout = dropout

    def forward(self, inp):
        inp = F.dropout(inp, self.dropout, training=self.training)
        x = torch.transpose(inp, dim0=0, dim1=1)
        x = torch.mm(inp, x)
        return self.act(x)


def process(basepath: str):
    edge_list_path = os.listdir(basepath)

    if 'bitcoin' in basepath or 'ucimsg' in basepath:
        edge_list_path.sort(key=lambda x: int(x[8:-6]))
    elif 'enron' in basepath:
        edge_list_path.sort(key=lambda x: int(x[5:-6]))
    elif 'HS11' in basepath:
        edge_list_path.sort(key=lambda x: int(x[-5:-4]))
    elif 'HS12' in basepath:
        edge_list_path.sort(key=lambda x: int(x[-11:-10]))
    elif 'cellphone' in basepath:
        edge_list_path.sort(key=lambda x: int(x[9:-6]))
    node_num = 0
    edges_list = []
    for i in range(len(edge_list_path)):
        file = open(os.path.join(basepath, edge_list_path[i]), 'r')
        # 不同的数据文件分隔符不一样
        if 'ucimsg' in basepath or 'bitcoin' in basepath:
            edges = list(y.split(' ')[:2] for y in file.read().split('\n'))[:-1]
        elif 'large' in basepath:
            edges = list(y.split(' ')[:2] for y in file.read().split('\n'))
        else:
            edges = list(y.split('\t')[:2] for y in file.read().split('\n'))[:-1]
        for j in range(len(edges)):
            # 将字符的边转为int型
            edges[j] = list(int(z) - 1 for z in edges[j])

        # 去除重复的边
        edges = list(set([tuple(t) for t in edges]))
        edges_temp = []
        for j in range(len(edges)):
            # 去除反向的边和自环
            if [edges[j][1], edges[j][0]] not in edges_temp and edges[j][1] != edges[j][0]:
                edges_temp.append(edges[j])
            # 找到节点数
            for z in edges[j]:
                node_num = max(node_num, z)
        edges_list.append(edges_temp)

    # 节点总数要加1
    node_num += 1
    # 时间片的邻接矩阵
    x = torch.zeros(len(edge_list_path), node_num, node_num)
    for i in range(len(edge_list_path)):
        for j, k in edges_list[i]:
            x[i, j, k] = 1
            x[i, k, j] = 1

    return x, edges_list


def compute_auc(true_edges, pred_adj):
    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    false_edges = []
    while len(false_edges) < len(true_edges):
        idx_i = np.random.randint(0, pred_adj.shape[0])
        idx_j = np.random.randint(0, pred_adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], true_edges):
            continue
        if false_edges:
            if ismember([idx_j, idx_i], np.array(false_edges)):
                continue
            if ismember([idx_i, idx_j], np.array(false_edges)):
                continue
        false_edges.append([idx_i, idx_j])

    preds = []
    pos = []
    for e in true_edges:
        preds.append(pred_adj[e[0], e[1]])
        pos.append(pred_adj[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in false_edges:
        preds_neg.append(pred_adj[e[0], e[1]])
        neg.append(pred_adj[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)

    return roc_score


def result_to_csv(start_idx, metric_results, description, timesteps):
    result = {}
    label = []
    for i in range(len(metric_results)):
        column = '第' + str(start_idx + i) + '-' + str(start_idx + i + lookback) + '个时间片'
        result[column] = [format(x, '.4f') for x in metric_results[i]]
    for j in range(timesteps):
        row = 'T+' + str(j)
        label.append(row)
    if description == 'MAP':
        # 求所有输入组的平均MAP值
        MAP_all = np.array(metric_results)
        mean_MAP = np.mean(MAP_all, axis=0).tolist()
        result['mean_MAP'] = [format(x, '.4f') for x in mean_MAP]
        for i in range(timesteps):
            print('预测未来第' + str(i) + '个时间片的mean MAP score is ' + str(mean_MAP[i]))
    else:
        AUC_all = np.array(metric_results)
        mean_AUC = np.mean(AUC_all, axis=0).tolist()
        result['mean_AUC'] = [format(x, '.4f') for x in mean_AUC]
        for i in range(timesteps):
            print('预测未来第' + str(i) + '个时间片的mean AUC score is ' + str(mean_AUC[i]))
    if description == 'MAP':
        if timesteps == 1:
            csv_path = data + '_MAP.csv'
        else:
            csv_path = data + '_M_MAP.csv'
    else:
        if timesteps == 1:
            csv_path = data + '_AUC.csv'
        else:
            csv_path = data + '_M_AUC.csv'
    df = pd.DataFrame(result, index=label)
    df.to_csv(csv_path)


if __name__ == '__main__':
    # , 'primary', 'enron_large'
    datas = ['HS11', 'HS12', 'cellphone', 'enron', 'ucimsg', 'bitcoin_alpha']
    lookback = 3
    timesteps = 1
    for data in datas:
        if timesteps == 1:
            embedding_path = os.path.join(data, '2.embedding/VGRNN')
            embedding_file_list = os.listdir(embedding_path)
        else:
            embedding_path = os.path.join(data, '3.multi_embedding/VGRNN')
            embedding_dir_list = os.listdir(embedding_path)
        x_list, edges_list = process('../../data/' + data)

        MAP_all = []
        AUC_all = []

        if timesteps == 1:
            for i in range(len(embedding_file_list)):
                MAP_list = []
                AUC_list = []
                pd_frame = pd.read_csv(os.path.join(embedding_path, embedding_file_list[i]), sep='\t', index_col=0)
                nodes_str = pd_frame.index.values
                for j in range(len(nodes_str)):
                    nodes_str[j] = int(nodes_str[j][1:])
                pd_frame.set_index(nodes_str)
                sorted_pd_frame = pd_frame.sort_index()
                embedding_matrix = sorted_pd_frame.values
                embedding_matrix = torch.tensor(embedding_matrix)
                pred_adj = torch.sigmoid(torch.mm(embedding_matrix, embedding_matrix.t()))
                adj_reconstruct = graphify(pred_adj)
                edge_index_pre = getEdgeListFromAdj(adj=adj_reconstruct)
                true_graph = nx.Graph()
                true_graph.add_nodes_from([i for i in range(x_list.size()[1])])
                true_graph.add_edges_from(edges_list[i + lookback + 1])
                MAP = computeMAP(edge_index_pre, true_graph)
                MAP_list.append(MAP)
                AUC = compute_auc(np.array(edges_list[i + lookback + 1]), pred_adj)
                AUC_list.append(AUC)
                MAP_all.append(MAP_list)
                AUC_all.append(AUC_list)
        else:
            for i in range(len(embedding_dir_list)):
                MAP_list = []
                AUC_list = []
                embedding_file_list = os.listdir(os.path.join(embedding_path, embedding_dir_list[i]))
                for m in range(len(embedding_file_list)):
                    pd_frame = pd.read_csv(os.path.join(embedding_path, os.path.join(embedding_dir_list[i],
                                            embedding_file_list[m])), sep='\t', index_col=0)
                    nodes_str = pd_frame.index.values
                    for j in range(len(nodes_str)):
                        nodes_str[j] = int(nodes_str[j][1:])
                    pd_frame.set_index(nodes_str)
                    sorted_pd_frame = pd_frame.sort_index()
                    embedding_matrix = sorted_pd_frame.values
                    embedding_matrix = torch.tensor(embedding_matrix)
                    pred_adj = torch.sigmoid(torch.mm(embedding_matrix, embedding_matrix.t()))
                    adj_reconstruct = graphify(pred_adj)
                    edge_index_pre = getEdgeListFromAdj(adj=adj_reconstruct)
                    true_graph = nx.Graph()
                    true_graph.add_nodes_from([i for i in range(x_list.size()[1])])
                    true_graph.add_edges_from(edges_list[i + lookback + 1])
                    MAP = computeMAP(edge_index_pre, true_graph)
                    MAP_list.append(MAP)
                    AUC = compute_auc(np.array(edges_list[i + lookback + 1]), pred_adj)
                    AUC_list.append(AUC)
                MAP_all.append(MAP_list)
                AUC_all.append(AUC_list)
        result_to_csv(lookback + 1, MAP_all, 'MAP', timesteps)
        result_to_csv(lookback + 1, AUC_all, 'AUC', timesteps)
# if __name__ == '__main__':
#     # , 'primary', 'enron_large'
#     datas = ['HS11', 'HS12', 'cellphone', 'enron']
#     lookback = 3
#     timesteps = 1
#     for data in datas:
#         embedding_path = os.path.join(data, '2.embedding/VGRNN')
#         # embedding_path = os.path.join(data, '3.multi_embedding/VGRNN')
#         embedding_dir_list = os.listdir(embedding_path)
#         raw_file_list = [embedding_file for embedding_file in os.listdir(data) if 'edge' in embedding_file]
#         x_list, edges_list = process(data, raw_file_list)
#
#         MAP_all = []
#         AUC_all = []
#         result = {}
#
#         for i in range(len(embedding_dir_list)):
#             MAP_list = []
#             AUC_list = []
#             embedding_file_list = os.listdir(os.path.join(embedding_path, embedding_dir_list[i]))
#             # for m in range(len(embedding_file_list)):
#             for m in range(-1, 0):
#                 pd_frame = pd.read_csv(os.path.join(embedding_path, os.path.join(embedding_dir_list[i],
#                                                                                  embedding_file_list[m])), sep='\t', index_col=0)
#                 nodes_str = pd_frame.index.values
#                 for j in range(len(nodes_str)):
#                     nodes_str[j] = int(nodes_str[j][1:])
#                 pd_frame.set_index(nodes_str)
#                 sorted_pd_frame = pd_frame.sort_index()
#                 embedding_matrix = sorted_pd_frame.values
#                 embedding_matrix = torch.tensor(embedding_matrix)
#                 pred_adj = torch.sigmoid(torch.mm(embedding_matrix, embedding_matrix.t()))
#                 pred_adj = pred_adj.detach().numpy()
#                 # adj_reconstruct = graphify(pred_adj)
#                 # edge_index_pre = getEdgeListFromAdj(adj=adj_reconstruct)
#                 # print('预测得到的边数为', len(edge_index_pre))
#                 # true_graph = nx.Graph()
#                 # true_graph.add_nodes_from([i for i in range(x_list.size()[1])])
#                 # true_graph.add_edges_from(edge_index_list[int(embedding_file_list[m][0])].permute(1, 0).cpu().numpy().tolist())
#                 # true_graph.add_edges_from(edges_list[int(embedding_file_list[m][0]) + 2])
#                 # MAP = computeMAP(edge_index_pre, true_graph)
#                 # MAP_list.append(format(MAP, '.4f'))
#                 AUC = compute_auc(np.array(edges_list[int(embedding_file_list[m][0])]), pred_adj)
#                 AUC_list.append(AUC)
#             column = '第' + str(i) + '-' + str(i + lookback) + '个时间片'
#             # result[column] = MAP_list
#             # MAP_all.append(MAP_list)
#             result[column] = [format(x, '.4f') for x in AUC_list]
#             AUC_all.append(AUC_list)

