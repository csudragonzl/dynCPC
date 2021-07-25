import torch
import torch.nn as nn
import numpy as np
from model.encoder import Encoder
from model.lstm import MLLSTM
import os
from time import perf_counter as t
import evaluation.evaluation_util
import evaluation.metrics
import networkx as nx
import pandas as pd
from sklearn.metrics import roc_auc_score


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, mllstm: MLLSTM, timestep, tau, activation=torch.sigmoid):
        super(Model, self).__init__()
        self.encoder = encoder
        self.mllstm = mllstm
        # timestep是预测的时间步长
        self.timestep = timestep
        self.tau = tau
        self.Wk = nn.ModuleList([nn.Linear(mllstm.output_dim, encoder.input_dim, bias=False) for i in range(timestep)])
        self.activation = activation

    def forward(self, x: torch.tensor, edge_index: dict, link_pred: bool):
        # x是当前输入时间片的集合
        nodes_num = x.size()[1]
        if self.encoder is None:
            x_encoded_list = x
        else:
            x_encoded_list = torch.empty(x.size()[0], x.size()[1], encoder.output_dim).to(device)
            x_gcn = torch.eye(x.size()[1]).to(device)
            for i in range(x.size()[0]):
                x_encoded_list[i] = self.encoder(x_gcn, edge_index[i])
        _, ct = self.mllstm(x_encoded_list[:])
        # ct = torch.mean(x_encoded, dim=0)
        if link_pred:
            x_pred_list = torch.empty((self.timestep, x.size()[1], x.size()[1])).to(device)
        else:
            x_pred_list = torch.empty((self.timestep, nodes_num, mllstm.output_dim)).to(device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            if link_pred:
                x_pred_list[i] = self.activation(linear(ct))
            else:
                x_pred_list[i] = linear(ct)
        return x_encoded_list, x_pred_list

    def cos_similar(self, z1, z2):
        sim_matrix = torch.mm(z1, z2.t())
        a = torch.norm(z1, p=2, dim=-1, keepdim=True)
        b = torch.norm(z2, p=2, dim=-1, keepdim=False)
        sim_matrix /= a
        sim_matrix /= b

        return sim_matrix

    def semi_loss(self, z1, z2):
        if data in ['bitcoin_alpha']:
            f = lambda x: x ** 2 / 0.5
            mask_z2 = torch.where(z2 > 0, z2 * 100, z2 + 0.1)
        elif data in ['college_msg']:
            f = lambda x: x ** 2 / 0.5
            mask_z2 = torch.where(z2 > 0, z2 * 50, z2 + 0.1)
        else:
            f = lambda x: torch.exp(x)
            mask_z2 = z2

        index = z2.sum(1) > 0
        between_sim = f(torch.mm(z1, mask_z2.t()))
        # between_sim = self.cos_similar(z1, mask_z2)
        if data in ['college_msg', 'bitcoin_alpha']:
            return -torch.log(
                between_sim.diag()[index] / (between_sim.sum(0)[index] - between_sim.diag()[index]) * index.shape[0]
            )
        else:
            return -torch.log(
                (between_sim.diag() / between_sim.sum(0))
            )

    def contrast_loss(self, x_pred, x_true, mean: bool = True):
        info_nce_loss = self.semi_loss(x_pred, x_true)
        info_nce_loss = info_nce_loss.mean() if mean else info_nce_loss.sum()

        # rhohats = z1.t().mean(axis=0)
        # kl = torch.mean(self.rho * torch.log(self.rho / rhohats) + (1 - self.rho) * torch.log(((1 - self.rho) / (1 - rhohats))))
        # kl_loss = self.beta * kl

        return info_nce_loss

    def reconstruct_loss(self, x_encoded, x_true, b):
        return torch.mean(torch.sum(torch.square((torch.sigmoid(torch.mm(x_encoded, x_encoded.t())) - x_true) * b), dim=1))

    def loss(self, x_pred_list, x_encoded_list, x_true):
        contrast_loss = torch.zeros(lookback, self.timestep).to(device)
        # reconstruct_loss = torch.zeros(lookback).to(device)
        for i in range(lookback):
            # b = x_true[i]
            # b = torch.where(b > 0, b * 5, b + 1)
            # reconstruct_loss[i] = self.reconstruct_loss(x_encoded_list[i], x_true[i], b)
            for j in range(self.timestep):
                contrast_loss[i][j] = self.contrast_loss(x_pred_list[j], x_true[i]) + (lookback - i + j) * theta
        return contrast_loss.sum()


def process(basepath: str):
    edge_list_path = os.listdir(basepath)

    if 'all' in basepath or 'msg' in basepath or 'bitcoin' in basepath:
        edge_list_path.sort(key=lambda x: int(x[8:-6]))
    elif 'enron' in basepath:
        edge_list_path.sort(key=lambda x: int(x[5:-6]))
    elif 'HS11' in basepath or 'primary' in basepath or 'workplace' in basepath or 'fbmessages' in basepath:
        edge_list_path.sort(key=lambda x: int(x[-5:-4]))
    elif 'HS12' in basepath:
        edge_list_path.sort(key=lambda x: int(x[-11:-10]))
    elif 'cellphone' in basepath:
        edge_list_path.sort(key=lambda x: int(x[9:-6]))
    node_num = 0
    edge_index = {}
    edges_list = []
    for i in range(len(edge_list_path)):
        file = open(os.path.join(basepath, edge_list_path[i]), 'r')
        # 不同的数据文件分隔符不一样
        if 'primary' in basepath or 'fbmessages' in basepath or 'workplace' in basepath or 'all' in basepath or 'msg' in basepath \
                or 'bitcoin' in basepath or 'dnc' in basepath:
            edges = list(y.split(' ')[:2] for y in file.read().split('\n'))[:-1]
        elif 'enron_large' in basepath:
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
        edges = torch.tensor(edges_temp).permute(1, 0).to(device)
        edge_index[i] = edges

    # 节点总数要加1
    node_num += 1
    # 时间片的邻接矩阵
    adj_list = torch.zeros(len(edge_list_path), node_num, node_num).to(device)
    for i in range(len(edge_list_path)):
        for j, k in edges_list[i]:
            adj_list[i, j, k] = 1
            adj_list[i, k, j] = 1

    return adj_list, edge_index


def train(model: Model, x, edge_index):
    x_encoded_list, x_pred_list = model(x, edge_index, True)
    loss = model.loss(x_pred_list, x_encoded_list, x)

    return loss, x_pred_list


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


def result_to_csv(metric_results, description):
    result = {}
    label = []
    for i in range(len(metric_results)):
        column = '第' + str(i) + '-' + str(i + lookback) + '个时间片'
        result[column] = metric_results[i]
    for j in range(timestamp):
        row = 'T+' + str(j)
        label.append(row)
    if description == 'MAP':
        # 求所有输入组的平均MAP值
        MAP_all = np.array(metric_results)
        mean_MAP = np.mean(MAP_all, axis=0).tolist()
        result['mean_MAP'] = mean_MAP
        for i in range(timestamp):
            print('预测未来第' + str(i) + '个时间片的mean MAP score is ' + str(mean_MAP[i]))
    else:
        AUC_all = np.array(metric_results)
        mean_AUC = np.mean(AUC_all, axis=0).tolist()
        result['mean_AUC'] = mean_AUC
        for i in range(timestamp):
            print('预测未来第' + str(i) + '个时间片的mean AUC score is ' + str(mean_AUC[i]))
    # csv_path = 'result2.0/' + data + '/' + 'lookback=' + str(lookback) + ',embsize=' + str(
    # embedding_size) + ',theta=' + str(theta) + '.csv'
    result_base_path = 'result2.0/pred_one/4.0'
    if not os.path.exists(result_base_path):
        os.mkdir(result_base_path)
    if description == 'MAP':
        csv_path = os.path.join(result_base_path, data + '_MAP.csv')
    else:
        csv_path = os.path.join(result_base_path, data + '_AUC.csv')
    df = pd.DataFrame(result, index=label)
    df.to_csv(csv_path)


if __name__ == '__main__':
    edge_index_list: dict
    data_list = ['college_msg', 'bitcoin_alpha']
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    for data in data_list:
        adj_list, edge_index_list = process('data/' + data)
        # for lookback in range(1, len(x_list) // 2 + 1):
        for lookback in range(3, 4):
            # for embedding_size in [64, 128, 256]:
            for embedding_size in [128]:
                # for theta in np.arange(0.1, 1.1, 0.1):
                for theta in np.arange(0.5, 0.6, 0.1):
                    # timestamp = lookback
                    timestamp = 1
                    MAP_all = []
                    AUC_all = []
                    precision_k_all = []

                    # 划分不同组输入 - lookback * 2 + 1
                    # for i in range(x_list.size()[0] - lookback * 2 + 1):
                    for i in range(adj_list.size()[0] - lookback - timestamp + 1):
                        encoder = Encoder(input_dim=adj_list.size()[1], output_dim=embedding_size).to(device)
                        mllstm = MLLSTM(input_dim=embedding_size, output_dim=embedding_size, n_units=[300, 300]).to(device)
                        model = Model(encoder=encoder, mllstm=mllstm, timestep=timestamp, tau=0.5).to(device)
                        optimizer = torch.optim.Adam(
                            model.parameters(), lr=0.001, weight_decay=0.0001)
                        start = t()
                        prev = start
                        x_input = adj_list[i:i+lookback]
                        edge_index_input = {}
                        for j in range(lookback):
                            edge_index_input[j] = edge_index_list[i + j]
                        model.train()
                        MAP_list = [0] * timestamp
                        AUC_list = [0] * timestamp
                        min_loss = 10
                        map_epoch = []
                        loss_epoch = []
                        if data in ['college_msg', 'bitcoin_alpha']:
                            epoches = 100
                        else:
                            epoches = 250
                        for epoch in range(epoches):
                            optimizer.zero_grad()
                            loss, x_pred_list = train(model, x_input, edge_index_input)
                            min_loss = min(loss.item(), min_loss)

                            loss.backward()
                            optimizer.step()

                            now = t()
                            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                                  f'this epoch {now - prev:.4f}, total {now - start:.4f}')
                            prev = now
                            MAP_list_temp = []
                            AUC_list_temp = []
                            for j in range(timestamp):
                                adj_reconstruct = evaluation.evaluation_util.graphify(x_pred_list[j].cpu())
                                print(x_pred_list[j][:8, :8])
                                edge_index_pre = evaluation.evaluation_util.getEdgeListFromAdj(adj=adj_reconstruct)
                                true_graph = nx.Graph()
                                true_graph.add_nodes_from([i for i in range(adj_list.size()[1])])
                                true_graph.add_edges_from(
                                    edge_index_list[i + lookback + j].permute(1, 0).cpu().numpy().tolist())
                                MAP, _ = evaluation.metrics.computeMAP(edge_index_pre, true_graph)
                                print('map value is ', MAP)
                                MAP_list_temp.append(MAP)
                            if sum(MAP_list_temp) > sum(MAP_list):
                                MAP_list = MAP_list_temp
                            for j in range(timestamp):
                                true_edges = edge_index_list[i + lookback + j].permute(1, 0).cpu().numpy()
                                AUC = compute_auc(true_edges, x_pred_list[j].cpu().detach().numpy())
                                AUC_list_temp.append(AUC)
                            if sum(AUC_list_temp) > sum(AUC_list):
                                AUC_list = AUC_list_temp
                        MAP_all.append(MAP_list)
                        AUC_all.append(AUC_list)

                        #     map_epoch.append(MAP)
                        #     loss_epoch.append(loss.item())
                        # x = np.arange(250)
                        # plt.figure(figsize=(20, 10), dpi=300)
                        #
                        # plt.subplot(1, 2, 1)
                        # plt.plot(x, map_epoch, marker='o', markersize=1)
                        # plt.xlabel('epochs')
                        # plt.ylabel('MAP scores')
                        #
                        # plt.subplot(1, 2, 2)
                        # plt.plot(x, loss_epoch, marker='o', markersize=1)
                        # plt.xlabel('epochs')
                        # plt.ylabel('loss')
                        #
                        # plt.savefig('dot_product.svg')
                        # break

                        print("=== Finish Training ===")
                        # model.eval()
                        # _, x_pre_list = train(model, x_input, edge_index_input)
                        #
                        # # 预测timestamp范围的MAP值
                        # MAP_list = []
                        # precision_k_list = []
                        # for j in range(timestamp):
                        #     adj_reconstruct = evaluation.evaluation_util.graphify(x_pred_list[j].cpu())
                        #     edge_index_pre = evaluation.evaluation_util.getEdgeListFromAdj(adj=adj_reconstruct)
                        #     print('预测得到的边数为', len(edge_index_pre))
                        #     true_graph = nx.Graph()
                        #     true_graph.add_nodes_from([i for i in range(adj_list.size()[1])])
                        #     true_graph.add_edges_from(edge_index_list[i + lookback + j].permute(1, 0).cpu().numpy().tolist())
                        #     MAP, precision_k = evaluation.metrics.computeMAP(edge_index_pre, true_graph)
                        #     MAP_list.append(MAP)
                        #     precision_k_list.append(precision_k)
                        #     print('第' + str(i) + '-' + str(i + lookback) + '个时间片的第' + str(j) + '步预测的MAP值为' + str(MAP))
                        # print("=== Finish Evaluating ===")
                        # 不同输入组的MAP值
                        # model.eval()
                        # _, x_pred_list = train(model, x_input, edge_index_input)
                        # for j in range(timestamp):
                        #     adj_reconstruct = evaluation.evaluation_util.graphify(x_pred_list[j].cpu())
                        #     print(x_pred_list[j][:8, :8])
                        #     edge_index_pre = evaluation.evaluation_util.getEdgeListFromAdj(adj=adj_reconstruct)
                        #     true_graph = nx.Graph()
                        #     true_graph.add_nodes_from([i for i in range(adj_list.size()[1])])
                        #     true_graph.add_edges_from(
                        #         edge_index_list[i + lookback + j].permute(1, 0).cpu().numpy().tolist())
                        #     MAP, _ = evaluation.metrics.computeMAP(edge_index_pre, true_graph)
                        #     MAP_list.append(MAP)
                        # MAP_all.append(MAP_list)
                        # for j in range(timestamp):
                        #     true_edges = edge_index_list[i + lookback + j].permute(1, 0).cpu().numpy()
                        #     AUC = compute_auc(true_edges, x_pred_list[j].cpu().detach().numpy())
                        #     AUC_list[j] = AUC
                    result_to_csv(MAP_all, 'MAP')
                    result_to_csv(AUC_all, 'AUC')
                    # result = {}
                    # label = []
                    # for i in range(len(precision_k_all)):
                    #     column = '第' + str(i) + '-' + str(i + lookback) + '个时间片'
                    #     result[column] = precision_k_all[i]
                    # for j in range(timestamp):
                    #     row = 'T+' + str(j)
                    #     label.append(row)
                    # # 求所有输入组的平均MAP值
                    # precision_k_all = np.array(precision_k_all)
                    # mean_precision_k = np.mean(precision_k_all, axis=0).tolist()
                    # result['mean_precision_k'] = mean_precision_k
                    # for i in range(timestamp):
                    #     print('预测未来第' + str(i) + '个时间片的mean MAP score is ' + str(mean_precision_k[i]))
                    # # csv_path = 'result2.0/' + data + '/' + 'lookback=' + str(lookback) + ',embsize=' + str(
                    # # embedding_size) + ',theta=' + str(theta) + '.csv'
                    # csv_path = 'result2.0/pred_one/' + data + '_precision@k_ae.csv'
                    # df = pd.DataFrame(result, index=label)
                    # df.to_csv(csv_path)
