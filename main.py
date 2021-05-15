import torch
import torch.nn as nn
import numpy as np
from model.gcn_encoder import Encoder
from model.lstm import MLLSTM
import os
from time import perf_counter as t
import evaluation.evaluation_util
import evaluation.metrics
import networkx as nx
import pandas as pd


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, mllstm: MLLSTM, timestep, tau, beta, rho, activation=torch.sigmoid):
        super(Model, self).__init__()
        self.encoder = encoder
        self.mllstm = mllstm
        # timestep是预测的时间步长
        self.timestep = timestep
        self.tau = tau
        self.Wk = nn.ModuleList([nn.Linear(mllstm.output_dim, encoder.input_dim, bias=False) for i in range(timestep)])
        self.beta = beta
        self.rho = rho
        self.activation = activation

    def forward(self, x: torch.tensor, edge_index: dict, link_pred: bool):
        # x是当前输入时间片的集合
        nodes_num = x.size()[1]
        if self.encoder is None:
            x_encoded = x
        else:
            x_encoded = torch.empty(x.size()[0], x.size()[1], encoder.output_dim).to(device)
            for i in range(x.size()[0]):
                x_encoded[i] = self.encoder(x[i], edge_index[i])
        _, ct = self.mllstm(x_encoded[:])
        # ct = torch.mean(x_encoded, dim=0)
        if link_pred:
            pred = torch.empty((self.timestep, nodes_num, nodes_num)).to(device)
        else:
            pred = torch.empty((self.timestep, nodes_num, mllstm.output_dim)).to(device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            if link_pred:
                pred[i] = self.activation(linear(ct))
            else:
                pred[i] = linear(ct)
        return pred

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        if exp_flag:
            between_sim = f(torch.mm(z1, z2.t()))
        else:
            index = torch.nonzero(z2.sum(0), as_tuple=True)[0]
            z1 = z1[index]
            z1 = z1[:, index]
            z2 = z2[index]
            z2 = z2[:, index]
            between_sim = torch.zeros(z1.shape).to(device)
            for i in range(between_sim.shape[0]):
                # between_sim[i] = torch.cosine_similarity(z1[i], z2, dim=0) / self.tau
                between_sim[i] = torch.pairwise_distance(z1[i], z2, p=1) / self.tau
        # between_sim = f(torch.mm(z1[index], z2[index].t()))
        # else:
        #     between_sim = torch.mm(z1, z2.t())
        #     between_ones = torch.ones(between_sim.shape).to(device)
        #     between_sim = torch.where(between_sim < 1, between_ones, between_sim)
        # pos = between_sim.diag()
        # res = torch.zeros(pos.size())
        # for i in range(pos.size()[0]):
        #     node_indices = np.append(np.arange(0, i), np.arange(i + 1, z1.size()[0]))
        #     np.random.shuffle(node_indices)
        #     node_indices = node_indices[:len(node_indices) // 10]
        #     res[i] = pos[i] / (between_sim[i, node_indices].sum(0) + pos[i])
        return -torch.log(
            between_sim.diag() / between_sim.sum(0)
        )

    def loss(self, z1, z2, mean: bool = True):
        ret = self.semi_loss(z1, z2)
        ret = ret.mean() if mean else ret.sum()

        rhohats = z1.t().mean(axis=0)
        kl = torch.mean(self.rho * torch.log(self.rho / rhohats) + (1 - self.rho) * torch.log(((1 - self.rho) / (1 - rhohats))))
        kl_loss = self.beta * kl

        return ret + kl_loss


def process(basepath: str):
    edge_list_path = os.listdir(basepath)
    if 'enron_all' in basepath:
        exp_flag = False
    else:
        exp_flag = True
    exp_flag = False
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
        if 'primary' in basepath or 'fbmessages' in basepath or 'workplace' in basepath or 'all' in basepath or 'msg' in basepath or 'bitcoin' in basepath:
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
    x = torch.zeros(len(edge_list_path), node_num, node_num).to(device)
    for i in range(len(edge_list_path)):
        for j, k in edges_list[i]:
            x[i, j, k] = 1
            x[i, k, j] = 1

    return x, edge_index, exp_flag


def train(model: Model, x, edge_index):
    x_pred = model(x, edge_index, True)
    loss = torch.zeros(lookback, model.timestep).to(device)
    for i in range(lookback):
        for j in range(model.timestep):
            loss[i][j] = model.loss(x_pred[j], x[i]) + (lookback - i + j) * theta
            if i == j:
                print('true:', x[i][:8, :8])
                print('pred:', x_pred[j][:8, :8])

    return loss.sum(), x_pred


if __name__ == '__main__':
    edge_index_list: dict
    # data_list = ['cellphone', 'enron', 'enron_large', 'HS11', 'HS12', 'primary', 'workplace']
    # data_list = ['bitcoin_alpha', 'bitcoin_otc', 'college_msg', 'enron_all', 'enron_all_shuffle']
    data_list = ['bitcoin_alpha']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for data in data_list:
        x_list, edge_index_list, exp_flag = process('data/' + data)
        # for lookback in range(1, len(x_list) // 2 + 1):
        for lookback in range(3, 4):
            # for embedding_size in [64, 128, 256]:
            for embedding_size in [128]:
                # for theta in np.arange(0.1, 1.1, 0.1):
                for theta in np.arange(0.5, 0.6, 0.1):
                    for rho in np.arange(0.1, 1.1, 0.1):
                        # timestamp = lookback
                        timestamp = 1
                        MAP_all = []
                        precision_k_all = []

                        # 划分不同组输入 - lookback * 2 + 1
                        # for i in range(x_list.size()[0] - lookback * 2 + 1):
                        for i in range(x_list.size()[0] - lookback - timestamp + 1):
                            encoder = Encoder(in_channels=x_list.size()[1], out_channels=embedding_size).to(device)
                            mllstm = MLLSTM(input_dim=embedding_size, output_dim=embedding_size, n_units=[300, 300]).to(device)
                            model = Model(encoder=encoder, mllstm=mllstm, timestep=timestamp, tau=0.1, beta=1.0, rho=rho).to(device)
                            optimizer = torch.optim.Adam(
                                model.parameters(), lr=0.001, weight_decay=0.0001)
                            start = t()
                            prev = start
                            x_input = x_list[i:i+lookback]
                            edge_index_input = {}
                            for j in range(lookback):
                                edge_index_input[j] = edge_index_list[i + j]
                            model.train()
                            for epoch in range(1, 251):
                                optimizer.zero_grad()
                                loss, _ = train(model, x_input, edge_index_input)
                                loss.backward()
                                optimizer.step()

                                now = t()
                                print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
                                      f'this epoch {now - prev:.4f}, total {now - start:.4f}')
                                prev = now

                            print("=== Finish Training ===")
                            model.eval()
                            _, x_pre_list = train(model, x_input, edge_index_input)

                            # 预测timestamp范围的MAP值
                            MAP_list = []
                            precision_k_list = []
                            for j in range(timestamp):
                                adj_reconstruct = evaluation.evaluation_util.graphify(x_pre_list[j].cpu())
                                edge_index_pre = evaluation.evaluation_util.getEdgeListFromAdj(adj=adj_reconstruct)
                                print('预测得到的边数为', len(edge_index_pre))
                                true_graph = nx.Graph()
                                true_graph.add_nodes_from([i for i in range(x_list.size()[1])])
                                true_graph.add_edges_from(edge_index_list[i + lookback + j].permute(1, 0).cpu().numpy().tolist())
                                MAP, precision_k = evaluation.metrics.computeMAP(edge_index_pre, true_graph)
                                MAP_list.append(MAP)
                                precision_k_list.append(precision_k)
                                print('第' + str(i) + '-' + str(i + lookback) + '个时间片的第' + str(j) + '步预测的MAP值为' + str(MAP))
                            print("=== Finish Evaluating ===")
                            # 不同输入组的MAP值
                            MAP_all.append(MAP_list)
                            precision_k_all.append(precision_k_list)
                        result = {}
                        label = []
                        for i in range(len(MAP_all)):
                            column = '第' + str(i) + '-' + str(i + lookback) + '个时间片'
                            result[column] = MAP_all[i]
                        for j in range(timestamp):
                            row = 'T+' + str(j)
                            label.append(row)
                        # 求所有输入组的平均MAP值
                        MAP_all = np.array(MAP_all)
                        mean_MAP = np.mean(MAP_all, axis=0).tolist()
                        result['mean_MAP'] = mean_MAP
                        for i in range(timestamp):
                            print('预测未来第' + str(i) + '个时间片的mean MAP score is ' + str(mean_MAP[i]))
                        # csv_path = 'result2.0/' + data + '/' + 'lookback=' + str(lookback) + ',embsize=' + str(
                        # embedding_size) + ',theta=' + str(theta) + '.csv'
                        csv_path = 'result2.0/pred_one/rho_analysis/' + data + ',rho=' + str(rho) + '.csv'
                        df = pd.DataFrame(result, index=label)
                        df.to_csv(csv_path)
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
