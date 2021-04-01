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


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, mllstm: MLLSTM, timestep, tau: float = 0.5, activation = torch.sigmoid):
        super(Model, self).__init__()
        self.encoder = encoder
        self.mllstm = mllstm
        # timestep是预测的时间步长
        self.timestep = timestep
        self.tau = tau
        self.Wk = nn.ModuleList([nn.Linear(mllstm.output_dim, encoder.input_dim, bias=False) for i in range(timestep)])
        self.activation = activation

    def forward(self, x: torch.tensor, edge_index: dict, link_pred: bool):
        # x是所有时间片的集合
        nodes_num = x.size()[1]
        x_encoded = torch.empty(x.size()[0], x.size()[1], 64)
        for i in range(x.size()[0]):
            x_encoded[i] = self.encoder(x[i], edge_index[i])
        _, ct = self.mllstm(x_encoded[:])
        if link_pred:
            pred = torch.empty((self.timestep, nodes_num, nodes_num))
        else:
            pred = torch.empty((self.timestep, nodes_num, 256))
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            if link_pred:
                pred[i] = self.activation(linear(ct))
            else:
                pred[i] = linear(ct)
        return pred

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(torch.mm(z1, z2.t()))
        # pos = between_sim.diag()
        # res = torch.zeros(pos.size())
        # for i in range(pos.size()[0]):
        #     node_indices = np.append(np.arange(0, i), np.arange(i + 1, z1.size()[0]))
        #     np.random.shuffle(node_indices)
        #     node_indices = node_indices[:len(node_indices) // 10]
        #     res[i] = pos[i] / (between_sim[i, node_indices].sum(0) + pos[i])

        return -torch.log(
            between_sim.diag() / between_sim.sum(1)
        )

    def loss(self, z1, z2, mean: bool = True):
        ret = self.semi_loss(z1, z2)
        ret = ret.mean() if mean else ret.sum()

        return ret


def process(basepath: str):
    edge_list_path = os.listdir(basepath)
    if 'enron' in basepath:
        edge_list_path.sort(key=lambda x: int(x[5:-6]))
    elif 'HS11' in basepath:
        edge_list_path.sort(key=lambda x: int(x[-5:-4]))
    elif 'HS12' in basepath:
        edge_list_path.sort(key=lambda x: int(x[-11:-10]))
    elif 'primary' in basepath:
        edge_list_path.sort(key=lambda x: int(x[-5:-4]))
    node_num = 0
    edge_index = {}
    edges_list = []
    for i in range(len(edge_list_path)):
        file = open(os.path.join(basepath, edge_list_path[i]), 'r')
        if 'primary' in basepath:
            edges = list(y.split(' ')[:-1] for y in file.read().split('\n'))[:-1]
        else:
            edges = list(y.split('\t') for y in file.read().split('\n'))[:-1]
        edges = list(set([tuple(t) for t in edges]))
        for j in range(len(edges)):
            edges[j] = list(int(z) - 1 for z in edges[j])
            for z in edges[j]:
                node_num = max(node_num, z)
        edges_list.append(edges)
        edges = torch.tensor(edges).permute(1, 0)
        edge_index[i] = edges

    node_num += 1
    x = torch.zeros(len(edge_list_path), node_num, node_num)
    for i in range(len(edge_list_path)):
        for j, k in edges_list[i]:
            x[i, j, k] = 1

    return x, edge_index


def train(model: Model, x, edge_index, lookback=3):
    x_pred = model(x, edge_index, True)
    loss = torch.zeros(lookback, model.timestep)
    for i in range(lookback):
        for j in range(model.timestep):
            loss[i][j] = model.loss(x[i], x_pred[j]) + (lookback - i + j) * 0.2

    return loss.sum(), x_pred


if __name__ == '__main__':
    edge_index_list: dict
    data_list = ['enron', 'HS11', 'HS12', 'primary']
    for data in data_list:
        x_list, edge_index_list = process('data/' + data)
        lookback = 3
        timestamp = 3
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        MAP_all = []
        # x_list.size()[0] - lookback
        for i in range(x_list.size()[0] - lookback * 2 + 1):
            encoder = Encoder(in_channels=x_list.size()[1], out_channels=64).to(device)
            mllstm = MLLSTM(input_dim=64, output_dim=64, n_units=[300, 300]).to(device)
            model = Model(encoder=encoder, mllstm=mllstm, timestep=timestamp).to(device)
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

            MAP_list = []
            for j in range(timestamp):
                adj_reconstruct = evaluation.evaluation_util.graphify(x_pre_list[j])
                edge_index_pre = evaluation.evaluation_util.getEdgeListFromAdj(adj=adj_reconstruct)
                true_graph = nx.Graph()
                true_graph.add_nodes_from([i for i in range(x_list.size()[1])])
                true_graph.add_edges_from(edge_index_list[i + lookback + j].permute(1, 0).numpy().tolist())
                MAP = evaluation.metrics.computeMAP(edge_index_pre, true_graph)
                MAP_list.append(MAP)
                print('第' + str(i) + '-' + str(i + lookback) + '个时间片的第' + str(j) + '步预测的MAP值为' + str(MAP))
            print("=== Finish Evaluating ===")
            MAP_all.append(MAP_list)
        MAP_all = np.array(MAP_all).T
        for i in range(timestamp):
            print('预测未来第' + str(i) + '个时间片的mean MAP score is ' + str(np.mean(MAP_all, axis=1)[i]))
        with open('result/' + data + '/MAP.txt', mode='w+') as file:
            file.write('数据集共有' + str(x_list.size()[0]) + '个时间片\n')
            file.write('lookback的值为' + str(lookback) + ' 预测未来' + str(timestamp) + '步的时间片\nMAP的值分别为：\n')
            for i in range(MAP_all.shape[0]):
                file.write('预测未来第' + str(i) + '步：')
                for MAP in MAP_all[i]:
                    file.write(str(MAP) + ' ')
                file.write('\n')
                file.write('mean MAP: ' + str(np.mean(MAP_all[i])) + '\n')
