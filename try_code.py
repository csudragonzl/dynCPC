import torch

if __name__ == '__main__':
    ct = torch.tensor([[1, 2, 3], [4, 5, 6]])
    a = ct.resize(2, 3)
    nodes_num = ct.size()[0]
    ct_composed = torch.zeros(nodes_num ** 2, ct.size()[1] * 2)
    for i in range(nodes_num):
        ct_composed[i * nodes_num + i + 1:(i + 1) * nodes_num, :ct.size()[1]] = ct[i]
        ct_composed[i * nodes_num + i + 1:(i + 1) * nodes_num, ct.size()[1]:] = ct[i + 1:]
    print('1')