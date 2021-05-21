import torch

if __name__ == '__main__':
    sim = torch.FloatTensor([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    a = torch.tensor([[1, 0, 1], [1, 2, 3], [1, 2, 3]])
    index = sim.sum(1) > 0
    a = a[sim.sum(1) > 0]
    a = a[:, sim.sum(1) > 0]
    index = torch.nonzero(a, as_tuple=True)[0]
    sim = sim[index]
    sim = sim[:, index]

    print('1')