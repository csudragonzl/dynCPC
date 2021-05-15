import torch

if __name__ == '__main__':
    sim = torch.FloatTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    a = sim.t().mean(axis=0)
    a = torch.tensor([1, 0, 1])
    index = torch.nonzero(a, as_tuple=True)[0]
    sim = sim[index]
    sim = sim[:, index]

    print('1')