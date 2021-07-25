import torch
import pandas as pd


if __name__ == '__main__':
    # tem = pd.DataFrame({'source': [0, 1, 2, 3], 'target': [1, 2, 3, 0]})
    # node_dataframe = pd.DataFrame({'node': [0, 1, 2]})
    # tem = tem[tem.apply(lambda x: (x['source'] in node_dataframe['node']) and (x['target'] in node_dataframe['node']),
    #                     axis=1)]
    # a1 = torch.FloatTensor([[1, 2],
    #                         [1, 3]])
    # b1 = torch.FloatTensor([[1, 1],
    #                         [1, 1]])
    sim = torch.FloatTensor([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
    a = torch.tensor([[1, 0, 1], [1, 2, 3], [1, 2, 3]])
    index = sim.sum(1) > 0
    # a = a[sim.sum(1) > 0]
    # a = a[:, sim.sum(1) > 0]
    # index = torch.nonzero(a, as_tuple=True)[0]
    # sim = sim[index]
    # sim = sim[:, index]

    print('1')