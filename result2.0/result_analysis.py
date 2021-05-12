import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np


color = ['#95a2ff', '#fa8080', '#fae768', '#87e885', '#3cb9fc',
         '#73abf5', '#cb9bff', '#434348', '#90ed7d', '#f7a35c', '#8085e9']


def analyze_lookback(basepath='enron'):
    result_list = os.listdir(basepath)
    result_list = [name for name in result_list if 'embsize=128,theta=0.5' in name]
    result_list.sort(key=lambda x: int(x[9:10]))
    results = []
    for result_file in result_list:
        result_data = pd.read_csv(os.path.join(basepath, result_file))
        result_tmp = []
        for mean_MAP in result_data['mean_MAP']:
            result_tmp.append(mean_MAP)
        results.append(result_tmp)

    total_width, n = 0.8, len(results[-1])
    width = total_width / n
    x = []
    y = []
    count = 1
    while len(x) < len(results):
        x_tmp = []
        y_tmp = []
        for i in range(len(results)):
            if len(results[i]) < count:
                continue
            base_width = (count - 1) * width + i * (i + 1) / 2 * width + (i + 1) * width
            x_tmp.append(base_width)
            y_tmp.append(results[i][count - 1])
        count += 1
        x.append(x_tmp)
        y.append(y_tmp)
    for i in range(len(x)):
        plt.bar(x[i], y[i], width=width, label='T+' + str(i), color=color[i])
    plt.legend(ncol=4, loc=9)
    plt.ylim((0, 1.0))
    # 绘制y轴的网格线
    plt.grid(axis='y', linestyle='--')
    # 将网格线设置在图形下方
    plt.gca().set_axisbelow(True)
    # 设置x轴刻度的位置
    x_ticks = []
    for i in range(len(x)):
        if i > 0:
            x_ticks.append(x_ticks[-1] + (i + 1 + 0.5) * width)
        else:
            x_ticks.append((i + 1) * width)
    plt.xticks(x_ticks, [i for i in range(1, len(x) + 1)])
    plt.xlabel('Lookback numbers')
    plt.ylabel('Mean MAP')
    plt.show()


def analyze_embed_size(basepath=['cellphone', 'enron', 'HS11', 'HS12', 'workplace']):
    file_name = 'lookback=3,embsize={},theta=0.2.csv'
    # 用预测的第T+index步的值
    index = 2
    embsize_list = [64, 128, 256]
    results = []
    for data in basepath:
        result_tmp = []
        for embsize in embsize_list:
            result_data = pd.read_csv(os.path.join(data, file_name.format(embsize)))
            result_tmp.append(result_data['mean_MAP'][index])
        results.append(result_tmp)
    results = np.array(results).T
    total_width, n = 0.8, len(embsize_list)
    width = total_width / n
    x = np.arange(results.shape[1])
    x = x - (total_width - width) / 2
    for i in range(results.shape[0]):
        plt.bar(x + i * width, results[i], width=width, label='embedding_size=' + str(embsize_list[i]), color=color[i])
    plt.legend()
    plt.ylim((0, 1.0))
    # 绘制y轴的网格线
    plt.grid(axis='y', linestyle='--')
    # 将网格线设置在图形下方
    plt.gca().set_axisbelow(True)
    # 设置x轴刻度的位置
    if results.shape[0] % 2 != 0:
        x_ticks = (x + results.shape[0] // 2 * width)
    else:
        x_ticks = (x + (results.shape[0] / 2 + 0.5) * width)
    plt.xticks(x_ticks, basepath)
    plt.xlabel('Dataset')
    plt.ylabel('Mean MAP')
    plt.show()


def analyze_theta(basepath=['cellphone', 'enron', 'HS11', 'HS12', 'workplace']):
    file_name = 'lookback=3,embsize=128,theta={}.csv'
    # 用预测的第T+index步的值
    index = 2
    results = []
    x = np.arange(0.1, 1.1, 0.1)
    for data in basepath:
        result_tmp = []
        for theta in x:
            result_data = pd.read_csv(os.path.join(data, file_name.format(theta)))
            result_tmp.append(result_data['mean_MAP'][index])
        results.append(result_tmp)
    for i in range(len(results)):
        plt.plot(x, results[i], label=basepath[i], color=color[i], marker='o', markersize=5)
    plt.legend(ncol=4, loc=9)
    plt.ylim((0, 1.0))
    plt.xlabel('theta')
    plt.ylabel('Mean MAP')
    plt.show()


def analyze_algos(basepath=['cellphone', 'enron', 'HS11', 'HS12', 'primary', 'workplace']):
    results = []
    result_temp = []
    for data in basepath:
        result_data = pd.read_csv(os.path.join('pred_one', data + '.csv'))
        result_temp.append(result_data['mean_MAP'][0])
    results.append(result_temp)
    x_ticks_labels = ['DynCPC', 'DynAE', 'DynRNN', 'DynAERNN']
    total_width, n = 0.8, len(results)
    width = total_width / n
    x = np.arange(len(basepath))
    x = x - (total_width - width) / 2
    for i in range(len(results)):
        plt.bar(x + i * width, results[i], width=width, label=x_ticks_labels[i], color=color[i])
    plt.legend()
    plt.ylim((0, 1.0))
    # 绘制y轴的网格线
    plt.grid(axis='y', linestyle='--')
    # 将网格线设置在图形下方
    plt.gca().set_axisbelow(True)
    # 设置x轴刻度的位置
    if len(results) % 2 != 0:
        x_ticks = (x + len(results) // 2 * width)
    else:
        x_ticks = (x + (len(results) / 2 + 0.5) * width)
    plt.xticks(x_ticks, basepath)
    plt.ylabel('Mean MAP')
    plt.show()


def analyze_alogs_plot(basepath='cellphone'):
    results = []
    result_data3 = pd.read_csv(os.path.join('pred_one', basepath + '/DynAE.csv'))['MAP值'].tolist()[:-1]
    results.append(result_data3)
    result_data4 = pd.read_csv(os.path.join('pred_one', basepath + '/DynRNN.csv'))['MAP值'].tolist()[:-1]
    results.append(result_data4)
    result_data5 = pd.read_csv(os.path.join('pred_one', basepath + '/DynAERNN.csv'))['MAP值'].tolist()[:-1]
    results.append(result_data5)
    result_data8 = pd.read_csv(os.path.join('pred_one', basepath + '/TGC.csv')).values.tolist()[0][2:-1]
    results.append(result_data8)

    # alogs = ['DeepWalk', 'DGI', 'DynAE', 'DynRNN', 'DynAERNN', 'VGRNN', 'CTGCN', 'TGC']
    alogs = ['DynAE', 'DynRNN', 'DynAERNN', 'TGC']
    x = np.arange(5, 11)
    for i in range(len(results)):
        plt.plot(x, results[i], label=alogs[i], color=color[i], marker='o', markersize=5)
    plt.legend(ncol=4, loc=9)
    plt.ylim((0, 1.0))
    plt.xlabel('Snapshots of Cellphone Graphs')
    plt.ylabel('MAP scores')
    plt.savefig('pics/analyze_alogs_plot.svg', dpi=300)
    plt.show()


def analyze_algos_multi():
    tgc = [[0.8744305794224451, 0.864398870231273, 0.8642806466948763],
           [0.5984862225182755, 0.5502274079634539, 0.5190236001771468],
           [0.26717326835527605, 0.24601435224145132, 0.22033711599579955]]
           # 'HS11': [0.3830546557920702, 0.4784097760795658, 0.4526309180374959],
           # 'HS12': [0.4474901625890099, 0.42989769966850444, 0.4397206472960196],
           # 'Workplace': [0.4626418119533488, 0.47392299806266713, 0.46778719885289516]}
    vgrnn_m = []
    x_ticks_labels = ['Cellphone', 'Enron', 'Enron_all']
    bar_labels = ['TGC']
    # , 'HS11', 'HS12', 'Workplace']
    total_width, n = 0.8, len(x_ticks_labels) * 3
    width = total_width / n
    x = np.arange(len(x_ticks_labels))
    x = x - (total_width - width) / 2
    for i in range(len(tgc)):
        plt.bar([x[i] + j * width for j in range(len(tgc))], tgc[i], width=width, label=bar_labels[0], color=color[1])
    plt.legend()
    plt.ylim((0, 1.0))
    # 绘制y轴的网格线
    plt.grid(axis='y', linestyle='--')
    # 将网格线设置在图形下方
    plt.gca().set_axisbelow(True)
    # 设置x轴刻度的位置
    if len(tgc) % 2 != 0:
        x_ticks = (x + len(tgc) // 2 * width * 3)
    else:
        x_ticks = (x + (len(tgc) / 2 + 0.5) * width * 3)
    plt.xticks(x_ticks, x_ticks_labels)
    plt.ylabel('Mean MAP')
    plt.show()


if __name__ == '__main__':
    analyze_algos_multi()