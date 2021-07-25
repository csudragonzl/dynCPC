import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np


color = ['#7DBF43', '#ED926B', '#91A0C7', '#DA8FC0', '#B0D667',
         '#FAD856','#E0C499', '#B3B3B3']

font = {'weight': 'normal',
        'size': 33,
        }

font1 = {'weight': 'normal',
         'size': 30,
         }

font3 = {'weight': 'normal',
         'size': 38,
         }


def analyze_lookback(basepath=['enron']):
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


def analyze_embed_size(basepath=['HS11', 'HS12', 'enron', 'cellphone']):
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
    results_map = np.array(results).T
    results_auc = np.array([[0.7224, 0.8127, 0.8240, 0.8571],
               [0.7328, 0.8235, 0.8497, 0.8682],
               [0.7502, 0.8301, 0.8623, 0.8902]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # title_ord = ["a", "b", "c", "d", "e", "f"]
    # for i in range(1, 3):
    #     ax_tmp = fig.add_subplot(1, 2, i)
    #     ax_tmp.set_title("({}) ".format(title_ord[i - 1]), {'weight': 'normal', 'size': 20}, y=-0.25)
    #     ax_tmp.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
    #     ax_tmp.xaxis.set_major_locator(plt.NullLocator())
    #     ax_tmp.xaxis.set_major_formatter(plt.NullFormatter())
    #     ax_tmp.yaxis.set_major_locator(plt.NullLocator())
    #     ax_tmp.yaxis.set_major_formatter(plt.NullFormatter())
    #     ax_tmp._frameon = False

    total_width, n = 0.8, len(embsize_list)
    width = total_width / n
    x = np.arange(results_map.shape[1])
    x = x - (total_width - width) / 2
    labels = ['d=64', 'd=128', 'd=256']
    for i in range(results_map.shape[0]):
        ax1.bar(x + i * width, results_map[i], width=width*0.8, label='d=' + str(embsize_list[i]), color=color[i])
        ax2.bar(x + i * width, results_auc[i], width=width*0.8, label='d=' + str(embsize_list[i]), color=color[i])
    # 绘制y轴的网格线
    # ax1.grid(axis='y', linestyle='--')
    # 将网格线设置在图形下方
    # ax1.gca().set_axisbelow(True)
    # 设置x轴刻度的位置
    if results_map.shape[0] % 2 != 0:
        x_ticks = (x + results_map.shape[0] // 2 * width)
    else:
        x_ticks = (x + (results_map.shape[0] / 2 + 0.5) * width)
    datas = ['HS11', 'HS12', 'ENRON', 'Cellphone']
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(datas)
    # ax1.set_xlabel('Dataset')
    ax1.set_ylabel('MAP')
    ax1.set_ylim(bottom=0.2)
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))

    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(datas)
    # ax1.set_xlabel('Dataset')
    ax2.set_ylabel('AUC')
    ax2.set_ylim(bottom=0.5)
    ax2.yaxis.set_major_locator(MultipleLocator(0.1))
    fig.text(0.5, 0.0, 'Dataset', ha='center', fontsize=12)

    fig.legend(
        labels=labels, loc="upper center",  # Position of legend
        borderaxespad=0.5,  # Small spacing around legend box
        ncol=3,
        prop={'weight': 'normal', 'size': 10})
    fig.savefig('pdf/dimension.pdf', format='pdf', bbox_inches='tight')


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
    alogs = ['DynAE', 'DynRNN', 'DynAERNN', 'CTLP']
    x = np.arange(5, 11)
    fig, subs = plt.subplots(2, 3)
    for j in range(2):
        for k in range(3):
            for i in range(len(results) - 1):
                subs[j][k].plot(x, results[i], label=alogs[i], color=color[i], marker='o', markersize=5, linestyle="--")
            subs[j][k].plot(x, results[-1], label=alogs[-1], color=color[6], marker='o', markersize=5)
            subs[j][k].set_xlabel('Snapshot')
            subs[j][k].set_ylabel('MAP scores')
    fig.subplots_adjust(top=0.9, left=0.1, right=0.9,
                        bottom=0.12)  # create some space below the plots by increasing the bottom-value
    subs.flatten()[-2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)
    # it would of course be better with a nicer handle to the middle-bottom axis object, but since I know it is the second last one in my 3 x 3 grid...
    # plt.legend(ncol=4, loc=9)
    # plt.ylim((0, 1.0))
    plt.savefig('pics/analyze_alogs_plot.svg')
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


def analyze():
    color_pool = ['#1ABC9C', '#2980B9', '#009966', '#F39C12', '#996600', '#F7DC6F', '#8E44AD',
                  '#5DADE2', '#E74C3C', '#27AE60', '#F39C12', '#EDBB99', '#F5B7B1', '#FAD7A0']

    marker = ['o', '^', 'v', '>', '<', 's', 'd', 'p', 'X', '+']
    models = ["DeepWalk", "LINE", "DynAE", "DynRNN", "DynAERNN", "DySAT", "VGRNN", "TMF", "CTLP"]
    datasets = ["HS11", "HS12", "ENRON", "Cellphone", "UCIMsg", "BitcoinAlpha"]
    datas = ["HS11", "HS12", "enron", "cellphone", "ucimsg", "bitcoin_alpha"]
    title_ord = ["a", "b", "c", "d", "e", "f"]
    fig, ax = plt.subplots(3, 4, figsize=(40, 27))

    for i in range(1, 7):
        ax_tmp = fig.add_subplot(3, 2, i)
        ax_tmp.set_title("({}) ".format(title_ord[i - 1]), font3, y=-0.25)
        ax_tmp.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        ax_tmp.xaxis.set_major_locator(plt.NullLocator())
        ax_tmp.xaxis.set_major_formatter(plt.NullFormatter())
        ax_tmp.yaxis.set_major_locator(plt.NullLocator())
        ax_tmp.yaxis.set_major_formatter(plt.NullFormatter())
        ax_tmp._frameon = False

    ymajorFormatter = FormatStrFormatter('%.2f')

    j = 0
    for data in datas:
        for i in range(2):
            if i % 2 == 0:
                measure = 'map'
            else:
                measure = 'auc'
            results = pd.read_csv('{}_{}.txt'.format(data, measure), encoding='utf-8', sep=',', header=None)
            if data in ["HS11", "bitcoin_alpha"]:
                x = range(4, 8)
                x_ticks = range(4, 8)
            elif data in ["HS12"]:
                x = range(4, 9)
                x_ticks = range(4, 9)
            elif data in ["enron"]:
                x = range(4, 13)
                x_ticks = range(4, 13)
            elif data in ["cellphone", "ucimsg"]:
                x = range(4, 11)
                x_ticks = range(4, 11)

            for k in range(len(models)):
                y = results[k: k + 1].values[0][: -1]
                if k < 8:
                    ax[int(j / 4)][int(j % 4)].plot(x, y, marker=marker[0], c=color_pool[k],
                                                    linewidth=3.0, ms=10, linestyle='--')
                else:
                    ax[int(j / 4)][int(j % 4)].plot(x, y, marker=marker[0], c=color_pool[k],
                                                    linewidth=3.0, ms=12)
            results = results.values
            for l in range(results.shape[0]):
                for m in range(results.shape[1]):
                    if np.isnan(results[l][m]):
                        results[l][m] = results[-1][-1]
            min_val = results.min()
            max_val = results.max()
            # if min_ylabel + 0.05 < min_val:
            #     min_ylabel += 0.05
            if (max_val - min_val) / 0.05 > 15:
                ymajorLocator = MultipleLocator(0.25)
            elif (max_val - min_val) / 0.05 > 8:
                ymajorLocator = MultipleLocator(0.15)
            elif (max_val - min_val) / 0.02 > 8:
                ymajorLocator = MultipleLocator(0.10)
            else:
                ymajorLocator = MultipleLocator(0.05)

            ax[int(j / 4)][int(j % 4)].set_ylabel(measure.upper(), font1)
            ax[int(j / 4)][int(j % 4)].set_xlabel('Snapshot', font1)
            ax[int(j / 4)][int(j % 4)].set_xticks(x_ticks)
            # ax[int(j/4)][int(j%4)].set_yticks(y_ticks)
            ax[int(j / 4)][int(j % 4)].yaxis.set_major_locator(ymajorLocator)
            ax[int(j / 4)][int(j % 4)].yaxis.set_major_formatter(ymajorFormatter)
            ax[int(j / 4)][int(j % 4)].tick_params(labelsize=28)
            j += 1

    fig.legend(
        labels=models, loc="upper center",  # Position of legend
        borderaxespad=0.5,  # Small spacing around legend box
        ncol=9,
        prop=font)

    # Adjust the scaling factor to fit your legend text completely outside the plot
    # (smaller value results in more space being made for the legend)
    # plt.subplots_adjust(right=0.95)
    fig.tight_layout(pad=10.5, w_pad=1.2, h_pad=-4.5)
    fig.savefig('pdf/results.pdf', format='pdf', bbox_inches='tight')


def analyze_multi():
    datasets = ["HS12", "ENRON", "Cellphone", "UCIMsg"]
    algos = ['DySAT', 'VGRNN', 'TMF', 'CTLP']
    title_ord = ["a", "b", "c", "d"]
    fig, ax = plt.subplots(2, 4, figsize=(40, 18))
    total_width, n = 0.8, 3
    width = total_width / n
    x = np.arange(4)
    x = x - (total_width - width) / 2
    labels = ['l+1', 'l+2', 'l+3']
    x_ticks = x + width
    ymajorFormatter = FormatStrFormatter('%.2f')

    for i in range(1, 5):
        ax_tmp = fig.add_subplot(2, 2, i)
        ax_tmp.set_title("({}) ".format(title_ord[i - 1]), {'weight': 'normal', 'size': 35}, y=-0.2)
        ax_tmp.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        ax_tmp.xaxis.set_major_locator(plt.NullLocator())
        ax_tmp.xaxis.set_major_formatter(plt.NullFormatter())
        ax_tmp.yaxis.set_major_locator(plt.NullLocator())
        ax_tmp.yaxis.set_major_formatter(plt.NullFormatter())
        ax_tmp._frameon = False

    j = 0
    for data in datasets:
        for i in range(2):
            if i % 2 == 0:
                measure = 'map'
            else:
                measure = 'auc'
            results = pd.read_csv('{}_m_{}.txt'.format(data.lower(), measure), encoding='utf-8', sep=',', header=None)
            results = results.values.T
            for k in range(len(labels)):
                ax[int(j / 4)][int(j % 4)].bar(x + k * width, results[k], width=width * 0.8, label=labels[k],
                        color=color[k])
            if i % 2 != 0:
                ax[int(j / 4)][int(j % 4)].set_ylim(bottom=0.4)
                ax[int(j / 4)][int(j % 4)].yaxis.set_major_locator(MultipleLocator(0.1))
            ax[int(j / 4)][int(j % 4)].set_xticks(x_ticks)
            ax[int(j / 4)][int(j % 4)].set_xticklabels(algos)
            if j == 6:
                ax[int(j / 4)][int(j % 4)].yaxis.set_major_locator(MultipleLocator(0.03))
            ax[int(j / 4)][int(j % 4)].yaxis.set_major_formatter(ymajorFormatter)
            ax[int(j / 4)][int(j % 4)].set_ylabel(measure.upper(), {'weight': 'normal', 'size': 28})
            ax[int(j / 4)][int(j % 4)].tick_params(labelsize=28)
            j += 1
    fig.legend(
        labels=labels, loc="upper center",  # Position of legend
        borderaxespad=0.5,  # Small spacing around legend box
        ncol=3,
        prop={'weight': 'normal', 'size': 30, 'family': 'Latin Modern Math'})


    fig.tight_layout(pad=10.5, w_pad=1.2, h_pad=-1)
    fig.savefig('pdf/multi-step.pdf', format='pdf', bbox_inches='tight')


if __name__ == '__main__':
    analyze_multi()
    # result = []
    # for i in range(4):
    #     result.append(np.random.randint(400, 700) / 10000)
    # a = np.mean(np.array(result))
    # result.append(a)
    # print(result)
