import pandas as pd
import numpy as np


def preprocess_bitcoin_otc():
    # SOURCE, TARGET, RATING, TIME
    # SOURCE: node id of source, i.e., rater
    # TARGET: node id of target, i.e., ratee
    # RATING: the source's rating for the target, ranging from -10 to +10 in steps of 1
    # TIME: the time of the rating, measured as seconds since Epoch.
    filepath = '../data/raw_data/soc-sign-bitcoinotc.csv'
    dataframe = pd.read_csv(filepath, header=None, names=['source', 'target', 'rating', 'timestamp'])
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='s').dt.strftime('%Y-%m')
    candidate = ['2010-11', '2010-12', '2011-01', '2011-02', '2011-03', '2011-04', '2011-05',
                 '2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11',
                 '2011-12', '2012-01', '2012-02', '2012-03', '2012-04', '2012-05',
                 '2012-06', '2012-07', '2012-08', '2012-09', '2012-10', '2012-11',
                 '2012-12', '2013-01', '2013-02', '2013-03', '2013-04', '2013-05',
                 '2013-06', '2013-07', '2013-08', '2013-09', '2013-10', '2013-11',
                 '2013-12', '2014-01', '2014-02', '2014-03', '2014-04', '2014-05',
                 '2014-06', '2014-07', '2014-08', '2014-09', '2014-10', '2014-11',
                 '2014-12', '2015-01', '2015-02', '2015-03', '2015-04', '2015-05',
                 '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12', '2016-01']
    snap_candidate = ['2011-05', '2011-11', '2012-05', '2012-11', '2013-05',
                      '2013-11', '2016-01']

    for i in range(len(snap_candidate)):
        if i < 1:
            tem = dataframe[['source', 'target']][dataframe['timestamp'] <= snap_candidate[i]]
        else:
            tem = dataframe[['source', 'target']][
                (dataframe['timestamp'] <= snap_candidate[i]) & (dataframe['timestamp'] > snap_candidate[i - 1])]
        print(i, ':', len(tem))
        tem.to_csv('../data/bitcoin_otc/snapshot' + str(i) + ".edges", sep=' ', header=0, index=0)


def preprocess_bitcoin_alpha():
    # SOURCE, TARGET, RATING, TIME
    # SOURCE: node id of source, i.e., rater
    # TARGET: node id of target, i.e., ratee
    # RATING: the source's rating for the target, ranging from -10 to +10 in steps of 1
    # TIME: the time of the rating, measured as seconds since Epoch.
    filepath = '../data/raw_data/soc-sign-bitcoinalpha.csv'
    dataframe = pd.read_csv(filepath, header=None, names=['source', 'target', 'rating', 'timestamp'])
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='s').dt.strftime('%Y-%m')
    dataframe.sort_values(['timestamp'], inplace=True, ignore_index=True)
    candidate = ['2010-11', '2010-12', '2011-01', '2011-02', '2011-03', '2011-04', '2011-05',
                 '2011-06', '2011-07', '2011-08', '2011-09', '2011-10', '2011-11',
                 '2011-12', '2012-01', '2012-02', '2012-03', '2012-04', '2012-05',
                 '2012-06', '2012-07', '2012-08', '2012-09', '2012-10', '2012-11',
                 '2012-12', '2013-01', '2013-02', '2013-03', '2013-04', '2013-05',
                 '2013-06', '2013-07', '2013-08', '2013-09', '2013-10', '2013-11',
                 '2013-12', '2014-01', '2014-02', '2014-03', '2014-04', '2014-05',
                 '2014-06', '2014-07', '2014-08', '2014-09', '2014-10', '2014-11',
                 '2014-12', '2015-01', '2015-02', '2015-03', '2015-04', '2015-05',
                 '2015-06', '2015-07', '2015-08', '2015-09', '2015-10', '2015-11', '2015-12', '2016-01']
    snap_candidate = ['2011-05', '2011-11', '2012-05', '2012-11', '2013-05',
                      '2013-11', '2016-01']

    for i in range(len(snap_candidate)):
        if i < 1:
            tem = dataframe[['source', 'target']][dataframe['timestamp'] <= snap_candidate[i]]
        else:
            tem = dataframe[['source', 'target']][
                (dataframe['timestamp'] <= snap_candidate[i]) & (dataframe['timestamp'] > snap_candidate[i - 1])]
        print(i, ':', len(tem))
        tem.to_csv('../data/bitcoin_alpha/snapshot' + str(i) + ".edges", sep=' ', header=0, index=0)


def preprocess_college_msg():
    # SRC DST UNIXTS
    # SRC: id of the source node(a user)
    # TGT: id of the target node(a user)
    # UNIXTS: Unix timestamp(seconds since the epoch)
    filepath = '../data/raw_data/CollegeMsg.txt'
    file = open(filepath, 'r')
    dataframe = pd.DataFrame(list(y.split(' ') for y in file.read().split('\n'))[:-1], columns=['source', 'target', 'timestamp'])
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
    dataframe.sort_values(['timestamp'], inplace=True, ignore_index=True)
    snap_candidate = ['2004-04-30', '2004-05-06', '2004-05-11', '2004-05-16', '2004-05-21', '2004-05-26',
                      '2004-05-31', '2004-06-30', '2004-07-31', '2004-10-31']
    for i in range(len(snap_candidate)):
        if i < 1:
            tem = dataframe[['source', 'target']][dataframe['timestamp'] <= snap_candidate[i]]
        else:
            tem = dataframe[['source', 'target']][
                (dataframe['timestamp'] <= snap_candidate[i]) & (dataframe['timestamp'] > snap_candidate[i - 1])]
        print(i, ':', len(tem))
        tem.to_csv('../data/college_msg/snapshot' + str(i) + ".edges", sep=' ', header=0, index=0)


def preprocess_enron_all():
    filepath = '../data/raw_data/ia-enron-email-dynamic.edges'
    file = open(filepath, 'r')
    dataframe = pd.DataFrame(list(y.split(' ') for y in file.read().split('\n'))[:-1], columns=['source', 'target', 'label', 'timestamp'])
    dataframe['timestamp'] = pd.to_datetime(dataframe['timestamp'], unit='s').dt.strftime('%Y-%m-%d')
    dataframe.sort_values(['timestamp'], inplace=True, ignore_index=True)
    good_candidate = ['2002-01-01', '2002-01-02', '2002-01-03', '2002-01-06', '2002-01-07',
                      '2002-01-08', '2002-01-09', '2002-01-10', '2002-01-11', '2002-01-14']
    dataframe = dataframe[['source', 'target', 'timestamp']][
        (dataframe['timestamp'] <= good_candidate[-1]) & (dataframe['timestamp'] > good_candidate[0])]
    nodes_set = pd.concat([dataframe['source'], dataframe['target']], ignore_index=True).unique()
    np.random.shuffle(nodes_set)
    node_dataframe = pd.DataFrame(nodes_set, columns=['node'])
    for i in range(1, len(good_candidate)):
        tem = dataframe[['source', 'target']][
            (dataframe['timestamp'] <= good_candidate[i]) & (dataframe['timestamp'] > good_candidate[i - 1])]
        tem['source'] = tem['source'].map(lambda x: node_dataframe[(node_dataframe['node'] == x)].index.tolist()[0] + 1)
        tem['target'] = tem['target'].map(lambda x: node_dataframe[(node_dataframe['node'] == x)].index.tolist()[0] + 1)
        print(i, ':', len(tem))
        tem.to_csv('../data/enron_all_shuffle/snapshot' + str(i) + ".edges", sep=' ', header=0, index=0)


if __name__ == '__main__':
    preprocess_enron_all()
