import pandas as pd


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
                      '2013-11', '2014-05', '2014-11', '2015-05', '2016-01']

    for i in range(len(snap_candidate)):
        if i < 1:
            tem = dataframe[['source', 'target']][dataframe['timestamp'] <= snap_candidate[i]]
        else:
            tem = dataframe[['source', 'target']][dataframe['timestamp'] <= snap_candidate[i] and dataframe['timestamp'] > snap_candidate[i-1]]
        print('1')

def preprocess_bitcoin_alpha():
    # SOURCE, TARGET, RATING, TIME
    # SOURCE: node id of source, i.e., rater
    # TARGET: node id of target, i.e., ratee
    # RATING: the source's rating for the target, ranging from -10 to +10 in steps of 1
    # TIME: the time of the rating, measured as seconds since Epoch.
    filepath = '../data/raw_data/soc-sign-bitcoinalpha.csv'
    data = pd.read_csv(filepath)
    print('1')


def preprocess_college_msg():
    # SRC DST UNIXTS
    # SRC: id of the source node(a user)
    # TGT: id of the target node(a user)
    # UNIXTS: Unix timestamp(seconds since the epoch)
    filepath = '../data/raw_data/CollegeMsg.txt'
    data = pd.read_csv(filepath)
    print('1')


if __name__ == '__main__':
    preprocess_bitcoin_otc()
