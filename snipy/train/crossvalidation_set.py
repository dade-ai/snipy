# -*- coding: utf-8 -*-
import numpy as np
from ..irandom import np_seed


def split_rand(data_or_size, ratio, seed):
    """
    data(1-ratio), data(with ratio) = split_rand(data_or_size, ratio, seed)
    :param data_or_size: data or count
    :param ratio:
    :param seed:
    :return:
    """
    if not isinstance(data_or_size, int):
        sz = len(data_or_size)
        data = np.asarray(data_or_size)
    else:
        sz = data_or_size
        data = np.arange(sz)
    if not ratio:
        return data, []

    i = np.zeros(sz, dtype='bool')
    lattersz = int(sz * ratio)
    i[:lattersz] = True

    with np_seed(seed):
        np.random.shuffle(i)

    return data[~i], data[i]


def testsets(data_or_sz, p_testset=None, seed=7238):
    """

    :param data_or_sz:
    :param p_testset:
    :param seed:
    :return:
    """
    trains, tests = split_rand(data_or_sz, p_testset, seed)
    return tests, trains


def kfolds(n, k, sz, p_testset=None, seed=7238):
    """
    return train, valid  [,test]
    testset if p_testset
    :param n:
    :param k:
    :param sz:
    :param p_testset:
    :param seed:
    :return:
    """
    trains, tests = split_rand(sz, p_testset, seed)

    ntrain = len(trains)

    # np.random.seed(seed)
    with np_seed(seed):
        np.random.shuffle(trains)
    if n == k:
        # no split
        train, valid = trains, trains
    else:
        foldsz = ntrain // k

        itrain = np.arange(ntrain) // foldsz != n
        train = trains[itrain]
        valid = trains[~itrain]

    if not p_testset:
        return train, valid
    else:
        return train, valid, tests


if __name__ == '__main__':
    out = kfolds(2, 4, 100, 0.2)
    print out
    print map(len, out)

