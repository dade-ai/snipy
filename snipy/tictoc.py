# -*- coding: utf-8 -*-
import time
from contextlib import contextmanager

from . ilogging import logg


def tic():
    """
    ex1)
    tic()  # save start time - time1
    toc()  # print elapsed time from last calling tic()
    toc()  # print elapsed time from last calling tic()

    ex2)
    t0 = tic()  # simple
    t1 = tic()
    toc(t1)  # print time from t1
    toc(t0)  # print time from t0

    :return: time:
    """
    t = time.time()
    setattr(tic, 'last_tic_time', t)
    return t


# noinspection PyUnresolvedReferences
def toc(t=None, name='tictoc'):
    """
    ex1)
    tic()  # save start time - time1
    toc()  # print elapsed time from last calling tic()
    toc()  # print elapsed time from last calling tic()

    ex2)
    t0 = tic()  # simple
    t1 = tic()
    toc(t1)  # print time from t1
    toc(t0)  # print time from t0

    :param t: time: 시작 시간 (tic()의 리턴 값)
    :param name: str: 출력시 포함할 문자 ['tictoc']
    """
    try:
        t = t or tic.last_tic_time
    except AttributeError:
        # tic()부터 콜하세요
        logg.warn('calling tic() need to use toc()')
        return
    elapsed = time.time() - t
    logg.info('%s Elapsed: %s secs' % (name, elapsed))
    return elapsed


@contextmanager
def tictoc(name='tictoc'):
    """
    with tictoc('any string or not'):
        print 'cool~~~'
    cool~~~
    2015-12-30 14:39:28,458 [INFO] tictoc Elapsed: 7.10487365723e-05 secs
    :param name: str
    """
    t = time.time()
    yield
    logg.info('%s Elapsed: %s secs' % (name, time.time() - t))

