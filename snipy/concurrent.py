# -*- coding: utf-8 -*-
"""thread and multiprocess common util functions
"""
from functools import wraps
from threading import Thread
from multiprocessing import Pool, Process
from multiprocessing.queues import Queue


class SharedPool(object):
    """
    shared multiprocessing.pool instance
    """
    sharedpool = None

    @classmethod
    def get(cls, processes=None):
        if cls.sharedpool is None:
            cls.set(processes)
        return cls.sharedpool

    @classmethod
    def set(cls, processes=None):
        cls.sharedpool = Pool(processes)


# noinspection PyUnresolvedReferences
class ConcurrentCommon(object):
    """
    by multiprocess! not by thread
    res = Spawn(func, args...)
    print res.result  # wait for result..
    """

    def _maketarget(self, target):
        def _realtarget(*args, **kwargs):
            self.res.put(target(*args, **kwargs))

        return _realtarget

    @property
    def result(self):
        # make func interruptible
        while self.is_alive():
            self.join(0.01)
        return self.res.get()

    @property
    def done(self):
        return not self.is_alive()


class Threaded(Thread, ConcurrentCommon):
    """
    threaded call

    res = Threaded(func, args..., kwargs)
    print res.result  # wait for result..

    """

    def __init__(self, target, *args, **kwargs):
        target = self._maketarget(target)
        super(Threaded, self).__init__(target=target, args=args, kwargs=kwargs)
        self.daemon = True
        self.res = Queue(maxsize=1)
        self.start()


class Spawn(Process, ConcurrentCommon):
    """
    multiprocess call
    """

    def __init__(self, target, *args, **kwargs):
        target = self._maketarget(target)
        super(Spawn, self).__init__(target=target, args=args, kwargs=kwargs)
        self.daemon = True
        self.res = Queue(maxsize=1)
        self.start()


def threaded(f, *args, **kwargs):
    """function decorator
    """
    if args or kwargs:
        return Threaded(f, *args, **kwargs)

    @wraps(f)
    def wrapped(*wargs, **wkwargs):
        return Threaded(f, *wargs, **wkwargs)

    return wrapped


def spawn(f):
    """decorator
    """
    @wraps(f)
    def wrapped(*args, **kwargs):
        return Spawn(f, *args, **kwargs)

    return wrapped


