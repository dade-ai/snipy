# -*- coding: utf-8 -*-
try:
    import Queue as PQ
except ModuleNotFoundError:
    import queue as PQ

from random import randrange

import errno

from .concurrent import Threaded


class QueueRandom(PQ.Queue):

    """
    queue class for random put or get
    note: not a multiprocess.queue
    """

    def __init__(self, producer=None, maxsize=1024, processor=None):
        # old style class
        PQ.Queue.__init__(self, maxsize=maxsize)
        self.producer = producer
        self.proc = processor
        self.th = None
        self.produce()

    def _init(self, maxsize):
        self.queue = []

    def _produce(self, producer):
        if self.proc:
            for d in producer:
                self.put(self.proc(d))
        else:
            for d in producer:
                self.put(d)

    def produce(self):
        self.th = Threaded(self._produce, self.producer)

    def _get(self):
        # if empty?
        return self.queue.pop(randrange(0, self._qsize()))

    def _put(self, item):
        self.queue.insert(randrange(0, self._qsize() + 1), item)

    def __iter__(self):
        if self.done:
            self.reset()
        return self
        # self.produce()
        # while self.th.is_alive() and not self.full():
        #     pass
        # while not self.done:
        #     yield self.get()

    def next(self):
        while self.th.is_alive() and not self.full():
            pass
        if not self.done:
            return self.get(block=True)
        else:
            raise StopIteration

    # def reset(self):
    #     self.produce()
    reset = produce

    @property
    def done(self):
        if self.th is None:
            return True
        return self.empty() and not self.th.is_alive()


from time import sleep
import threading
import multiprocessing.queues as MQ


def is_main_alive():
    """
    is 메인 쓰레드 alive?
    :rtype: bool
    """

    for t in threading.enumerate():
        if t.name == 'MainThread':
            return t.is_alive()
    print('MainThread not found')
    return False


class QueueInterruptable(MQ.Queue):
    """
    thread 구현용 put에서 blocking 시키지 않음
    """
    Empty = MQ.Empty
    Full = MQ.Full

    def __init__(self, maxsize=0):
        self.finish = False
        super(QueueInterruptable, self).__init__(maxsize=maxsize)

    def get(self, block=True, timeout=None, verbose=True):
        if not block:
            return super(QueueInterruptable, self).get(block=False, timeout=timeout)

        # ignore blocking options for not blocking
        while not self.finish and is_main_alive():
            try:
                data = super(QueueInterruptable, self).get(block=False, timeout=0.01)
                return data
            except MQ.Empty as e:
                # raise e
                sleep(0.01)
        # if verbose:
        #     print('Interrupted or stopped. OK! finish in get')
        # if not is_main_alive():
        #     self.close()
        self.if_no_main_alive_close()

    def put(self, item, block=True, timeout=None, verbose=True):
        # ignore blocking options for not blocking
        while not self.finish and is_main_alive():
            try:
                super(QueueInterruptable, self).put(item, False, timeout=0.01)
                return True
            except MQ.Full:
                sleep(0.01)
        # if verbose:
        #     print('Interrupted or stopped. OK! finish in put')
        self.if_no_main_alive_close()
        return False

    def if_no_main_alive_close(self):
        try:
            if not is_main_alive():
                self.close()
        except IOError as e:
            if e.errno == errno.EPIPE:
                pass

    def stop(self):
        self.finish = True

    def close(self):
        super(QueueInterruptable, self).close()
        self.join_thread()

