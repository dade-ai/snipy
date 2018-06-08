# -*- coding: utf-8 -*-
from .queues import QueueInterruptable, is_main_alive
from .concurrent import Threaded

from .ilogging import logg
from time import sleep


class ActiveQ(QueueInterruptable):

    def __init__(self, maxsize=256, jobq=None, resq=None):
        self._run_thread = None
        self._resq = resq  # result q
        self._jobq = jobq  # input q
        super(ActiveQ, self).__init__(maxsize)

    def start(self):
        # assert self._thread is None
        assert self._run_thread is None
        self._run_thread = Threaded(self._run)
        return self

    def _run(self):
        logg.info('run started')

        while not self.finish and is_main_alive():
            try:
                item = self.get(block=False, timeout=0.01)
            except QueueInterruptable.Empty as e:
                # raise e
                sleep(0.01)
                if self._jobq:
                    try:
                        item = self._jobq.get(block=False)
                    except StopIteration:
                        logg.info('stop iteration')
                        break
                    except QueueInterruptable.Empty as e:
                        continue
                    self.put(item, block=True)
                continue
            # logg.info(item)
            assert isinstance(item, tuple)

            # fun, args, kwargs = item
            # res = fun(*args, **kwargs)
            res = self.action(item)

            if self._resq:
                self._resq.put(res, block=True)

        self._run_thread = None

    def action(self, item):
        """
        for overriding
        :param item:
        :return:
        """
        fun, args, kwargs = item
        return fun(*args, **kwargs)

    def push_job(self, fun, *args, **kwargs):
        """
        put job if possible, non-blocking
        :param fun:
        :param args:
        :param kwargs:
        :return:
        """
        assert callable(fun)
        return self.put((fun, args, kwargs), block=True)

    def put_job(self, fun, *args, **kwargs):
        """
        put job if possible, non-blocking
        :param fun:
        :param args:
        :param kwargs:
        :return:
        """
        if not args and not kwargs and isinstance(fun, (tuple, list)):
            # ex) q.put_job([fun, args, kwargs])
            fun, args, kwargs = fun

        assert callable(fun)
        return self.put((fun, args, kwargs), block=False)


if __name__ == '__main__':
    import sys

    def testfun(i):
        print(i)
        sys.stdout.flush()
        return i

    q = ActiveQ(10).start()

    for i in range(10):
        q.push_job(testfun, i)

    while not q.empty():
        pass

    q.stop()
    print('done')
