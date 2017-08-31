# -*- coding: utf-8 -*-
import functools
from .basic import tuple_args

from .concurrent import Threaded
from .listx import ListCallable, ListArgs


def get_call_chain(something):
    try:
        return something.get_call_chain()
    except AttributeError:
        return [something]


class IterChain(object):
    def __init__(self, left, right):
        self._iterables = get_call_chain(left) + get_call_chain(right)

    def get_call_chain(self):
        return self._iterables

    def __call__(self, it):
        chained = it
        for i in self._iterables:
            chained = i(chained)
        return chained

    def __iter__(self):
        chained = self._iterables[0]
        for i in self._iterables[1:]:
            chained = i(chained)
        return chained

    def __rshift__(self, right):
        return IterChain(self, right)

    def __rrshift__(self, left):
        return IterChain(left, self)

    __lshift__ = __rrshift__
    __rlshift__ = __rshift__

    def __str__(self):
        return 'IterChain%s' % str(self._iterables)


class iterable(object):
    """
    decorator
    ex)
    @iterable
    def example(n):
        for i in range(n):
            yield i
    e1 = example(10)
    e2 = example(8)
    for i in e1:
        print i
    for i in e2:
        print i
    """

    def __init__(self, gen):
        self._gen = gen

    def __call__(self, *args, **kwargs):
        return iterator(self._gen, *args, **kwargs)


class iterator(object):
    """
    reiterable object
    """

    def __init__(self, gen, *args, **kwargs):
        import threading

        self.gen = gen
        self.it = None
        self.args = args  # ()
        self.kwargs = kwargs  # {}
        self.lock = threading.Lock()

        functools.update_wrapper(self, gen)

        if args or kwargs:
            self.reset()

    def __call__(self, *args, **kwargs):
        """
        decorator call
        """
        self.args = args
        self.kwargs = kwargs
        self.reset()
        return self

    def __iter__(self):
        if self.it is None:
            self.reset()
        return self

    def next(self):
        # http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            if self.it is None:
                raise StopIteration
            try:
                return self.it.next()
            except StopIteration:
                self.it = None
                raise StopIteration

    def reset(self):
        self.it = self.gen(*self.args, **self.kwargs)

    def __str__(self):
        return str(self.gen)

    def __repr__(self):
        return str(self.gen)


def iterate(g, *args, **kwargs):
    it = iterator(g)
    return it(*args, **kwargs)


class iterflow(iterable):

    def __rshift__(self, other):
        return IterChain(self, other)

    def __rrshift__(self, other):
        return IterChain(other, self)

    def __lshift__(self, other):
        return IterChain(other, self)

    def __rlshift__(self, other):
        return IterChain(self, other)

    def get_call_chain(self):
        return [self]


class FlowList(ListArgs):

    def __iter__(self):
        res = self[0]
        for call in self[1:]:
            res = call(res)
        return res

    def __call__(self, *args, **kwargs):
        res = self[0](*args, **kwargs)
        for call in self[1:]:
            res = call(res)
        return res


@tuple_args
def flows(args):
    """
    todo : add some example
    :param args:
    :return:
    """

    def flow_if_not(fun):
        # t = type(fun)
        if isinstance(fun, iterator):
            return fun
        elif isinstance(fun, type) and 'itertools' in str(fun.__class__):
            return fun
        else:
            try:
                return flow(fun)
            except AttributeError:
                # generator object has no attribute '__module__'
                return fun

    return FlowList(map(flow_if_not, args))


def flow(fun):

    @iterflow
    @functools.wraps(fun)
    def _gen(it):
        for data in it:
            yield fun(data)

    return _gen


def iflow(ifun):

    @iterflow
    @functools.wraps(ifun)
    def _gen(it):
        for data in it:
            for d in ifun(data):
                yield d
    return _gen


def applyit(f):

    @flow
    @functools.wraps(f)
    def apply(data):
        return [f(i) for i in data]

    return apply


def loop(times=None):
    """
    None for infinite loop
    :param times:
    :return:
    """
    import itertools
    return iterate(itertools.repeat, True, times)


@iterflow
def forever(it):
    """ forever
    todo : add example
    """
    while True:
        # generator 두번쨰 iteration 무한 루프 방지
        i = iter(it)
        try:
            yield i.next()
        except StopIteration:
            raise StopIteration
        while True:
            try:
                yield i.next()
            except StopIteration:
                break


@iterflow
def irange(*args, **kwargs):
    for i in xrange(*args, **kwargs):
        yield i


def nexts(count, iterable=None):

    @iterflow
    def nextsit(it):
        for _ in loop(count):
            yield it.next()

    return nextsit(iterable) if iterable else nextsit


def cycle(c, iterable=None):

    @iterflow
    def cycleit(it):
        for _ in loop(c):
            i = iter(it)
            for d in i:
                yield d

    return cycleit(iterable) if iterable else cycleit


def ifilter(pred=None, iterable=None):
    pred = pred or bool

    @iterflow
    def _filter(it):
        for d in it:
            if pred(d):
                yield d

    return _filter(iterable) if iterable else _filter


@flow
def zipflow(data):
    return zip(*data)


def ibatch(size, iterable=None, rest=False):
    """
    add example
    :param size:
    :param iterable:
    :param rest:
    :return:
    """

    @iterflow
    def exact_size(it):
        it = iter(it)
        while True:
            yield [it.next() for _ in xrange(size)]

    @iterflow
    def at_most(it):
        it = iter(it)
        while True:
            data = []
            for _ in xrange(size):
                try:
                    data.append(it.next())
                except StopIteration:
                    if data:
                        yield data
                    raise StopIteration
            yield data

    ibatchit = at_most if rest else exact_size

    return ibatchit if iterable is None else ibatchit(iterable)


def batchzip(size, iterable=None, rest=False):
    """
    todo : add example
    :param size:
    :param iterable:
    :param rest:
    :return:
    """
    fn = ibatch(size, rest=rest) >> zipflow

    return fn if iterable is None else fn(iterable)


# todo check nargout == 1 or .. some refactoring
def batchstack(size, iterable=None, rest=False):
    """
    todo : add example
    :param size:
    :param iterable:
    :param rest:
    :return:
    """

    def stack(data):
        import numpy as np
        return map(np.vstack, data)

    fn = batchzip(size, rest=rest) >> flow(stack)

    return fn if iterable is None else fn(iterable)


def shuffle(qsize=1024, iterable=None):
    """
    add example
    :param qsize:
    :param iterable:
    :return:
    """

    @iterflow
    def shuffleit(it):
        from random import randrange
        q = []

        for i, d in enumerate(it):
            q.insert(randrange(0, len(q) + 1), d)
            if i < qsize:
                continue
            yield q.pop(randrange(0, len(q)))

        while q:
            yield q.pop(randrange(0, len(q)))

    return shuffleit if iterable is None else shuffleit(iterable)


def queue_class(QueueClass):

    class _iqueue(QueueClass, iterflow):

        __name__ = 'iqueue'

        def __init__(self, iterable=None, maxsize=1024, processor=None):
            super(_iqueue, self).__init__(maxsize=maxsize)
            self.producer = iterable
            self.proc = processor
            self.th = None
            if iterable:
                self.produce()

        def produce(self):

            # self.th = Thread(target=self._produce, args=(self.producer,))
            # self.th.start()
            self.th = Threaded(self._produce, self.producer)

        def _produce(self, producer):
            if self.proc:
                for d in producer:
                    # self.put(self.proc(d))
                    if self.put(self.proc(d)) is False:
                        break

            else:
                for d in producer:
                    # self.put(d)
                    if self.put(d) is False:
                        break

        def __call__(self, iterable):
            self.producer = iterable
            self.produce()
            return self

        def __iter__(self):
            if self.done:
                self.reset()
            return self

        def next(self):
            while not self.done:
                try:
                    return self.get(block=False, timeout=0.01)
                except QueueClass.Empty:
                    continue
            else:
                raise StopIteration

            # if not self.done:
            #     return self.get(block=True)
            # else:
            #     raise StopIteration

        def reset(self):
            # reset = produce
            print('reset Q')
            self.produce()

        @property
        def done(self):
            if self.th is None:
                return True
            return self.empty() and not self.th.is_alive()

        def __repr__(self):
            return str(self.producer)

    return _iqueue


# multiprocessing queue has sometime unexpectable bloken pipe ... T T
# todo(dade) test with pyspark
from .queues import QueueInterruptable

_IQueueMulti = queue_class(QueueInterruptable)


def iqueue(iterable=None, maxsize=1024, processor=None, qclass=None):
    if qclass is None:
        return _IQueueMulti(iterable, maxsize, processor)
    else:
        qklass = queue_class(qclass)
        return qklass(iterable, maxsize, processor)


