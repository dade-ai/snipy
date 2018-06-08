# -*- coding: utf-8 -*-
import time
import sys
import math
from .dictobj import dictobj
from .term import write, clear, TermCursor


class ProgressBase(object):

    defaultopt = dictobj(show=True, update_per=1, show_item=False, width=10)

    def __init__(self, iter=None, **kwargs):
        self.iter = iter
        self.opt = self.defaultopt.copy()
        self.update_opt(iter, kwargs)
        write('\n')

    def update_opt(self, iter, args=None):
        self.iter = iter or self.iter
        if args:
            args = {k: v for k, v in args.items() if v is not None}
            self.opt.update(args)
        self.opt.start = time.time()
        self.opt.seen_so_far = 0
        self.setup()

    @property
    def target(self):
        return self.opt.target

    @target.setter
    def target(self, value):
        self.opt.target = value
        self.setup()

    def setup(self):
        pass

    def __call__(self, iter=None, **kwargs):
        self.update_opt(iter, kwargs)
        return self

    def __iter__(self):
        if not self.opt.show:
            return iter(self.iter)
        else:
            return self.iter_verbose(enum=False)

    def enumerate(self, iter=None, **kwargs):
        self.update_opt(iter, kwargs)
        if not self.opt.show:
            return enumerate(self.iter)
        else:
            return self.iter_verbose(enum=True)

    def iter_verbose(self, enum=False):

        def enum_yied(ind, data):
            return ind, data

        def plain_yield(_, data):
            return data

        fun = enum_yied if enum else plain_yield
        update_per = self.opt.update_per
        show_item = self.opt.show_item
        for i, item in enumerate(self.iter):
            if i % update_per == 0:
                if show_item:
                    self.update(i, item)
                else:
                    self.update(i)
            yield fun(i, item)
        self.stop()

    def stop(self, message=None):
        self.write(message or '[Done]')

    @classmethod
    def write(cls, message):
        if message:
            sys.stdout.write(str(message))
        sys.stdout.flush()

    def format_current(self, current):
        bar = self.format_bar(current)
        bar += self.format_eta(1.0)
        return bar

    def update(self, current, message=None):
        bar = self.format_current(current)
        write(clear.line + TermCursor.home)
        # write(TermCursor.home)
        if message:
            self.write(bar + str(message))
        else:
            self.write(bar)

    def format_bar(self, current):
        bar = '[%d]' % (current,)
        width = self.opt.width
        bar += self.format_progress(current % width, width, width)
        return bar

    def format_eta(self, ratio):
        from .ansi import ansi
        if ratio == 0:
            return ansi.green('-[ETA:unknown]')
        elif ratio < 1.:
            return ansi.green('-[ETA:{0:.2f}h]'.format(self.get_eta(ratio)))
        else:
            return ansi.green('-[ETA:{0:.2f}h]'.format((time.time() - self.opt.start)/3600.))

    def get_eta(self, ratio):
        elapsed = time.time() - self.opt.start
        eta = 0 if ratio == 0 else elapsed / ratio * (1 - ratio)

        return eta/3600.

    @classmethod
    def format_progress(cls, current, target, width):
        from .ansi import ansi
        prog = int(float(current) / target * width)
        arrow = '>' if current < target else '='
        bar = '=' * prog + arrow + '.' * (width - prog)

        return ansi.yellow('[' + bar + ']')


# noinspection PyAttributeOutsideInit
class Progress(ProgressBase):

    defaultopt = dictobj(target=None, width=10, show=True, step=None, update_per=None,
                         show_item=False)

    def __init__(self, iter=None, **kwargs):
        super(Progress, self).__init__(iter, **kwargs)

    def setup(self):
        super(Progress, self).setup()

        opt = self.opt

        if opt.target is None:
            target = len(self.iter) if self.iter else opt.target
            target = target or opt.width
        else:
            target = opt.target

        step = opt.step or target
        self.opt.target = target
        opt.update_per = opt.update_per or target / float(step)
        opt.bar_format = self.bar_format(target)
        # opt.seen_so_far = 0

    def stop(self, message=None):
        self.update(self.target, message)

    def format_current(self, current):
        bar = self.format_bar(current)
        bar += self.format_eta(float(current)/self.target)
        return bar

    @classmethod
    def bar_format(cls, target):
        if target is None:
            return ' %s/%s'
        numdigits = int(math.floor(math.log10(target))) + 1
        barstr = '  %{0}d/%{0}d '.format(numdigits)
        return barstr

    def format_bar(self, current):
        opt = self.opt
        bar = opt.bar_format % (current, self.target)
        bar += self.format_progress(current, self.target, opt.width)
        return bar


def progress(iter, **kwargs):
    """
    프로그래스 bar
    for i in progress(10):
        print i

    for i in progress(iter):
        print i
    """
    if isinstance(iter, int):
        iter = xrange(iter)
    if hasattr(iter, '__len__') or 'target' in kwargs:
        cls = Progress
    else:
        cls = ProgressBase

    return cls(iter, **kwargs)


def iprogress(iter, **kwargs):
    """
    enumerate with progress
    """
    return progress(iter, **kwargs).enumerate()

