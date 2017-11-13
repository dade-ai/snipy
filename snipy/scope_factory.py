# -*- coding: utf-8 -*-
from contextlib import contextmanager
# from snipy.odict import odict

_stack = [None]


class Scope(object):
    def __init__(self, name):
        self.name = name


@contextmanager
def vscope(name):

    # get current scope or new scope
    # append hierachical information
    sc = Scope(name)
    _stack.append(sc)

    try:
        yield sc
    finally:
        _stack.pop(-1)


def get_scope():
    return _stack[-1]

