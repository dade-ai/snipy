# -*- coding: utf-8 -*-
from functools import wraps
from contextlib import contextmanager
import signal


def functional(ifunctional):
    """
    fun(fn) -> function or
    fun(fn, args...) -> call of fn(args...)
    :param ifunctional: f
    :return: decorated function
    """

    @wraps(ifunctional)
    def wrapper(fn, *args, **kw):

        fn = ifunctional(fn)
        if args or kw:
            return fn(*args, **kw)
        else:
            return fn

    return wrapper


def tuple_pack(*args):
    return args


@functional
def tupleout(fn):
    """

    ex) code ::

        def asis(a):
        return a

        f = tupleout(asis)
        print f(1)  # (1,)
        print tupleout(asis, 1)  # (1,)

    :param Function fn:
    :return:
    """
    @wraps(fn)
    def wrapped(*args, **kwargs):
        return fn(*args, **kwargs),

    return wrapped


def tuplefy(x):
    """
    make tuple if not tuple or list
    :param x:
    :return:
    """
    if x is None or isinstance(x, (tuple, list)):
        return x
    else:
        return x,


def tuple_arg(fn):
    """
    fun(1,2) -> fun((1,), (2,))로
    f(1,2,3) =>  f((1,), (2,), (3,))
    :param fn:
    :return:
    """
    @wraps(fn)
    def wrapped(*args, **kwargs):
        args = map(tuplefy, args)
        return fn(*args, **kwargs)

    return wrapped


def tuple_args(fn):
    """
    args 파싱 유틸 function
    fun(p1, p2, ...pn, **kwargs) or fun([p1, p2, ..], **kwargs)
    ex) 샘플::

        @tuple_arg
        def f(args, **kwargs):
            for d in args:
                print d
        f(1,2,3) =>  f([1,2,3])
    :param function fn:
    :return:
    """

    @wraps(fn)
    def wrapped(*args, **kwargs):

        if len(args) == 1:
            if isinstance(args[0], tuple):
                return fn(args[0], **kwargs)
            elif isinstance(args[0], list):
                return fn(tuple(args[0]), **kwargs)
        return fn(args, **kwargs)

    return wrapped


def unpack_args(classfun, nth=0):
    """
    args 갯수가 nth + 1 개 일때, 그게 만약 tuple이면, unpack
    :param classfun:
    :param nth: nth = 0, 일반 함수, nth = 1: 클래스 함수 1이 self니깐.
    :return:
    """
    if classfun:
        nth = 1

    def deco(fn):
        def wrapped(*args, **kwargs):
            if len(args) == nth + 1 and isinstance(args[nth], (tuple, list)):
                args = tuple(args[:nth] + args[nth])
                return fn(*args, **kwargs)
            else:
                return fn(*args, **kwargs)
        return wrapped

    return deco


def optional_str(deco):
    """
    string 1개만 deco 인자로 오거나 없거나.
    :param deco:
    :return:
    """

    @wraps(deco)
    def dispatcher(*args, **kwargs):

        # when only function arg
        if not kwargs and len(args) == 1 and not isinstance(args[0], str) \
                and args[0] is not None:
            decorator = deco()
            return decorator(args[0])
        else:
            # decorator with args
            decorator = deco(*args, **kwargs)
            return decorator

    return dispatcher


def patchmethod(*cls, **kwargs):
    """
    클래스 멤버 패치 @patchmethod(Cls1, ..., [name='membername'])
    ex)
    class A(object):
        def __init__(self, data):
            self.data = data

    @patchmethod(A)
    def sample(self):
        ''' haha docstrings '''
        print self.data

    @patchmethod(A, name='membermethod)
    def sample(self):
        ''' haha docstrings '''
        print self.data

    a = A()
    a.sample()

    """

    def _patch(fun):
        m = kwargs.pop('name', None) or fun.__name__
        for c in cls:
            setattr(c, m, fun)
            # c.__dict__[m].__doc__ = fun.__doc__

    def wrap(fun):
        _patch(fun)
        return fun

    return wrap


def patchproperty(*cls, **kwargs):
    """
    class getter 함수 패치 decorator
    EX)
    class B(A):
        pass

    @patchproperty(B)
    def prop(self):
        return 'hello'

    :param cls:
    :param kwargs:
    """

    def _patch(fun):
        m = kwargs.pop('property', None) or fun.__name__
        p = property(fun)
        for c in cls:
            setattr(c, m, p)

    def wrap(fun):
        _patch(fun)
        return fun

    return wrap


class patch(object):

    @staticmethod
    def method(clz, fun, name=None):
        name = name or fun.__name__
        for c in clz:
            setattr(c, name, fun)

    @staticmethod
    def methods(clz, funs, names=None):
        if names is None:
            for f in funs:
                name = f.__name__
                for c in clz:
                    setattr(c, name, f)
        else:
            for f, name in zip(funs, names):
                for c in clz:
                    setattr(c, name, f)

    @staticmethod
    def getter(clz, fun, name=None):
        name = name or fun.__name__
        p = property(fun)
        for c in clz:
            setattr(c, name, p)

    @staticmethod
    def getters(clz, funs):
        for f in funs:
            p = property(f)
            name = f.__name__
            for c in clz:
                setattr(c, name, p)


@contextmanager
def on_interrupt(handler, reraise=False):
    """
    context for handling keyboardinterrupt
    ex)
    with on_interrupt(handler):
        critical_work_to_prevent()

    from logger import logg
    on_interrupt.signal = None

    :param function handler:
    :param bool reraise:
    :return: context
    """

    def _handler(sig, frame):
        handler.signal = (sig, frame)
        handler._reraise = handler()

    handler._reraise = False
    handler.signal = None
    oldhandler = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, _handler)

    yield handler

    signal.signal(signal.SIGINT, oldhandler)
    if (reraise or handler._reraise) and handler.signal:
        oldhandler(*handler.signal)


def interrupt_guard(msg='', reraise=True):
    """
    context for guard keyboardinterrupt
    ex)
    with interrupt_guard('need long time'):
        critical_work_to_prevent()

    :param str msg: message to print when interrupted
    :param reraise: re-raise or not when exit
    :return: context
    """
    def echo():
        print(msg)

    return on_interrupt(echo, reraise=reraise)


