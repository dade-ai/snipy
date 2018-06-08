# -*- coding: utf-8 -*-
import functools
import inspect


def deco_optional(decorator):
    """
    optional argument 를 포함하는 decorator를 만드는 decorator
    """

    @functools.wraps(decorator)
    def dispatcher(*args, **kwargs):
        one_arg = len(args) == 1 and not kwargs
        if one_arg and inspect.isfunction(args[0]):
            decor_obj = decorator()
            return decor_obj(args[0])
        else:
            return decorator(*args, **kwargs)

    return dispatcher


def optional(deco):
    """
    decorator option은 kwargs만 허용
    :param deco:
    :return:
    """
    @functools.wraps(deco)
    def dispatcher(*args, **kwargs):
        decorator = deco(**kwargs)
        if args:
            assert len(args) == 1
            return decorator(args[0])
        else:
            return decorator
    return dispatcher


def optional_str(deco):
    """
    string 1개만 deco 인자로 올 수 있다.
    :param deco:
    :return:
    """

    @functools.wraps(deco)
    def dispatcher(*args, **kwargs):

        # when only function arg
        if not kwargs and len(args) == 1 \
                and not isinstance(args[0], str) \
                and args[0] is not None:
            decorator = deco()
            return decorator(args[0])
        else:
            # decorator with args
            decorator = deco(*args, **kwargs)
            return decorator

    return dispatcher


# noinspection PyProtectedMember
class bind(object):
    """
    # def bind(fun, *argsbind, **kwbind):
    #     if argsbind:
    #         return bindargs(fun, *argsbind, **kwbind)
    #     else:
    #         return bindkw(fun, **kwbind)
    #
    # # for bind function
    # bind._ = bind.placeholder = object()
    """
    _ = placeholder = object()

    def __new__(cls, fun, *argsbind, **kwbind):
        if argsbind:
            return bindargs(fun, *argsbind, **kwbind)
        else:
            return bindkw(fun, **kwbind)


def bindargs(fun, *argsbind, **kwbind):
    """
    _ = bind.placeholder   # unbound placeholder (arg)
    f = bind(fun, _, _, arg3, kw=kw1, kw2=kw2), f(arg1, arg2)
    :param fun:
    :param argsbind:
    :param kwbind:
    :return:
    """

    assert argsbind
    argsb = list(argsbind)
    iargs = [i for i in range(len(argsbind)) if argsbind[i] is bind.placeholder]
    # iargs = [a is bind.placeholder for a in argsbind]

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        kws = kwbind.copy()

        args_this = [a for a in argsb]
        for i, arg in zip(iargs, args):
            args_this[i] = arg
        args_this.extend(args[len(iargs):])

        # kwargs.update(kwbind)
        kws.update(kwargs)
        # return fun(*argsb, **kws)
        return fun(*args_this, **kws)

    return wrapped


def bindkw(fun, **kwbind):
    """
    kwarg 바인딩된 함수 return.
    ex)
    def fun(opt1, opt2):
        print opt1, opt2

    f = bind(fun, opt1=2, opt2=3)
    f()
    :param function fun:
    :param kwbind:
    :return: function
    """

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        kws = kwbind.copy()
        kws.update(kwargs)
        return fun(*args, **kws)

    return wrapped


class default_arg(object):
    """
    function decorator 디폴트 param control
    @default_arg(opt1=1, opt2=8)
    def example2(a, opt2, opt1):
        '''
        example2 function for test default decorator
        '''
        print a, opt1, opt2

    example2(1)  # print 1, 1, 8
    """

    def __init__(self, **kwargs):
        self.default = kwargs

    def __call__(self, fun):
        @functools.wraps(fun)
        def wrapped(*args, **kwargs):
            # default kwargs dict
            # wrapped.default = self.kwargs.copy()
            merge = wrapped.default.copy()
            merge.update(kwargs)

            return fun(*args, **merge)

        wrapped.default = self.default

        return wrapped


def default(fun, **kwdefault):
    """
    change default value for function
    ex)
    def sample(a, b=1, c=1):
        print 'from sample:', a, b, c
        return a, b, c
    fun = default(sample, b=4,c=5)
    print fun.default  # get default value dictionary
    fun(1)  # print 1, 5, 5 and return

    :param fun:
    :param kwdefault:
    :return:
    """

    @functools.wraps(fun)
    def wrapped(*args, **kwargs):
        merge = wrapped.default.copy()
        merge.update(kwargs)
        return fun(*args, **merge)

    wrapped.default = kwdefault

    return wrapped


def setup_once(initfn):
    """
    call class instance method for initial setup ::

        class B(object):

            def init(self, a):
                print 'init call:', a

            @setup_once(init)
            def mycall(self, a):
                print 'real call:', a

        b = B()
        b.mycall(222)
        b.mycall(333)

    :param function initfn:
    :return: decorated method
    """
    def wrap(method):

        finit = initfn.__name__
        fnname = method.__name__

        @functools.wraps(method)
        def wrapped(self, *args, **kwargs):

            @functools.wraps(method)
            def aftersetup(*a, **kw):
                return method(self, *a, **kw)

            setupfn = getattr(self, finit)
            setupfn(*args, **kwargs)

            res = method(self, *args, **kwargs)
            setattr(self, fnname, aftersetup)
            return res
        return wrapped

    return wrap


def call_once(method):

    fnname = method.__name__

    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):

        @functools.wraps(method)
        def dummy(*_, **_kw):
            pass

        res = method(self, *args, **kwargs)

        setattr(self, fnname, dummy)
        return res
    return wrapped


def static(**kwargs):
    """ USE carefully ^^ """
    def wrap(fn):
        fn.func_globals['static'] = fn
        fn.__dict__.update(kwargs)
        return fn
    return wrap


@optional
def deprecated(message=''):
    import warnings

    def wrap(fun):
        @functools.wraps(fun)
        def wrapped(*args, **kwargs):
            warnings.warn("Call to deprecated function {}.".format(fun.__name__) + message,
                          DeprecationWarning)
            return fun(*args, **kwargs)
        return wrapped
    return wrap


