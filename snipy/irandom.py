# -*- coding: utf-8 -*-
from contextlib import contextmanager
import numpy as np


@contextmanager
def np_seed(seed):
    """
    numpy random seed context
    :param seed:
    :return:
    """
    if seed is not None:
        state = np.random.get_state()
        np.random.seed(seed)
        yield
        np.random.set_state(state)

    else:
        yield


def rand_lazy_size(fun, **kw):

    def wrapper(*args, **kwargs):
        kwargs.update(kw)
        dtype = kwargs.pop('dtype', 'float32')

        def wrapped(shape):
            return fun(*args, size=shape).astype(dtype)
        return wrapped

    return wrapper

# most common only implemented
uniform = rand_lazy_size(np.random.uniform)
normal = rand_lazy_size(np.random.normal)
randint = rand_lazy_size(np.random.randint, dtype='int32')

