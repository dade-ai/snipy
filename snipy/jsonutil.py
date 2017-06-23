# -*- coding: utf-8 -*-
import cjson
import numpy as np


def numpy_to_json(x):
    return cjson.encode(x.tolist())


def tojson(o):
    """
    recursive implementation
    """
    try:
        return cjson.encode(o)
    except cjson.EncodeError:
        pass
    try:
        return o.tojson()
    except AttributeError as e:
        pass

    t = type(o)
    if isinstance(o, list):
        return '[%s]' % ', '.join([tojson(e) for e in o])
    elif isinstance(o, dict):
        d = ['%s:%s' % (k, tojson(v)) for k, v in o.iteritems()]
        return '{%s}' % ', '.join(d)
    elif isinstance(o, set):
        d = ['%s:%s' % (tojson(e)) for e in o]
        return '{%s}' % ', '.join(d)
    elif isinstance(o, np.ndarray):
        return numpy_to_json(o)
    else:
        raise ValueError('error, failed encoding type(%s) to json' % t)

