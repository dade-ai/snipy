# -*- coding: utf-8 -*-
try:
    import cjson as json
except Exception:
    import ujson as json


import numpy as np


def numpy_to_json(x):
    return json.encode(x.tolist())


def tojson(o):
    """
    recursive implementation
    """
    try:
        return json.encode(o)
    except json.EncodeError:
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

