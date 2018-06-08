# -*- coding: utf-8 -*-
from collections import (OrderedDict, MutableMapping, Mapping)


class ODict(MutableMapping):

    def __init__(self, *args, **kwargs):
        self._odict = OrderedDict(*args, **kwargs)

    def __setitem__(self, key, value):
        self._odict[key] = value

    def __getitem__(self, item):
        return self._odict[item]

    def __delitem__(self, key):
        del self.__dict__[key]

    def __setattr__(self, key, value):
        if key == '_odict':
            super(ODict, self).__setattr__(key, value)
        else:
            self._odict[key] = value

    def __getattr__(self, item):
        return self._odict.get(item, None)

    def __iter__(self):
        return iter(self._odict)

    def __len__(self):
        return len(self._odict)

    def __str__(self):
        return str(self._odict).replace('OrderedDict', 'odict', count=1)

    def __repr__(self):
        return self._odict.__repr__().replace('OrderedDict', 'odict', count=1)

    def pop(self, key, **kw):
        return self._odict.pop(key, **kw)

    def popitem(self):
        return self._odict.popitem()

    def clear(self):
        self._odict.clear()

    def update(self, *args, **kwds):
        self._odict.update(*args, **kwds)

    def setdefault(self, key, default=None):
        return self._odict.setdefault(key, default=default)

    def __sub__(self, other):
        return ODict((k, v) for k, v in self.items() if k not in other)

    def __add__(self, other):
        if not other:
            return self.copy()
        added = self.copy()
        added.update(other)
        return added

    __radd__ = __add__

    def __getnewargs__(self):
        """pickling related"""
        return tuple()

    def __getstate__(self):
        return self._odict

    def __setstate__(self, d):
        self._odict.update(d)

    def copy(self):
        return ODict((k, v) for k, v in self.items())

    def intersect(self, other):
        """
        self와 other 키가 동일한 아이템의 dictobj
        :type other: dict
        :rtype: dictobj:
        """
        return ODict((k, self[k]) for k in self if k in other)

    @staticmethod
    def from_dict(dic):
        """
        recursive dict to dictobj 컨버트
        :param dic:
        :return:
        """
        return ODict((k, ODict.convert_ifdic(v)) for k, v in dic.items())

    @staticmethod
    def convert_ifdic(value):
        if isinstance(value, Mapping):
            return ODict.from_dict(value)
        else:
            return value


# alias
odict = ODict

