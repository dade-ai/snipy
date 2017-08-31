# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from collections import (Mapping, OrderedDict)


class OrderedDictObj(OrderedDict):
    """
    object attribute 혹은 dictionary처럼 접근하는 클래스
    """

    def __init__(self, *args, **kwargs):
        super(OrderedDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattr__(self, item):
        """
        해당 키가 없을 때 None을 준다.
        :param item: 키
        :rtype None:
        """
        return None

    def __sub__(self, other):
        """
        :type other: dict
        :rtype: dictobj: other dict에 없는 items만 리턴
        """
        return OrderedDict({k: v for k, v in self.items() if k not in other})

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
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def copy(self):
        return OrderedDict(super(OrderedDict, self).copy())

    def intersect(self, other):
        """
        self와 other 키가 동일한 아이템의 dictobj
        :type other: dict
        :rtype: dictobj:
        """
        return OrderedDict({k: self[k] for k in self if k in other})

    @staticmethod
    def convert_ifdic(value):
        if isinstance(value, Mapping):
            return OrderedDict.from_dict(value)
        else:
            return value

    @staticmethod
    def from_dict(dic):
        """
        recursive dict to dictobj 컨버트
        :param dic:
        :return:
        """
        return OrderedDict({k: OrderedDict.convert_ifdic(v) for k, v in dic.items()})


# alias
odict = OrderedDictObj
