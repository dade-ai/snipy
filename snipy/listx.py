# -*- coding: utf-8 -*-


#region List class blocks


class ListObj(list):
    pass


class ListArgs(list):
    """
    list extention, can initialize by *args
    """
    __slots__ = ()

    def __init__(self, *args):
        try:
            # args가 1개이하이고, iterable 이라면
            super(ListArgs, self).__init__(*args)
        except TypeError:
            # tuple(args)들로 초기화
            super(ListArgs, self).__init__(args)


# noinspection PyCallingNonCallable
class ListCallable(ListArgs):
    """
    list content as callable-chain
    """
    __slots__ = ()

    def __call__(self, *args, **kwargs):
        res = self[0](*args, **kwargs)
        for call in self[1:]:
            res = call(res)
        return res


class ListAddable(ListArgs):
    """
    list addable (add(+) = extend or append)
    """
    __slots__ = ()

    def add(self, list_or_not):
        # self.add(other) = self[1,2,3] + other[4,5,6] => [1,2,3, 4,5,6]
        try:
            self.extend(list_or_not)
        except TypeError:
            self.append(list_or_not)
        return self

    def radd(self, other):
        # self.radd(other) = other[4,5,6] + self[1,2,3]  => [4,5,6, 1,2,3]
        try:
            self[:0] = other
        except TypeError:
            self.insert(0, other)
        return self

    def __add__(self, other):
        return self.__class__(self).add(other)

    def __radd__(self, other):
        return self.__class__(self).radd(other)

    def __iadd__(self, other):
        return self.add(other)


class ListCallChain(ListAddable, ListCallable):
    """
    list.add(extend or append) by >> operator
    """

    # self[1,2,3] >> other[4,5,6] => [1,2,3, 4,5,6]
    # other >> self => self(other)
    # self[1,2,3] << other[4,5,6] => [4,5,6, 1,2,3]
    # other << self => self.add(other)
    __slots__ = ()

    def __rshift__(self, other):
        # self[1,2,3] >> other[4,5,6] => [1,2,3, 4,5,6]
        # self >> other : lazy calling sequence add
        return self.__add__(other)

    def __rrshift__(self, other):
        # other >> self => self(other)
        return self.__call__(other)

    # << support
    def __lshift__(self, other):
        # self << other => as call : self(other)
        if hasattr(other, '__rlshift__'):
            return other.__rlshift__(self)
        else:
            return self.__call__(other)

    # << support
    def __rlshift__(self, other):
        # other << self => append other [self, other]
        return self.__add__(other)


class ICall(object):

    __slots__ = ()

    def __rrshift__(self, other):
        # other >> self => self(other)
        return self.__call__(other)

    # << support
    def __lshift__(self, other):
        # self << other => as call : self(other)
        if hasattr(other, '__rlshift__'):
            return other.__rlshift__(self)
        else:
            return self.__call__(other)


class IChain(object):

    __slots__ = ()

    def __rshift__(self, other):
        # self >> other
        # self.add(other) 보다 나은점은.
        # ichain >> ichainlist 내가 리스트일 필요는 없지만,
        # chain list class type을 모른다. other 쪽에서 정하게 하자.
        # other 가 _radd 구현해야할 책임이
        # checkme if same type??

        return other.__radd__(self)

    # << support not in this, other decide what to do
    def __rlshift__(self, other):
        # other << self
        return other.__radd__(self)


class ICallChain(ICall, IChain):

    __slots__ = ()

    def __radd__(self, other):
        """ overidden in LayerBasic
        other >> self
        """
        return ListCallChain(other, self)


# aliases
listobj = ListObj
listargs = ListArgs
listcallable = ListCallable
listaddable = ListAddable
listchain = ListCallChain


# def ListX(bases, clsname=''):
#     """list class factory"""
#     clsname = clsname or '_'.join(b.__name__ for b in bases)
#     return type(clsname, bases, {})

#endregion


if __name__ == '__main__':
    f = [lambda x:x+1, lambda x:x+2, lambda y: y+3]
    l = listchain(f)
    ll = l >> (lambda x: x + 4)
    print(0 >> l)
    print(l(0))
    print(ll(0))


