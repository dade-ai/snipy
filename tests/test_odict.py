# -*- coding: utf-8 -*-
import snipy.common as sp


def test_odict():
    data = [('a', 0), ('b', 1)]
    d = sp.odict(data)

    assert d['a'] == d.a == data[0][1]
    assert d['b'] == d.b == data[1][1]

    d.c = 2121
    assert d['c'] == d.c == 2121
    print(list(d.keys()))
    assert list(d.keys()) == 'a b c'.split()


if __name__ == '__main__':
    test_odict()

