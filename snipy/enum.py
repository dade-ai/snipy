# -*- coding: utf-8 -*-


class EnumMeta(type):

    def __iter__(cls):
        return (k for k in cls.__dict__ if not k.startswith('__'))

    def iteritems(cls):
        return ((k, v) for k, v in cls.__dict__.items() if not k.startswith('__'))

    def values(cls):
        return [v for k, v in cls.iteritems()]

    def __contains__(cls, item):
        return (item in iter(cls)) or (item in cls.values())

    def __getitem__(self, item):
        return getattr(self, item)


class Enum(object):
    __metaclass__ = EnumMeta


def enum(name, *members, **withvalue):
    """class buider"""
    if len(members) == 1:
        if isinstance(members[0], str):
            members = members[0].split()
        elif isinstance(members[0], (list, tuple)):
            members = members[0]

    dic = {v: v for v in members}
    dic.update(withvalue)

    return type(name, (Enum,), dic)


if __name__ == '__main__':
    # Fruits = enum('Fruits', ['apple', 'banana'])
    class Fruits(Enum):
        banana = 1
        apple = 2

    for f in Fruits:
        print(f)
    print(1 in Fruits)

