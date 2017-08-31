# -*- coding: utf-8 -*-
import collections


def named(typename, *fieldnames, **defaults):
    """
    namedtuple with default values
    named('typename', fields | *fields, default=x, [**defaults])
    :param typename:
    :param fieldnames:
    :param defaults:
    :return:
    """
    if len(fieldnames) == 1:
        if isinstance(fieldnames[0], str):
            fieldnames = tuple(fieldnames[0].replace(',', ' ').split())
        elif isinstance(fieldnames[0], (list, tuple)):
            fieldnames = fieldnames[0]

    # set default of defaults
    default_of_defaults = defaults.pop('default', None)

    dfields = tuple(f for f in defaults if f not in fieldnames)

    T = collections.namedtuple(typename, fieldnames + dfields)
    T.__new__.__defaults__ = (default_of_defaults,) * len(T._fields)
    prototype = T(**defaults)
    T.__new__.__defaults__ = tuple(prototype)

    # make picklable
    globals()[typename] = T

    return T


class NamedMeta(type):

    def __new__(mcs, name, bases, dct):
        """ contains default values,
        dct contains __slots__ for being ordered
        bases is ignored
        """

        slots = dct.get('__slots__', ())
        defaults = dct.get('__defaults__', {})

        # defaults in base class arguments ^^
        # otherwise ignoring
        for base in bases:
            if isinstance(base, collections.Mapping):
                defaults.update(base)
            else:
                continue

        # if need inheritance?,
        # IDE cannot resolve attributes.
        # below is experiment for inheritance.
        #
        # for base in bases:
        #     # get slots of base
        #     slot = base._fields
        #     default = base.__new__.__defaults__
        #     defaults.update({s: v for s, v in zip(slot, default)})
        #     slots += slot

        return named(name, *slots, **defaults)


# region sample for namedmeta class

# class SampleNamed:
#     __metaclass__ = NamedMeta
#     __slots__ = ('prop1', 'prop2')
#     __defaults__ = dict(prop1=1, prop2=2, default=1)
#
#
# def test_named_default():
#     a = SampleNamed()
#     print a.prop1
#     print a.prop2
#     print a
#
#     import joblib
#     joblib.dump(a, '/tmp/named_default.pkl')
#     a2 = joblib.load('/tmp/named_default.pkl')
#     print a2
#     print isinstance(a, (tuple))

# endregion


# region experimental

# class NamedTestMeta(type):
#
#     def __new__(mcs, name, bases, dct):
#         """bases contains default values,
#         dct contains __slots__ for being ordered
#         """
#         # assert len(bases) == 1 and isinstance(bases[0], collections.Mapping)
#         slots = dct.get('__slots__', ())
#         defaults = {}
#         for base in bases:
#             if isinstance(base, collections.Mapping):
#                 defaults.update(base)
#             elif isinstance(base, (tuple, list)):
#                 slots += tuple(base)
#             elif isinstance(base, type):
#                 continue
#             else:
#                 slots += (base,)
#
#         return named(name, *slots, **defaults)
#
#
# class SampleTuplet('a b c', dict(a=2, b=3)):
#     __metaclass__ = NamedTestMeta
#     # __slots__ = ('a', 'b', 'c')

# endregion


