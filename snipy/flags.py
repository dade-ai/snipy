# -*- coding: utf-8 -*-
from collections import (MutableMapping)

import argparse as _argparse
import snipy.ilogging as logg

# modified from tensorflow.python.platform.flags


def _str2bool(v):
    return v.lower() in ('yes', 'true', 't', '1')


def _str2eval(v):
    # list or tuple or str ...
    return eval(v)


# add other type parsing methods
_parser = _argparse.ArgumentParser()
_parser.register('type', 'bool', _str2bool)
_parser.register('type', 'list', _str2eval)
# _parser.register('type', 'tuple', _str2eval)


# region flag class

class _FlagValuesNone(MutableMapping):
    """Global container and accessor for flags and their values."""

    def __init__(self):
        self.__dict__['__flags'] = {}
        self.__dict__['__parsed'] = False

# region dictionaly interface

    def __getitem__(self, item):
        return self.__getattr__(item)

    def __setitem__(self, key, value):
        self.__dict__['__flags'][key] = value

    def __delitem__(self, key):
        del self.__dict__['__flags'][key]

    def __iter__(self):
        return iter(self.__dict__['__flags'])

    def __len__(self):
        return len(self.__dict__['__flags'])

    def __str__(self):
        return str(self.__dict__['__flags'])

    def __repr__(self):
        return self.__dict__['__flags'].__repr__()
# endregion

    def _parse_flags(self, args=None, verbose=True):
        silent = not verbose
        result, unparsed = _parser.parse_known_args(args=args)
        for flag_name, val in vars(result).items():
            self.__dict__['__flags'][flag_name] = val
            silent or logg.info('--{}={}'.format(flag_name, val))

        self.__dict__['__parsed'] = True
        # if unparsed:
        #     logg.info('unparsed : {}'.format(unparsed))
        return unparsed

    def _parse_flags_kw(self, args=None, verbose=True):
        left = self._parse_flags(args=args, verbose=verbose)
        unparsed = []
        kw = dict()
        while left:
            p = left.pop(0)
            if p.startswith('--'):
                s = p[2:].split('=')
                if len(s) == 2:
                    kw[s[0]] = s[1]
                else:
                    kw[s[0]] = left.pop(0)
            else:
                unparsed.append(p)

        # flag_not_parsed = tuple(f for f in unparsed if f.startswith('--'))
        if verbose and (unparsed or kw):
            if unparsed:
                logg.warn('as args: {}'.format(unparsed))
            if kw:
                logg.warn('as kwargs: {}'.format(kw))
        return unparsed, kw

    def __getattr__(self, name):
        """Retrieves the 'value' attribute of the flag --name."""
        if not self.__dict__['__parsed']:
            self._parse_flags()
        if name not in self.__dict__['__flags']:
            return None
        return self.__dict__['__flags'][name]

    def __setattr__(self, name, value):
        """Sets the 'value' attribute of the flag --name."""
        if not self.__dict__['__parsed']:
            self._parse_flags()
        self.__dict__['__flags'][name] = value

    def add_flag(self, flagname, default_value, help='', dtype=None, **kwargs):
        def gettype_default(v):
            if v is None:
                return str, v
            elif isinstance(v, bool):
                return 'bool', v
            elif type(v) is type:
                # 디폴트 대신 type이 들어옴. 디폴트 없는 것으로
                return v, None
            # elif isinstance(v, (tuple, list)):
            #     return
            else:
                return type(v), v

        h = help or flagname.replace('_', ' ')
        if dtype is None:
            dtype, default_value = gettype_default(default_value)

        h += ' : default = {}'.format(default_value)

        _parser.add_argument('--' + flagname, type=dtype,
                             default=default_value, help=h, **kwargs)

        self.__dict__['__parsed'] = False

    def add_args(self, args):
        # add positional arguments
        for i, a in enumerate(args):
            _parser.add_argument(a, nargs=i+1)

flag = _FlagValuesNone()

# endregion


# def remove_flag(name):
#     """remove flag from parser"""
#     for action in _parser._actions:
#         if vars(action)['option_strings'][0] == name:
#             _parser._handle_conflict_resolve(None, [(name, action)])
#             break


def add_flag(*args, **kwargs):
    """
    define a single flag.
    add_flag(flagname, default_value, help='', **kwargs)
    add_flag([(flagname, default_value, help), ...])
    or
    define flags without help message
    add_flag(flagname, default_value, help='', **kwargs)

    add_flag('gpu', 1, help='CUDA_VISIBLE_DEVICES')
    :param args:
    :param kwargs:
    :return:
    """
    if len(args) == 1 and isinstance(args[0], (list, tuple)):
        for a in args[0]:
            flag.add_flag(*a)
    elif args:
        flag.add_flag(*args, **kwargs)
    else:
        for f, v in kwargs.items():
            flag.add_flag(f, v)


def parse_flag(*args, **kwargs):
    return flag._parse_flags(*args, **kwargs)



# region run

def run(main=None, argv=None, **flags):
    """
    :param main: main or sys.modules['__main__'].main
    :param argv: argument list used in argument parse
    :param flags: flags to define with defaults
    :return:
    """
    """Runs the program with an optional 'main' function and 'argv' list."""
    import sys as _sys
    import inspect
    main = main or _sys.modules['__main__'].main

    if main.__doc__:
        docstring = main.__doc__.split(':param')[0]
        _parser.usage = 'from docstring \n {}'.format(docstring)  # add_help

    # if not flags:
    try:
        a = inspect.getfullargspec(main)
    except AttributeError:
        a = inspect.getargspec(main)  # namedtuple(args, varargs, keywords, defaults)
    if a.defaults:
        kwargs = dict(zip(reversed(a.args), reversed(a.defaults)))
        add_flag(**kwargs)
    else:
        kwargs = dict()

    # add to command argument
    if a.defaults is None:
        nargs = len(a.args)
    else:
        nargs = len(a.args) - len(a.defaults)
    # if nargs > 0:
    posargs = a.args[:nargs]
    flag.add_args(posargs)
    add_flag(**flags)

    # Extract the args from the optional `argv` list.
    args = argv[1:] if argv else None

    # Parse the known flags from that list, or from the command
    # line otherwise.
    unparsed, kw = flag._parse_flags_kw(args=args)

    d = flag.__dict__['__flags']
    args = [d[k] for k in posargs]
    args += unparsed

    kwargs.update({k: d[k] for k in kwargs.keys()})
    kwargs.update(kw)

    # Call the main function, passing through any arguments,
    # with parsed flags as kwwargs
    # to the final program.
    _sys.exit(main(*args, **kwargs))

# endregion
