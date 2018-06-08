# -*- coding: utf-8 -*-
"""
package util, inspection related
"""
import os
import sys
from os.path import dirname

# fixme
from .ilogging import logg


def guess_package_path(searchfrom):
    """
    package path. return None if failed to guess
    """
    from snipy.io import fileutil

    current = searchfrom + '/'
    init_found = False
    pack_found = False
    while not init_found and current != '/':
        current = os.path.dirname(current)
        initfile = os.path.join(current, '__init__.py')
        init_found = os.path.exists(initfile)

    if not init_found:
        # search for breadth
        searchfrom = dirname(searchfrom)

        for folder in fileutil.listfolder(searchfrom):
            current = os.path.join(searchfrom, folder)
            initfile = os.path.join(current, '__init__.py')
            init_found = os.path.exists(initfile)
            if init_found:
                break

    while init_found:
        current = os.path.dirname(current)
        initfile = os.path.join(current, '__init__.py')
        init_found = os.path.exists(initfile)
        pack_found = not init_found

    return current if pack_found else None


def find_package_path(searchfrom):
    """
    package path. return None if failed to guess
    """

    current = searchfrom + '/'
    init_found = False
    pack_found = False
    while not init_found and current != '/':
        current = os.path.dirname(current)
        initfile = os.path.join(current, '__init__.py')
        init_found = os.path.exists(initfile)

    while init_found:
        current = os.path.dirname(current)
        initfile = os.path.join(current, '__init__.py')
        init_found = os.path.exists(initfile)
        pack_found = not init_found

    return current if pack_found else None


def append_sys_path(p):
    """
    append system path
    """
    if p not in sys.path:
        sys.path.insert(0, p)


def append_this_package_path(depth=1):
    """
    this_package.py 에서 사용
    import snipy.this_package
    """
    from .caller import caller

    logg.debug('caller module %s', caller.modulename(depth + 1))
    c = caller.abspath(depth + 1)
    logg.debug('caller path %s', c)

    p = guess_package_path(dirname(c))
    if p:
        logg.debug('appending sys path %s', p)
        append_sys_path(p)
    else:
        # do some logging
        logg.debug('failed to guess package path for: %s', c)


def import_this(packagename):
    if packagename in sys.modules:
        return
    import importlib
    append_this_package_path(depth=1)
    importlib.import_module(packagename)

