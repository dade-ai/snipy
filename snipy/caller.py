# -*- coding: utf-8 -*-
"""
caller, stack info
"""
import os
import inspect
import sys


class caller(object):

    """
    caller stack inspection, 호출하는 함수 입장에서 caller
    caller.path()
    caller.abspath()
    caller.modulename()
    caller.funname()
    ex) def p():
            # funname of calling p()
            callername = caller.funname()

    """

    extra_depth = 1

    @classmethod
    def path(cls, depth=1):
        """ caller path (*.py)
        """
        depth += cls.extra_depth
        return inspect.stack()[depth][1]

    @classmethod
    def abspath(cls, depth=1):
        """ caller path (*.py)
        """
        return os.path.abspath(cls.path(depth + 1))

    @classmethod
    def funname(cls, depth=1):
        depth += cls.extra_depth
        return inspect.stack()[depth][3]

    @classmethod
    def modulename(cls, depth=1):
        """
        get caller's __name__
        """
        depth += cls.extra_depth
        frame = sys._getframe(depth)
        return frame.f_globals['__name__']

    @classmethod
    def packagename(cls, depth=1):
        """
        get package name by splitting __name__
        """
        name = cls.modulename(depth+1)
        return name.split('.', 1)[0]


class this(caller):

    """
    this stack inspection, 호출하는 곳에서.
    this.path()
    this.abspath()
    this.modulename()
    this.funname()
    ex) def p():
            assert 'p' == this.funname()
    """
    extra_depth = 0

