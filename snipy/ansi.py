# -*- coding: utf-8 -*-

"""
https://en.wikipedia.org/wiki/ANSI_escape_code
http://wiki.bash-hackers.org/scripting/terminalcodes
http://ascii-table.com/ansi-escape-sequences.php
http://www.ecma-international.org/publications/files/ECMA-ST/Ecma-048.pdf
"""


def ansi_str(s, fmt):
    if isinstance(s, str):
        return '\x1b[%sm%s\x1b[0m' % (fmt, s)
    else:
        return '\x1b[%sm%s\x1b[0m' % (fmt, str(s))


def ansi_str_restore(s, fmt, patt='\x1b[0m'):
    fmt2 = '\x1b[0;%sm' % fmt
    a = str(s).replace(patt, fmt2)

    return '\x1b[%sm%s\x1b[0m' % (fmt, a)


class _AnsiBuilder(object):
    def __init__(self, fmt):
        self.fmt = fmt

    def __call__(self, s):
        """ make ansi string
        """
        return ansi_str_restore(s, self.fmt)

    __radd__ = __call__  # same

    def __add__(self, other):
        """ make ansi string (simple ver.)
        """
        return ansi_str(other, self.fmt)

    __rmul__ = __rdiv__ = __add__

    def __mul__(self, other):
        """ merge ansi format
        """
        self.fmt += ';' + other.fmt
        return self


class _AttrAnsi(_AnsiBuilder):
    def __mul__(self, other):
        return _AnsiBuilder(self.fmt) * other


class _ColorAnsi(object):
    def __init__(self, c, fmt=''):
        self.c = c
        self.fg = fmt + str(30 + c)
        self.bg = fmt + str(40 + c)

    def __call__(self, s):
        return ansi_str_restore(s, self.fg)

    __radd__ = __call__

    def __add__(self, other):
        return ansi_str(other, self.fg)

    def __mul__(self, other):
        return _AnsiBuilder(self.fg) * other

    def __rmul__(self, other):
        return other * _AnsiBuilder(self.fg)

    def __div__(self, other):
        return _AnsiBuilder(self.fg) / other

    def __rtruediv__(self, other):
        # python3
        return _AnsiBuilder(self.fg) / other

    def __rdiv__(self, other):
        return other * _AnsiBuilder(self.bg)

    def __rtruediv__(self, other):
        return other * _AnsiBuilder(self.bg)

    @property
    def fmt(self):
        return self.fg


class _NoAnsi(object):
    def __init__(self):
        pass

    def __call__(self, s):
        # if isinstance(s, (str, basestring)):  # python2 only
        if isinstance(s, str):
            return s
        else:
            return str(s)

    __radd__ = __add__ = __rmul__ = __rdiv__ = __call__

    def __mul__(self, _):
        return self

    __div__ = __mul__


noAnsi = _NoAnsi()


class ansi(object):

    attrName = ['normal', 'bold', 'dark', 'italic', 'underline', 'blink', 'rapid', 'negative']
    colorName = ['black', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white']

    normal = _AttrAnsi('0')
    bold = _AttrAnsi('1')
    dark = _AttrAnsi('2')

    italic = _AttrAnsi('3')
    underline= _AttrAnsi('4')
    blink = _AttrAnsi('5')
    rapid = _AttrAnsi('6')
    negative = _AttrAnsi('7')
    light = bold
    #blink = attr('5')

    black = _ColorAnsi(c=0)
    red = _ColorAnsi(c=1)
    green = _ColorAnsi(c=2)
    yellow = _ColorAnsi(c=3)
    blue = _ColorAnsi(c=4)
    magenta = _ColorAnsi(c=5)
    cyan = _ColorAnsi(c=6)
    white = _ColorAnsi(c=7)

    light_black = _ColorAnsi(c=0, fmt='1;')
    light_red = _ColorAnsi(c=1, fmt='1;')
    light_green = _ColorAnsi(c=2, fmt='1;')
    light_yellow = _ColorAnsi(c=3, fmt='1;')
    light_blue = _ColorAnsi(c=4, fmt='1;')
    light_magenta = _ColorAnsi(c=5, fmt='1;')
    light_cyan = _ColorAnsi(c=6, fmt='1;')
    light_white = _ColorAnsi(c=7, fmt='1;')

    dark_black = _ColorAnsi(c=0, fmt='2;')
    dark_red = _ColorAnsi(c=1, fmt='2;')
    dark_green = _ColorAnsi(c=2, fmt='2;')
    dark_yellow = _ColorAnsi(c=3, fmt='2;')
    dark_blue = _ColorAnsi(c=4, fmt='2;')
    dark_magenta = _ColorAnsi(c=5, fmt='2;')
    dark_cyan = _ColorAnsi(c=6, fmt='2;')
    dark_white = _ColorAnsi(c=7, fmt='2;')

    @staticmethod
    def colors():
        return [ansi.__dict__[c] for c in ansi.colorName]

    @staticmethod
    def ansiattr():
        return [ansi.__dict__[c] for c in ansi.attrName]


def enable_color():
    d = dict()

    for i, name in enumerate(ansi.attrName):
        d[name] = _AttrAnsi(str(i))

    d['light'] = d['bold']

    for i, name in enumerate(ansi.colorName):
        d[name] = _ColorAnsi(c=i)
        d['light_' + name] = _ColorAnsi(c=i, fmt='1;')
        d['dark_' + name] = _ColorAnsi(c=i, fmt='2;')

    for k, v in d.items():
        setattr(ansi, k, v)
    globals().update(d)


def disable_color():
    d = dict()

    for i, name in enumerate(ansi.attrName):
        d[name] = noAnsi

    d['light'] = noAnsi

    for i, name in enumerate(ansi.colorName):
        d[name] = noAnsi
        d['light_' + name] = noAnsi
        d['dark_' + name] = noAnsi

    for k, v in d.items():
        setattr(ansi, k, v)
    globals().update(d)


global normal, bold, dark, italic, underline, blink, rapid, negative, light
global black, red, green, yellow, blue, magenta, cyan, white
global light_black, light_red, light_green, light_yellow, light_blue, light_magenta, light_cyan, light_white
global dark_black, dark_red, dark_green, dark_yellow, dark_blue, dark_magenta, dark_cyan, dark_white

