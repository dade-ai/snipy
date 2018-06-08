# -*- coding: utf-8 -*-

"""
https://en.wikipedia.org/wiki/ANSI_escape_code
http://wiki.bash-hackers.org/scripting/terminalcodes
http://ascii-table.com/ansi-escape-sequences.php
http://www.ecma-international.org/publications/files/ECMA-ST/Ecma-048.pdf
"""
import termios
import sys
# from dictobj import dictobj

try:
    # prepare terminal settings
    _fd = sys.stdin.fileno()
    _old_settings = termios.tcgetattr(_fd)
    _new_settings = termios.tcgetattr(_fd)
    _new_settings[3] &= ~termios.ICANON
    _new_settings[3] &= ~termios.ECHO

except Exception as e:
    # from .ilogging import logg
    # logg.warn('not a tty?: %s:' % e)
    pass


def echo(content):
    write(str(content))


def write(content):
    """
    write and flush terminal code and(or) text
    """
    sys.stdout.write(content)
    sys.stdout.flush()


def csi_fixed(code):
    return '\x1b[%s' % code


class csi(object):
    """
    ansi csi code simple abstraction
    """
    def __init__(self, code):
        self.code = '\x1b[%s' + code

    def __call__(self, n=''):
        return self.code % str(n)

    def __str__(self):
        return self.__call__()


class csi_move(csi):
    """
    csi move command
    """
    def __call__(self, x='', y=''):
        xy = '%s;%s' % (str(x), str(y))
        return self.code % xy


class TermCursor(object):
    """
    ansi codes for TermCursor movement
    """

    up = csi('A')
    down = csi('B')
    right = csi('C')
    left = csi('D')
    newline = downhome = csi('E')
    uphome = csi('F')
    movex = csi('G')
    move = csi_move('H')
    # home = csi('E')() + csi('A')()  # downhome(1) + up(1)
    home = '\r'
    save = csi_fixed('s')
    restore = csi_fixed('u')
    hide = csi_fixed('?25l')
    show = csi_fixed('?25h')

    @staticmethod
    def moveby(x, y):
        fx = TermCursor.right(x) if x > 0 else TermCursor.left(-x)
        fy = TermCursor.down(y) if y > 0 else TermCursor.up(-y)
        return fx + fy


def save_pos():
    write(TermCursor.save)


def restore_pos():
    write(TermCursor.restore)


class clear(object):
    """
    ansi codes for clearing terminal
    """
    screen_after = csi_fixed('J')
    screen_before = csi_fixed('1J')
    screen = csi_fixed('2J')
    line_after = csi_fixed('K')
    line_before = csi_fixed('1K')
    line = csi_fixed('2K')


class scroll(object):
    """
    ansi codes for scrolling
    """
    up = csi('S')
    down = csi('T')


def newline():
    write(csi('E'))


def writexy(xy, *args):
    """
    writes text on on screen
    a tuple as first argument gives the relative position to current TermCursor position
    does change TermCursor position
    args = list of optional position, formatting tokens and strings
    """
    write(TermCursor.moveby(*xy) + ''.join(args))


def put(xy, *args):
    """
    put text on on screen
    a tuple as first argument tells absolute position for the text
    does not change TermCursor position
    args = list of optional position, formatting tokens and strings
    """
    cmd = [TermCursor.save, TermCursor.move(*xy), ''.join(args), TermCursor.restore]
    write(''.join(cmd))


def getpassword(prompt="Password: "):
    """
    get user input without echo
    """

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    new[3] &= ~termios.ECHO          # lflags
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, new)
        passwd = raw_input(prompt)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return passwd


def getch():
    """
    get character. waiting for key
    """
    try:
        termios.tcsetattr(_fd, termios.TCSANOW, _new_settings)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(_fd, termios.TCSADRAIN, _old_settings)
    return ch


# if __name__ == '__main__':
#     t = getpassword()
#     print t
#
