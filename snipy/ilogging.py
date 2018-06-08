# -*- coding: utf-8 -*-
import logging
import functools
import sys
from .decotool import optional
from .ansi import ansi
import collections


class FormatterX(logging.Formatter):

    """
    formatter extension.
    1) {} style formatting
    2) different format by level
    """

    def __init__(self, fmt=None, datefmt=None, **fmt_level):
        self.fmt_ = fmt or '{message}'
        self.fmt = fmt_level or dict()  # format by level
        self.datefmt = datefmt or "%m-%d %H:%M"

    def usesTime(self):
        return self._fmt.find("{asctime}") >= 0

    def getfmt(self, levelname):
        return self.fmt.get(levelname, self.fmt_)

    def setFormat(self, **kwargs):
        self.fmt.update(kwargs)

    def format(self, record):
        """tweaked from source of base"""
        try:
            record.message = record.getMessage()
        except TypeError:
            # if error during msg = msg % self.args
            if record.args:
                if isinstance(record.args, collections.Mapping):
                    record.message = record.msg.format(**record.args)
                else:
                    record.message = record.msg.format(record.args)
        self._fmt = self.getfmt(record.levelname)
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)

        s = self._fmt.format(**record.__dict__)

        if record.exc_info:
            # Cache the traceback text to avoid converting it multiple times
            # (it's constant anyway)
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            if s[-1:] != '\n':
                s += '\n'
            try:
                s = s + record.exc_text
            except UnicodeError:
                s = s + record.exc_text.decode(sys.getfilesystemencoding(), 'replace')
        return s

# noinspection PyStatementEffect
default_format = {
    'DEBUG': ansi.yellow('[D] {asctime}: [{funcName}()] {message}'),
    'WARNING': ansi.light_yellow('[W] {asctime}: [{funcName}()] {message}'),
    'INFO': ansi.light_green('[I] {asctime}: [{funcName}()] {message}'),
    'ERROR': ansi.light_red('[E] {asctime}: [{filename}][{lineno}] [{funcName}()] {message}'),
    'CRITICAL': ansi.bold('[F] {asctime}: [{filename}][{lineno}] [{funcName}()] {message}') / ansi.dark_red
}

# common formatter, handler
formatter = FormatterX(**default_format)
default_handler = logging.StreamHandler()
default_handler.setFormatter(formatter)


class Loggerx(logging.Logger):

    def _log(self, *args, **kwargs):
        if self.disabled:
            return
        super(Loggerx, self)._log(*args, **kwargs)


# register logger class
logging.setLoggerClass(Loggerx)


def getlogger(pkg='', handler=None):
    """
    패키지 혹은 채널 로거
    logging.getLogger(package_name) or logg.getLogger()
    :param pkg: str
    """
    from .caller import caller

    if not pkg:
        m = caller.modulename()
        s = m.split('.', 1)
        if len(s) > 1:
            pkg = s[0]

    if haslogger(pkg):
        return logging.getLogger(pkg)
    else:
        # local
        logger = logging.getLogger(pkg)
        logger.addHandler(handler or default_handler)
        logger.setLevel(logging.DEBUG)
        return logger


def haslogger(name):
    """
    name의 로거가 있는지 없는지 체크. root logger는 무조건 만들어진다.
    """
    return name in logging.Logger.manager.loggerDict or not name


# function shortcuts
logg = getlogger('snipy')
info = logg.info
warn = logg.warn
debug = logg.debug
fatal = logg.critical
error = logg.error
exception = logg.exception


@optional
class trace(object):
    """decorator. 해당 함수 콜할때 마다 로그를 남기개 함.
    @trace
    def fx(x):
    등과 같이 쓰던지

    @trace(level=logging.INFO)
    def fx(x):
    등과 같이 사용하면 됨

    @trace(level=logging.INFO, logger='package')
    def fx(x):

    @trace(level=logging.INFO, logger=logger)
    def fx(x):

    데코레이터를 포함시킨 함수가 있는 모듈에 해당 모듈을 이름으로 하는 ogger 객체가 있다면.
    logger.log를, 없다면 logging.log를 호출함.
    """

    def __init__(self, level=logging.DEBUG, logger=None):
        self.level = level
        if logger is None or isinstance(logger, str):
            logger = getlogger(logger)
        self.logger = logger

    def __call__(self, func):
        from .caller import caller

        def make_message(args, kwargs):
            params = map(str, args) + ['%s=%s' % (k, v) for k, v in kwargs.items()]
            msg = "%s(%s) @ %s" % (func.__name__, ', '.join(params), caller.funname(depth=2))
            return msg

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            msg = make_message(args, kwargs)
            self.logger.log(self.level, msg)

            return func(*args, **kwargs)

        return wrapped


def basicConfig(**kw):
    """logging의 로그를 한번 호출하면 basicConfig가 안먹으므로. 기존 핸들러 삭제후 재설정.
    http://stackoverflow.com/questions/1943747/python-logging-before-you-run-logging-basicconfig
    ex)
    basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
    :param filename: Specifies that a FileHandler be created, using the specified filename, rather than a StreamHandler.
    :param filemode: Specifies the mode to open the file, if filename is specified (if filemode is unspecified, it defaults to ‘a’).
    :param format: Use the specified format string for the handler. (https://docs.python.org/2.7/library/logging.html#logging.basicConfig
    :param datefmt: Use the specified date/time format.
    :param level: Set the root logger level to the specified level.
    :param stream: Use the specified stream to initialize the StreamHandler. Note that this argument is incompatible with ‘filename’ - if both are present, ‘stream’ is ignored.

    """
    while len(logging.root.handlers) > 0:
        logging.root.removeHandler(logging.root.handlers[-1])

    logging.basicConfig(**kw)
