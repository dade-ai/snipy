# -*- coding: utf-8 -*-
from . import img
from . import io
try:
    from . import plt
except NameError:
    # matplotlib not imported
    print('warning : matplotlib not imported')
    pass
from .flags import *

