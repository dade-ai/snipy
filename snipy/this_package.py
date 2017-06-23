# -*- coding: utf-8 -*-

"""
use case:

import snipy.this_package
import 하는 파일을 포함하는 패키지를 PYTHONPATH에 추가함
"""

from packageutil import append_this_package_path

append_this_package_path(depth=1)

