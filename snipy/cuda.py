# -*- coding: utf-8 -*-

def get_cuda_device_count():
    import pycuda
    import pycuda.autoinit
    from pycuda.driver import Device

    return Device.count()

