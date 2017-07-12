# -*- coding: utf-8 -*-
from __future__ import absolute_import
import matplotlib.pyplot as plt
from six import wraps
from matplotlib._cm import datad

# reverse all the colormaps.
# reversed colormaps have '_r' appended to the name.

# http://matplotlib.org/examples/color/colormaps_reference.html
# _cmaps = [('Perceptually Uniform Sequential', ['viridis', 'plasma', 'inferno', 'magma']),
#          ('Sequential', ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#                          'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#                          'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
#          ('Sequential (2)', ['binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
#                              'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
#                              'hot', 'afmhot', 'gist_heat', 'copper']),
#          ('Diverging', ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
#                         'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
#          ('Qualitative', ['Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3',
#                           'tab10', 'tab20', 'tab20b', 'tab20c']),
#          ('Miscellaneous', ['flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
#                             'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
#                             'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]


def _cmap_fun(cm):

    def wrapper(x, alpha=None, bytes=False, lut=None):
        cmap = plt.get_cmap(cm, lut=lut)
        return cmap(x, alpha=alpha, bytes=bytes)
    return wrapper


# explicte declaration (for example)
def jet(x, alpha=None, bytes=False, lut=None):
    return _cmap_fun('jet')(x, alpha=alpha, bytes=bytes, lut=lut)


# fill functions
locals().update({cm: _cmap_fun(cm) for cm in datad.keys()})
locals().update({cm: _cmap_fun(cm+'_r') for cm in datad.keys()})


