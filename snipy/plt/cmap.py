# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from functools import wraps
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


def cmap_fun(cm):

    def wrapper(x, alpha=None, bytes=False, lut=None, clim=None, norm=None):
        cmap = plt.get_cmap(cm, lut=lut)
        if norm is None:
            clim = clim or (0., 1.)
            norm = plt.Normalize(vmin=clim[0], vmax=clim[1])
        x = norm(x)

        return cmap(x, alpha=alpha, bytes=bytes)

    return wrapper


# explicte declaration (for example)
def jet(x, alpha=None, bytes=False, lut=None, clim=None, norm=None):
    return cmap_fun('jet')(x, alpha=alpha, bytes=bytes, lut=lut, clim=clim, norm=norm)


# fill functions
locals().update({cm: cmap_fun(cm) for cm in datad.keys()})
locals().update({cm+'_r': cmap_fun(cm + '_r') for cm in datad.keys()})


