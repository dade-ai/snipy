# -*- coding: utf-8 -*-
from snipy.ilogging import logg
from snipy.basic import tuple_args
from snipy.progress import progress
import contextlib
import matplotlib.pyplot as plt
import numpy as np


@tuple_args
def plots(data, **kwargs):
    """
    simple wrapper plot with labels and skip x
    :param yonly_or_xy:
    :param kwargs:
    :return:
    """
    labels = kwargs.pop('labels', '')
    loc = kwargs.pop('loc', 1)

    # if len(yonly_or_xy) == 1:
    #     x = range(len(yonly_or_xy))
    #     y = yonly_or_xy
    # else:
    #     x = yonly_or_xy[0]
    #     y = yonly_or_xy[1:]

    lines = plt.plot(np.asarray(data).T, **kwargs)
    if labels:
        plt.legend(lines, labels, loc=loc)
    return lines


def grid_recommend(count, ratio=(2., 3.)):
    # ratio = (30, 40)
    unit = np.sqrt(count / float(np.prod(ratio)))
    h = np.ceil(ratio[0] * unit)
    w = np.ceil(count / float(h))
    return int(h), int(w)


def imshow_grid(images, grid=None, showfun=plt.imshow, **opt):
    """
    :param images: nhwc
    :return:
    """
    # assert images.ndim == 4  or list

    count = len(images)
    grid = grid or grid_recommend(count, sorted(images[0].shape[:2]))

    res = []
    for i, img in enumerate(images):
        # grid row first index
        plt.subplot2grid(grid, (i % grid[0], i // grid[0]))
        res.append(showfun(img.squeeze(), **opt))

    return res

# todo enhance. with enumerate


def plt_range(*args, **kwargs):
    """
    for i in plot_range(n):
        plt.imshow(imgs[i])

    left arrow yield prev value
    other key yield next value
    :param args:
    :return:
    """
    wait = kwargs.pop('wait', True)
    if not wait:
        # no interactive just pass range
        for i in progress(range(*args)):
            yield i
        return

    class _holder(object):
        pass
    hold = _holder()
    hold.i = 0
    hold.done = False

    def press(event):
        # import sys
        # sys.stdout.flush()
        hold.i += -1 if event.key == 'left' else 1
        hold.i = 0 if hold.i < 0 else hold.i

    def onclose(event):
        hold.done = True

    fig = kwargs.pop('fig', None)
    figsize = kwargs.pop('figsize', None)
    if fig is None:
        fig = plt.gcf()
        if figsize:
            fig.set_size_inches(figsize)
    elif isinstance(fig, (int, str)):
        if figsize:
            fig = plt.figure(fig, figsize=figsize)
        else:
            fig = plt.figure(fig)
    elif isinstance(fig, plt.Figure):
        if figsize:
            fig.set_size_inches(figsize)
    else:
        raise ValueError

    onkey_fig(press, fig)
    onclose_fig(onclose, fig)

    ranges = range(*args)
    l = len(ranges)

    while hold.i < l:
        print('hold.i', ranges[hold.i])
        yield ranges[hold.i]  # yield first without keypress
        before = hold.i
        while before == hold.i:
            while not fig.waitforbuttonpress(0.01):
                if hold.done:
                    return
            while fig.waitforbuttonpress(0.1):
                if hold.done:
                    return


def pause(interval):
    return plt.pause(interval)


def plot_pause(timeout=None, msg=''):
    """
    todo : add some example
    :param timeout: wait time. if None, blocking
    :param msg:
    :return:
    """

    if timeout is not None:
        print(msg or 'Press key for continue in time {}'.format(timeout))
        plt.waitforbuttonpress(timeout=timeout)
        return True

    print(msg or 'Press key for continue')
    while not plt.waitforbuttonpress(timeout=0.01):
        if not plt.get_fignums():
            return False
    return len(plt.get_fignums()) != 0


def ploting(iterable, timeout=None, msg=''):
    for res in iterable:
        yield res
        if not plot_pause(timeout, msg):
            break
    raise StopIteration


def flat_images(images, grid=None, bfill=1.0, bsz=(1, 1)):
    """
    convert batch image to flat image with margin inserted
    [B,h,w,c] => [H,W,c]
    :param images:
    :param grid: patch grid cell size of (Row, Col)
    :param bfill: board filling value
    :param bsz: int or (int, int) board size
    :return: flatted image
    """
    if images.ndim == 4 and images.shape[-1] == 1:
        images = images.squeeze(axis=-1)

    grid = grid or grid_recommend(len(images), sorted(images[0].shape[:2]))
    if not isinstance(bsz, (tuple, list)):
        bsz = (bsz, bsz)

    # np.empty()
    imshape = list(images.shape)
    imshape[0] = grid[0] * grid[1]
    imshape[1] += bsz[0]
    imshape[2] += bsz[1]

    # data = np.empty((grid[0] * grid[1], imshape[1], imshape[2]), dtype=images.dtype)
    data = np.empty(imshape, dtype=images.dtype)

    data.fill(bfill)
    bslice0 = slice(0, -bsz[0]) if bsz[0] else slice(None, None)
    bslice1 = slice(0, -bsz[1]) if bsz[1] else slice(None, None)

    data[:len(images), bslice0, bslice1] = images

    imshape = list(grid) + imshape[1:]  # [grid[0], grid[1], H, W, [Channel]]
    data = data.reshape(imshape)
    if len(imshape) == 5:
        data = data.transpose(0, 2, 1, 3, 4)
        imshape = [imshape[0]*imshape[2], imshape[1]*imshape[3], imshape[4]]
    else:  # len == 4
        data = data.transpose(0, 2, 1, 3)
        imshape = [imshape[0]*imshape[2], imshape[1]*imshape[3]]
    data = data.reshape(imshape)

    # remove last margin
    data = data[bslice0, bslice1]

    return data


def imshow_flat(images, grid=None, showfun=plt.imshow, bfill=1.0, bsz=(1,1), **opt):
    """
    imshow after applying flat_images
    :param images: [bhwc]
    :param grid: None for auto grid
    :param showfun: plt.imshow
    :param bfill: color for board fill
    :param bsz: size of board
    :param opt: option for showfun
    :return:
    """
    count = len(images)
    # decide grid shape if need pick one
    grid = grid or grid_recommend(count, ratio=sorted(images[0].shape[:2]))

    flatted = flat_images(images, grid, bfill=bfill, bsz=bsz)
    res = showfun(flatted, **opt)
    plt.draw()


def imshows_flat(images, subgrid=None, grid=None, showfun=plt.imshow, bfill=1.0, bsz=(1,1), **opt):
    count = len(images[0])

    grid = grid or grid_recommend(count, ratio=sorted(images[0][0].shape[:2]))
    flatted = [flat_images(im, grid, bfill=bfill, bsz=bsz) for im in images]
    subgrid = subgrid or (1, len(images))
    return imshow_grid(flatted, grid=subgrid, showfun=showfun, **opt)


@tuple_args
def imshow(imagez, grid=None, showfun=plt.imshow, bfill=1.0, bsz=(1,1), tight=True, **opt):
    if len(imagez) == 1:
        if imagez[0].ndim == 2:
            res = showfun(imagez[0], **opt)
            plt.draw()
        elif imagez[0].ndim == 3:
            res = showfun(imagez[0], **opt)
            plt.draw()
        else:
            res = imshow_flat(imagez[0], grid=grid, showfun=showfun, bfill=bfill, bsz=bsz, **opt)
    else:
        # note : imshows_flat
        res = imshows_flat(imagez, grid=grid, showfun=showfun, bfill=bfill, bsz=bsz, **opt)
    if tight:
        plt.tight_layout()
    return res


def onclose_fig(onclose, fig=None):
    fig = fig or plt.gcf()
    fig.canvas.mpl_connect('close_event', onclose)


def onkey_fig(onkey, fig=None):
    fig = fig or plt.gcf()
    fig.canvas.mpl_connect('key_press_event', onkey)


def matshow(*args, **kwargs):
    """
    imshow without interpolation like as matshow
    :param args:
    :param kwargs:
    :return:
    """
    kwargs['interpolation'] = kwargs.pop('interpolation', 'none')
    return plt.imshow(*args, **kwargs)

# region plot loop tool


class PlotLoop(object):
    """
    imshow grid for looping speed up
    http://stackoverflow.com/questions/33602185/speed-up-plotting-images-in-matplotlib

    Example:

        import sflow.python.ploting as plt
        plot = plt.plotloop(plt.imshow_flat)
        for i in range(nstep):
            if i % 50 == 0:
                out = sess.run([styled, trainop])[0]
                plot(out)
                plt.pause(0.0001)
                print(i)
            else:
                l = sess.run([loss_content, loss_style, trainop])[:-1]
                print(i, l)
        plt.plot_pause()

    """

    def __init__(self, showfun=imshow, fig=None, **drawopt):
        self.showfun = showfun
        self.fig = fig or plt.figure()
        self.drawopt = drawopt
        self.setdata = None
        self.onclose = drawopt.pop('onclose', self._onclose)

    def _onclose(self, evt):
        # import sys
        # print('figure closed, exit')
        # sys.exit(0)
        pass

    def __call__(self, *args, **kwargs):

        if not self.setdata:
            if self.fig is None:
                self.fig = plt.figure()
            self.fig.canvas.mpl_connect('close_event', self.onclose)
            kwargs.update(self.drawopt)
            plt.figure(self.fig.number)
            self.setdata = self.showfun(*args, **kwargs)

            plt.show(block=False)
        else:
            # todo@dade : fixme
            try:
                self.setdata.set_data(*args, **kwargs)
            except AttributeError:
                plt.figure(self.fig.number)
                self.showfun(*args, **kwargs)

            self.fig.canvas.draw()
        plt.pause(0.0001)


class PlotGrid(object):
    """
    imshow grid for looping speed up
    http://stackoverflow.com/questions/33602185/speed-up-plotting-images-in-matplotlib

    """

    def __init__(self, grid=None, ratio=(2., 4.), showfun=imshow, fig=None, onclose=None, **drawopt):
        self.showfun = showfun
        self.ratio = ratio
        self.grid = grid
        self.setdata = []
        # self.figsize = drawopt.pop('figsize', None)
        self.drawopt = drawopt
        self.fig = fig or plt.figure()
        self.onclose = onclose or self._onclose

    def _onclose(self, evt):
        # import sys
        # print('figure closed, exit')
        # sys.exit(0)
        pass

    def __call__(self, images, **kwargs):
        grid = self.grid or grid_recommend(len(images), kwargs.pop('ratio', self.ratio))

        if not self.setdata:
            if self.fig is None:
                self.fig = plt.figure()
            self.fig.canvas.mpl_connect('close_event', self.onclose)
            kwargs.update(self.drawopt)
            for i, img in enumerate(images):
                plt.subplot2grid(grid, (i % grid[0], i // grid[0]))
                self.setdata.append(self.showfun(img.squeeze(), **kwargs))

        else:
            # todo@dade : fixme
            for i, img in enumerate(images):
                self.setdata[i].set_data(img.squeeze())
            # except AttributeError:
            #     for i, img in enumerate(images):
            #         plt.subplot2grid(grid, (i % grid[0], i // grid[0]))
            #         self.showfun(img.squeeze(), **kwargs)

            self.fig.canvas.draw()


plotloop = PlotLoop
plotgrid = PlotGrid


#


# region overlay and objects


def imshow_overlay(im1, im2, alpha, **kwargs):
    plt.imshow(im1)
    plt.hold(True)
    return plt.imshow(im2, alpha=alpha, **kwargs)


def imbox(xy, w, h, angle=0.0, **kwargs):
    """
    draw boundary box
    :param xy: start index xy (ji)
    :param w: width
    :param h: height
    :param angle:
    :param kwargs:
    :return:
    """
    from matplotlib.patches import Rectangle
    return imbound(Rectangle, xy, w, h, angle, **kwargs)


def imrect(xy, w, h, angle=0.0, **kwargs):
    # http://matplotlib.org/api/patches_api.html#matplotlib.patches.Rectangle
    from matplotlib.patches import Rectangle
    return impatch(Rectangle, xy, w, h, angle, **kwargs)


def imbound(clspatch, *args, **kwargs):
    """
    :param clspatch:
    :param args:
    :param kwargs:
    :return:
    """
    # todo : add example

    c = kwargs.pop('color', kwargs.get('edgecolor', None))
    kwargs.update(facecolor='none', edgecolor=c)
    return impatch(clspatch, *args, **kwargs)


def impatch(clspatch, *args, **kwargs):
    # http://matplotlib.org/api/patches_api.html#matplotlib.patches.Rectangle
    from matplotlib.patches import Rectangle
    ax = plt.gca()
    clspatch = clspatch or Rectangle
    patch = clspatch(*args, **kwargs)
    ax.add_patch(patch)

    return patch


def bar(values, labels=None, txtcolor='white', fmt='%s', txtoffset=-0.05, **kwargs):
    x = range(len(values))
    res = plt.bar(x, values)
    ax = plt.gca()
    if labels is not None:
        plt.xticks(np.asarray(x) + 0.4, labels)

    for ix, v in zip(x, values):
        ax.text(ix + 0.4, v + txtoffset, fmt % v, color=txtcolor, horizontalalignment='center',
                weight='bold', **kwargs)

    return res, ax

# endregion


# region superpixel

def imslic(img, n_segments=100, aspect=None):
    """
    slic args :
    n_segments=100, compactness=10., max_iter=10,
    sigma=0, spacing=None,
    multichannel=True, convert2lab=None, enforce_connectivity=True,
    min_size_factor=0.5, max_size_factor=3, slic_zero=False

    mark_boundaries args:
    label_img, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0

    imshow args:
    cmap=None, norm=None, aspect=None, interpolation=None,
    alpha=None, vmin=None, vmax=None, origin=None,
    extent=None, shape=None, filternorm=1, filterrad=4.0,
    imlim=None, resample=None, url=None, hold=None, data=None,

    :param img:
    :param slicarg:
    :param slickw:
    :return:
    """
    from skimage.segmentation import (slic, mark_boundaries)
    from skimage.morphology import (dilation)

    if img.ndim == 2 or img.ndim == 3 and img.shape[-1] == 1:
        imz = np.stack([img, img, img], 2)
    else:
        imz = img

    slics = slic(imz, n_segments=n_segments)

    boundaries = mark_boundaries(imz, slics)
    return plt.imshow(boundaries, aspect=aspect)


def imslic2(img, n_segments=100, color=None, outline_color=None, mode='thick', **kwargs):
    """
    slic args :
    n_segments=100, compactness=10., max_iter=10,
    sigma=0, spacing=None,
    multichannel=True, convert2lab=None, enforce_connectivity=True,
    min_size_factor=0.5, max_size_factor=3, slic_zero=False

    mark_boundaries args:
    label_img, color=(1, 1, 0), outline_color=None, mode='outer', background_label=0

    imshow args:
    cmap=None, norm=None, aspect=None, interpolation=None,
    alpha=None, vmin=None, vmax=None, origin=None,
    extent=None, shape=None, filternorm=1, filterrad=4.0,
    imlim=None, resample=None, url=None, hold=None, data=None,

    :param img:
    :param slicarg:
    :param slickw:
    :return:
    """
    from skimage.segmentation import (slic, find_boundaries) # mark_boundaries
    from skimage.morphology import (dilation)

    kwslic = {'compactness', 'max_iter', 'sigma', 'spacing', 'multichannel', 'convert2lab',
              'enforce_connectivity', 'min_size_factor', 'max_size_factor', 'slic_zero=False'}
    imshowkw = {'cmap', 'norm', 'aspect', 'interpolation', 'alpha', 'vmin', 'vmax', 'origin',
                'extent', 'shape', 'filternorm', 'filterrad', 'imlim', 'resample', 'url', 'hold', 'data'}

    slicarg = {k: v for k, v in kwargs.iteritems() if k in kwslic}
    imshowarg = {k: v for k, v in kwargs.iteritems() if k in imshowkw}

    if img.ndim == 2 or img.ndim == 3 and img.shape[-1] == 1:
        imz = np.stack([img, img, img], 2)
        color = color or 1.
    else:
        imgz = img
        color = color or (1,1,0)

    slics = slic(imz, n_segments=n_segments, **slicarg)

    boundaries = find_boundaries(slics, mode=mode)
    if outline_color is not None:
        outlines = dilation(boundaries, np.ones((3, 3), np.uint8))
        img[outlines] = outline_color
    img[boundaries] = color
    return plt.imshow(img, **imshowarg)

# endregion

# region movie

# from matplotlib.animation import (FFMpegWriter, ImageMagickFileWriter, AVConvWriter,
#                                   FFMpegFileWriter, ImageMagickWriter, AVConvFileWriter)
from matplotlib.animation import (FFMpegWriter, ImageMagickWriter, AVConvWriter)


class PlotMovieWriter(object):
    """
    PlotLoop + MovieWriter
    Example:
        import sflow.python.ploting as plt
        plot = plt.PlotMovieWriter(plt.imshow_flat, outputfile, dpi=100)
        for i in range(nstep):
            if i % 50 == 0:
                out = sess.run([styled, trainop])[0]
                plot(out)
                # plt.pause(0.0001)
                print(i)
            else:
                l = sess.run([loss_content, loss_style, trainop])[:-1]
                print(i, l)
        plot.finish()

        plt.plot_pause()

    """

    def __init__(self, outfile, showfun=plt.imshow, fig=None, drawopt=None, dpi=100, **movieopt):

        self.showfun = showfun
        self.fig = fig or plt.figure()
        drawopt = drawopt or dict()
        self.drawopt = drawopt
        self.setdata = None
        self.onclose = drawopt.pop('onclose', self._onclose)
        # for movie writing
        self.moviewriter = None
        self.movieopt = movieopt

        self.outfile = outfile
        self.dpi = dpi
        self._first = True

    def setup_movie(self, fig):
        # create moviewriter
        # then setup
        # self.movieopt.pop()

        # fps=5, codec=None, bitrate=None, extra_args=None, metadata
        self.moviewriter = FFMpegWriter(**self.movieopt)
        self.moviewriter.setup(fig, self.outfile, self.dpi)

    def _onclose(self, evt):
        # import sys
        # print('figure closed, exit')
        # sys.exit(0)
        pass

    def __call__(self, *args, **kwargs):

        if not self.setdata:
            if self.fig is None:
                self.fig = plt.figure()
                # self.fig = self.setup_figure()
                self.fig.canvas.mpl_connect('close_event', self.onclose)
            kwargs.update(self.drawopt)
            self.setdata = self.showfun(*args, **kwargs)
            if self._first:
                self.setup_movie(self.fig)
                self._first = False

            plt.show(block=False)
        else:
            # todo@dade : if error? showfun?
            try:
                self.setdata.set_data(*args, **kwargs)
            except AttributeError:
                self.showfun(*args, **kwargs)

            self.fig.canvas.draw()

        self.grab()

    def grab(self):
        self.moviewriter.grab_frame()

    def finish(self):
        if self._first:
            raise ValueError('setup not called')
        self.moviewriter.finish()
        logg.info('movie saved to [{}]'.format(self.outfile))


def get_ax_size(fig, ax):
    bbox = ax.get_window_extent()
    x_pad = ax.xaxis.get_tick_padding()
    y_pad = ax.yaxis.get_tick_padding()
    bbox.x0 += x_pad
    bbox.x1 -= x_pad
    bbox.y0 += y_pad
    bbox.y1 -= y_pad
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    # width *= fig.dpi
    # height *= fig.dpi
    return width, height


def tight_figure(fig):
    a = fig.gca()
    a.set_position([0, 0, 1, 1])
    a.set_xticks([])
    a.set_yticks([])

    # a.xaxis.set_tick_params(which='both', pad=0, reset=True)
    # a.yaxis.set_tick_params(which='both', pad=0, reset=True)
    plt.axis('off')
    fig.patch.set_visible(False)
    # force update and get axsize
    plt.show(False)
    w, h = get_ax_size(fig, a)
    fig.set_size_inches(w, h, forward=True)


class ImageMovieWriter(PlotMovieWriter):

    def __init__(self, outfile, *args, **kwargs):
        super(ImageMovieWriter, self).__init__(outfile, *args, **kwargs)

    def setup_movie(self, fig):
        # remove boarders
        tight_figure(fig)
        super(ImageMovieWriter, self).setup_movie(fig)


@contextlib.contextmanager
def movie_saving(outfile, showfun=imshow, fig=None, tight=True, drawopt=None, dpi=100, **movieopt):
    """
    contextmanager for PlotMovieWriter
    Example:

        with movie_saving('output.mp4', dpi=100) as plot:
            for i in range(10):
                plot(data[i])

    :param outfile:
    :param showfun:
    :param fig:
    :param tight:
    :param drawopt:
    :param dpi:
    :param movieopt: fps=5, codec=None, bitrate=None, extra_args=None, metadata=None
    :return:
    """
    if tight:
        plot_writer = ImageMovieWriter(outfile, showfun=showfun, fig=fig, drawopt=drawopt, dpi=dpi, **movieopt)
    else:
        plot_writer = PlotMovieWriter(outfile, showfun=showfun, fig=fig, drawopt=drawopt, dpi=dpi, **movieopt)

    try:
        yield plot_writer
    finally:
        plot_writer.finish()


# endregion


# if __name__ == '__main__':
#     import numpy as np
#     a = np.random.rand(4, 3, 5, 3)
#     b = flat_images(a, (2, 2), 0., (1, 1))
#     # plot_pause()
#     print('done')
#
