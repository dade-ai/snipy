# -*- coding: utf-8 -*-
import os
from codecs import open
import scandir
from snipy.ilogging import logg
import numpy as np


def mkdir_if_not(filepath, ispath=False):
    """
    path 부분이 없으면 mkdir 을 한다.
    :param filepath: 파일 패쓰
    :return: filpath 그대로 리턴
    """
    if not ispath:
        p, _ = os.path.split(filepath)
    else:
        p = filepath

    if not p:
        return filepath
    if not os.path.exists(p):
        # M.info('%s not exist, trying mkdir ', p)
        try:
            os.makedirs(p)
        except FileExistsError as e:
            logg.warn(str(e))

    return filepath


def readlines(filepath):
    """
    read lines from a textfile
    :param filepath:
    :return: list[line]
    """
    with open(filepath, 'rt') as f:
        lines = f.readlines()
        lines = map(str.strip, lines)
        lines = [l for l in lines if l]
    return lines


def writelines(filepath, lines):
    mkdir_if_not(filepath)
    with open(filepath, 'wt') as f:
        for l in lines:
            f.write(l + '\n')


def readtxt(filepath):
    """ read file as is"""
    with open(filepath, 'rt') as f:
        lines = f.readlines()
    return ''.join(lines)


def writetxt(filepath, txt):
    mkdir_if_not(filepath)
    with open(filepath, 'wt') as f:
        f.write(txt)


def savefile(obj, filepath, compress=True):
    """
    파일 있으면 덮어씀
    :param obj:
    :param str filepath:
    :param compress:
    :return:
    """
    try:
        import cPickle as pickle
    except Exception:
        import pickle
    import joblib

    # 일단 임시 파일에 저장.
    tmpfile = filepath + '.tmp'
    mkdir_if_not(tmpfile)
    if compress:
        joblib.dump(obj, tmpfile, compress=3, cache_size=100, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        joblib.dump(obj, tmpfile, compress=0)

    os.rename(tmpfile, filepath)

    return obj


def loadfile(filepath, mmap_mode=None):
    """
    :param filepath:
    :param mmap_mode: {None, ‘r+’, ‘r’, ‘w+’, ‘c’} see. joblib.load
    :return:
    """
    import joblib

    try:
        return joblib.load(filepath, mmap_mode=mmap_mode)
    except IOError:
        return None


def latest_file(fpattern, matchfun=None):
    """
    regular format으로 조회한 파일리스트중에 가장최신 파일 path리턴 file modified time 조회 비교
    :param function matchfun:
    :param fpattern: 파일 패턴 ( ex: data/some*.txt)
    :return: data/somelatest.txt, None 이면 파일이 없는 것
    """
    import glob
    # matchfun = matchfun or glob.glob
    files = glob.glob(fpattern)
    if matchfun:
        files = filter(matchfun, files)

    latest, maxtime = None, 0
    for f in files:
        t = os.path.getmtime(f)
        if t > maxtime:
            latest, maxtime = f, t

    return latest


def load_latest(fpattern):
    latest = latest_file(fpattern)
    if latest is None:
        return None
    else:
        return loadfile(latest)


def load_or_run(filepath, fun, *args, **kwargs):
    """
    계산된 결과 파일이 있으면 로딩하고, 없으면 계산후 저장
    ex)
    res = load_or_run('file_loadorsave', funlongtime, ...., force=False)
    :param filepath:
    :param fun:
    :param force:
    :return:
    """
    force = kwargs.pop('force', False)
    compress = kwargs.pop('compress', True)

    if not filepath.startswith('/') or not filepath.startswith('~'):
        filepath = os.path.join('/tmp/snipy/load_or_run/', filepath)

    if not force and os.path.exists(filepath):
        # 저장되어 있는 것 로딩
        mmap_mode = 'r+' if not compress else None
        return loadfile(filepath, mmap_mode=mmap_mode)

    res = fun(*args, **kwargs)
    savefile(res, filepath, compress=compress)

    return res


def readhdf5(f):
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    import h5py
    warnings.resetwarnings()

    return h5py.File(f, 'r')


def getpath(f, isfile=True):
    if isfile:
        return os.path.dirname(os.path.realpath(f))
    else:
        return os.path.realpath(f)


# list directory, file, recursive or not, ...
# import scandir

def any_match(fname, patterns, matchfun=None):
    """
    ANY matches?
    :param str fname: file name
    :param list[str] patterns: list of filename pattern. see fnmatch.fnamtch
    :rtype: bool
    """
    return any(fnmatches(fname, patterns, matchfun))


def fnmatches(fname, patterns, matchfun):
    """"
    matches?
    :param fname: file name
    :type fname: str
    :param patterns: list of filename pattern. see fnmatch.fnamtch
    :type patterns: [str]
    :rtype: generator of bool
    """
    import fnmatch
    matchfun = matchfun or fnmatch.fnmatch
    for p in patterns:
        yield matchfun(fname, p)


def listdir(p, match='*', exclude='', listtype='file', matchfun=None):
    """
    list file(or folder) for this path (NOT recursive)
    :param p:
    :param match:
    :param exclude:
    :param listtype: ('file' | 'filepath' |'dir' | 'all')
    :param matchfun: match fun (default fnmatch.fnmatch) True/False = matchfun(name, pattern)
    :rtype:
    """
    if listtype == 'file':
        gen = listfile(p)
    elif listtype == 'filepath':
        gen = listfilepath(p)
    elif listtype == 'dir':
        gen = listfolder(p)
    elif listtype == 'dirpath':
        gen = listfolderpath(p)
    else:  # list file or folder
        gen = (entry.name for entry in scandir.scandir(p))

    return filter_pattern(gen, match, exclude, matchfun)


def filter_pattern(gen, include='*', exclude='', matchfun=None):
    pred = _pred_pattern(include, exclude, matchfun)
    for f in gen:
        if pred(f):
            yield f


def listfile(p):
    """
    generator of list files in the path.
    filenames only
    """
    try:
        for entry in scandir.scandir(p):
            if entry.is_file():
                yield entry.name
    except OSError:
        return


def listfilepath(p):
    """
    generator of list files in the path.
    filenames only
    """
    for entry in scandir.scandir(p):
        if entry.is_file():
            yield entry.path


def listfolder(p):
    """
    generator of list folder in the path.
    folders only
    """
    for entry in scandir.scandir(p):
        if entry.is_dir():
            yield entry.name


def listfolderpath(p):
    """
    generator of list folder in the path.
    folders only
    """
    for entry in scandir.scandir(p):
        if entry.is_dir():
            yield entry.path


def get_match_fun(patterns, patterntype):
    import re

    def _fnmatch(fname):
        return any_match(fname, patterns, patterntype)

    if patterntype != 're':  # usually just fnmatch
        return _fnmatch

    patterns = [re.compile(p) for p in patterns]

    def _re_match(fname):
        for p in patterns:
            if p.match(fname):
                return True
        return False

    return _re_match


def _is_str(x):
    
    try:
        return isinstance(x, (str, basestring))
    except NameError:
        return isinstance(x, str)


def _pred_pattern(match='*', exclude='', patterntype='fnmatch'):

    """ internal use """
    m, x = match, exclude
    if m == '*':
        if not x:
            pred = lambda n: True
        else:
            x = [x] if _is_str(x) else x
            matcher = get_match_fun(x, patterntype)
            pred = lambda n: not matcher(n)
    else:
        m = [m] if _is_str(m) else m
        if not x:
            matcher = get_match_fun(m, patterntype)
            pred = lambda n: matcher(n)
        else:
            x = [x] if _is_str(x) else x
            matcher_m = get_match_fun(m, patterntype)
            matcher_x = get_match_fun(x, patterntype)
            pred = lambda n: matcher_m(n) and not matcher_x(n)

    return pred


def findfolder(toppath, match='*', exclude=''):
    """
    recursively find folder path from toppath.
    patterns to decide to walk folder path or not
    :type toppath: str
    :type match: str or list(str)
    :type exclude: str or list(str)
    :rtype: generator for path str
    """
    pred = _pred_pattern(match, exclude)

    return (p for p in walkfolder(toppath, pred))


def walkfolder(toppath, pred):
    """
    walk folder if pred(foldername) is True
    :type toppath: str
    :type pred: function(str) => bool
    """
    for entry in scandir.scandir(toppath):
        if not entry.is_dir() or not pred(entry.name):
            continue
        yield entry.path
        for p in walkfolder(entry.path, pred):
            yield p


def tempdir():
    import tempfile
    d = tempfile.gettempdir()
    p = os.path.join(d, 'her_temp')
    mkdir_if_not(p, ispath=True)

    return p


def tempfolder(prefix=''):
    """임시 폴더를 만들어서 리턴"""
    import uuid

    p = prefix + str(uuid.uuid4())
    d = tempdir()
    tmpd = os.path.join(d, p)
    return mkdir_if_not(tmpd, ispath=True)


def tempfile(mode, ext='', **kwargs):
    import uuid

    d = tempdir()

    if ext and not ext.startswith('.'):
        ext = '.' + ext

    fname = os.path.join(d, str(uuid.uuid4()) + ext)
    return open(fname, mode, **kwargs)


def renderhtml(template, **kwargs):
    # from packageutil import caller_path
    from .caller import caller

    from jinja2 import Environment, FileSystemLoader
    from os.path import dirname

    if '/' not in template:
        p = dirname(caller.abspath(depth=2))
    else:
        p, template = os.path.split(template)

    j2_env = Environment(loader=FileSystemLoader(p),
                         trim_blocks=True)
    rendered = j2_env.get_template(template).render(**kwargs)
    return rendered


def renderimages(images, width=80, height=80, space=0):
    import webbrowser

    template = os.path.dirname(os.path.realpath(__file__)) + '/template/images.html'
    rendered = renderhtml(template, data=images, width=width, height=height, space=space)
    tmp = tempfile('wt', '.html')
    tmp.write(rendered)
    tmp.flush()

    webbrowser.open(tmp.name)
    return tmp.name


def imsize(fname):
    """
    return image size (height, width)
    :param fname:
    :return:
    """
    from PIL import Image
    im = Image.open(fname)
    return im.size[1], im.size[0]


def imread(fname, size=None, expand=True, dtype='float32', **kwargs):
    from skimage import io, transform
    import numpy as np

    img = io.imread(fname, **kwargs)
    image_max = np.iinfo(img.dtype).max
    if size is not None:
        sz = list(img.shape)
        sz[:len(size)] = size
        img = transform.resize(img, sz, preserve_range=True)
    if dtype.startswith('float'):
        # normalize 0 to 1
        img = img.astype(dtype) / float(image_max)
    else:
        img = img.astype(dtype)

    if expand:
        img = np.expand_dims(img, 0)
        if img.ndim == 3:
            img = np.expand_dims(img, -1)
    return img


def imsave(fname, *args, **kwargs):
    from skimage import io

    mkdir_if_not(fname)
    res = io.imsave(fname, *args, **kwargs)
    logg.info('image saved to [{}]'.format(fname))

    return res


def imread_palette(fname, expand=True, dtype='uint8', mode='r'):
    from PIL import Image

    # use png? https://pythonhosted.org/pypng/png.html ?

    img = Image.open(fname, mode=mode)
    palette = img.getpalette()
    if palette is not None:
        img = np.array(img)
        num_colors = len(palette) / 3

        image_max = float(np.iinfo(img.dtype).max)
        palette = np.array(palette).reshape(num_colors, 3) / image_max

    else:
        colors = img.convert('RGBA').getcolors()
        num_colors = len(colors)
        assert num_colors <= 256
        palette = [c[:3] for _, c in colors]
        im = Image.new('P', img.size)
        palette = np.array(palette).reshape((-1))
        im.putpalette(palette, rawmode="RGB")
        im.paste(img)
        palette = im.getpalette()
        img = np.array(im)

    img = img.astype(dtype)
    palette = palette.astype('float32')
    if expand:
        img = np.expand_dims(img, axis=0)
        palette = np.expand_dims(palette, axis=0)

    return img, palette


def filecopy(src, dst):
    import shutil
    shutil.copy(src, dst)


def download(url, out=None):
    import wget
    if out is not None:
        mkdir_if_not(out)
    logg.info('downloading... [{}] to [{}]'.format(url, out))
    f = wget.download(url, out=out)

    return f


def download_if_not(url, f):
    if not os.path.exists(f):
        f = download(url, f)
    return f


def unzip(z, member=None):
    import zipfile
    path = os.path.dirname(z)
    zip = zipfile.ZipFile(z, 'r')
    if member is None:
        zip.extractall(path)
        logg.info('unzip [{}] to [{}]'.format(z, path))
    else:
        zip.extract(member, path)
        logg.info('unzip [{}] to [{}]'.format(member, path))
    zip.close()


def untar(t, member=None):
    import tarfile
    path = os.path.dirname(t)
    tar = tarfile.open(t)
    if member is None:
        tar.extractall(path)
        logg.info('unzip [{}] to [{}]'.format(t, path))
    else:
        tar.extract(member, path)
        logg.info('unzip [{}] to [{}]'.format(member, path))
    tar.close()


def anyfile(pattern):
    import glob
    return any(glob.glob(pattern))
