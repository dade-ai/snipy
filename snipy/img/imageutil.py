import numpy as np
from snipy import irandom as rand
from snipy.basic import tuple_args
# from snipy.concurrent import SharedPool

# todo imread with resize ..
# todo : add some examples

# region image augmentation oriented
# assume : images [B,H,W,C] formats


# region rand util


def rand_apply_batch(fun, imagez, **kwargs):
    assert isinstance(imagez, (tuple, list))
    # print imagez[0].shape, imagez[1].shape
    # print len(imagez[0])
    mask = np.random.choice([False, True], len(imagez[0]), p=kwargs.pop('p', None))
    # print 'masks', mask

    def apply(images):
        data = fun(images[mask], **kwargs)
        images[mask] = data
        return images

    return tuple(map(apply, imagez)) if np.any(mask) else imagez


def rand_apply_onebatch(fun, imagez, **kwargs):
    assert isinstance(imagez, (tuple, list))
    mask = np.random.choice([False, True], len(imagez[0]), p=kwargs.pop('p', None))
    # if mask is True, apply function to tuple of images. ex: [image, labelimage]

    def apply(images):
        images[mask] = [fun(d, **kwargs) for d in images[mask]]
        return images

    return tuple(map(apply, imagez)) if np.any(mask) else imagez

# endregion

# region random augmentation


def rand_augmentations(level=3):
    funs = (rand_fliplr, rand_flipud, rand_rot90,
            rand_brightness, rand_intensity, rand_elastic,)
    return funs[:level]


def rand_fliplr(*imagez):
    """ flip together """
    return rand_apply_onebatch(np.fliplr, imagez)


def rand_flipud(*imagez):
    """ flip together """
    return rand_apply_onebatch(np.flipud, imagez)


def rand_rot90(*imagez):
    """ rotate together """
    return rand_apply_onebatch(np.rot90, imagez)


def rand_crop(sz, *imagez):
    """
    random crop
    # assume imagez has same size (H, W)
    # assume sz is less or equal than size of image
    :param sz: cropped image sz
    :param imagez: imagez
    :return: rand cropped image pairs or function bound to sz
    """

    def _rand_crop(*imgz):
        imsz = imgz[0].shape[:2]

        assert imsz[0] >= sz[0] and imsz[1] >= sz[1]

        si = np.random.randint(imsz[0] - sz[0]) if imsz[0] > sz[0] else 0
        sj = np.random.randint(imsz[1] - sz[1]) if imsz[1] > sz[1] else 0

        slicei = slice(si, si+sz[0])
        slicej = slice(sj, sj+sz[1])

        outs = tuple(img[slicei, slicej] for img in imgz)
        return tuple_or_not(*outs)

    return _rand_crop(*imagez) if imagez else _rand_crop


def tuple_or_not(*args):
    return args[0] if len(args) == 1 else args


def rand_rotate(anglerange, *imagez):
    """
    :param anglerange:
    :param imagez:
    :return:
    """
    r = float(anglerange[1] - anglerange[0])
    s = anglerange[0]

    def _rand_rotate(*imgz):
        angle = np.random.random(1)[0] * r + s
        out = tuple(rotate(img, angle) for img in imgz)
        return tuple_or_not(out)

    return _rand_rotate(*imagez) if imagez else _rand_rotate


def rand_elastic(*imagez, **kwargs):
    return rand_apply_onebatch(elastic_transform, imagez, **kwargs)


def rand_blend(images):
    # make blending mask
    # apply to imagez
    inoise = rand_blend_mask(images.shape)
    discrete = np.round(inoise * images.shape[3])
    return blend_discrete(images, discrete)


def blend_discrete(images, depthmask, depth=None):
    """
    depthmask : shape of [batch, h, w]
    """
    imshape = images.shape
    depth = depth or images.shape[3]
    blend = np.empty(shape=(imshape[0], imshape[1], imshape[2]))
    for d in range(depth):
        imask = (depthmask == d)
        channel = images[..., d]
        blend[imask] = channel[imask]
    return np.expand_dims(blend, axis=-1)


def rand_blend_mask(shape, rand=rand.uniform(-10, 10), **kwargs):
    """ random blending masks """
    # batch, channel = shape[0], shape[3]
    z = rand(shape[0])  # seed
    noise = snoise2dz((shape[1], shape[2]), z, **kwargs)

    return noise  # np.round(noise * channel)


def snoise2dvec(size, *params, **kwargs):  #, vlacunarity):
    """
    vector parameters
    :param size:
    :param vz:
    :param vscale:
    :param voctave:
    :param vpersistence:
    :param vlacunarity:
    :return:
    """
    data = (snoise2d(size, *p, **kwargs) for p in zip(*params))  # , vlacunarity))
    return np.stack(data, 0)


def snoise2d(size, z=0.0, scale=0.05, octaves=1, persistence=0.25, lacunarity=2.0):
    """
    z value as like a seed
    """
    import noise
    data = np.empty(size, dtype='float32')
    for y in range(size[0]):
        for x in range(size[1]):
            v = noise.snoise3(x * scale, y * scale, z,
                              octaves=octaves, persistence=persistence, lacunarity=lacunarity)
            data[x, y] = v
    data = data * 0.5 + 0.5
    if __debug__:
        assert data.min() >= 0. and data.max() <= 1.0
    return data


def snoise2dz(size, z, scale=0.05, octaves=1, persistence=0.25, lacunarity=2.0):
    """
    z as seeds
    scale이 작을 수록 패턴이 커지는 효과
    """
    import noise
    z_l = len(z)

    data = np.empty((z_l, size[0], size[1]), dtype='float32')
    for iz in range(z_l):
        zvalue = z[iz]
        for y in range(size[0]):
            for x in range(size[1]):
                v = noise.snoise3(x * scale, y * scale, zvalue,
                                  octaves=octaves, persistence=persistence, lacunarity=lacunarity)
                data[iz, y, x] = v
    data = data * 0.5 + 0.5
    if __debug__:
        assert data.min() >= 0. and data.max() <= 1.0
    return data


@tuple_args
def rand_brightness(imagez, scale=1.0, randfun=rand.normal(0., .1), clamp=(0., 1.)):
    """
    :param images:
    :param scale: scale for random value
    :param randfun: any randfun binding except shape
    :param clamp: clamping range
    :return:
    """
    l, h = clamp
    r = randfun((imagez[0].shape[0], 1, 1, 1)) * scale

    def apply(im):
        im += r
        im[im < l] = l
        im[im > h] = h
        return im

    return tuple(map(apply, imagez))


import warnings
np.seterr(all='raise')

@tuple_args
def rand_intensity(imagez, scale=1.0, randfun=rand.normal(1., 0.1), clamp=(0., 1.)):

    l, h = clamp
    # c = (l + h) / 2.
    bshape = (imagez[0].shape[0], 1, 1, 1)
    phi = randfun(bshape) * scale
    theta = randfun(bshape) * scale
    p = randfun(bshape) * scale

    def apply(im):
        # (h / phi) * (im/(h/theta))**p
        with warnings.catch_warnings():
            warnings.filterwarnings('error')
            try:
                im2 = (h * phi) * np.power(im * theta / h, p)
                im2[im2 < l] = l
                im2[im2 > h] = h
            except Exception as e:
                # logg.warn('in rand_intensity: [%s], pass' % e)
                return im
        # im *= r
        return im2

    return tuple(map(apply, imagez))


# endregion

# region image transform implementations
def elastic_transform(im, alpha=0.5, sigma=0.2, affine_sigma=1.):
    """
    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    elastic deformation of images as described in [Simard2003]
    """
    # fixme : not implemented for multi channel !
    import cv2

    islist = isinstance(im, (tuple, list))
    ima = im[0] if islist else im

    # image shape
    shape = ima.shape
    shape_size = shape[:2]

    # Random affine transform
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-affine_sigma, affine_sigma, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    if islist:
        res = []
        for i, ima in enumerate(im):
            if i == 0:
                res.append(cv2.warpAffine(ima, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101))
            else:
                res.append(cv2.warpAffine(ima, M, shape_size[::-1]))
        im = res
    else:
        ima = cv2.warpAffine(ima, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        # ima = cv2.warpAffine(ima, M, shape_size[::-1])

    # fast gaussian filter
    blur_size = int(4 * sigma) | 1
    dx = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(*shape) * 2 - 1), ksize=(blur_size, blur_size), sigmaX=sigma) * alpha

    # remap
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x, map_y = (y + dy).astype('float32'), (x + dx).astype('float32')

    def remap(data):
        r = cv2.remap(data, map_y, map_x, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return r[..., np.newaxis]

    if islist:
        return tuple([remap(ima) for ima in im])
    else:
        return remap(ima)

# endregion


def rotate_crop(centerij, sz, angle, img=None, mode='constant', **kwargs):
    """
    rotate and crop
    if no img, then return crop function
    :param centerij:
    :param sz:
    :param angle:
    :param img: [h,w,d]
    :param mode: padding option
    :return: cropped image or function
    """
    # crop enough size ( 2 * sqrt(sum(sz^2) )
    # rotate
    from skimage import transform
    sz = np.array(sz)
    crop_half = int(np.ceil(np.sqrt(np.square(sz).sum())))

    if centerij[0] >= crop_half or centerij[1] >= crop_half:
        raise NotImplementedError

    slicei = slice(centerij[0] - crop_half, centerij[0] + crop_half)
    slicej = slice(centerij[1] - crop_half, centerij[1] + crop_half)

    # slicei = (centerij[0] - crop_half, centerij[0] + crop_half)
    # slicej = (centerij[1] - crop_half, centerij[1] + crop_half)

    # def _pad_if_need(im):
    #     imshape = im.shape
    #     pad_need = slicei[0] < 0 or slicej[0] < 0 or slice
    #     padwidth = [(slicei[0], np.maximum(0, slicei[1] - imshape[0])),
    #                 (slicej[0], np.maximum(0, slicej[1] - imshape[1]))]

    def _rotate_cropcenter(im):
        enoughcrop = im[slicei, slicej]

        rotated = transform.rotate(enoughcrop, angle, resize=False, preserve_range=True, mode=mode, **kwargs)
        return cropcenter(sz, rotated)
    if img is not None:
        return _rotate_cropcenter(img)

    return _rotate_cropcenter


def rotate(img, angle, resize=False, preserve_range=True, mode='constant', **kwargs):
    from skimage import transform
    return transform.rotate(img, angle, resize=resize, preserve_range=preserve_range, mode=mode, **kwargs)


def crop(img, center, sz, mode='constant'):
    """
    crop sz from ij as center
    :param img:
    :param center: ij
    :param sz:
    :param mode:
    :return:
    """
    center = np.array(center)
    sz = np.array(sz)
    istart = (center - sz / 2.).astype('int32')
    iend = istart + sz
    imsz = img.shape[:2]
    if np.any(istart < 0) or np.any(iend > imsz):
        # padding
        padwidth = [(np.minimum(0, istart[0]), np.maximum(0, iend[0]-imsz[0])),
                    (np.minimum(0, istart[1]), np.maximum(0, iend[1]-imsz[1]))]
        padwidth += [(0, 0)] * (len(img.shape) - 2)
        img = np.pad(img, padwidth, mode=mode)
        istart = (np.maximum(0, istart[0]), np.maximum(0, istart[1]))

        return img[istart[0]:istart[0]+sz[0], istart[1]:istart[1]+sz[1]]

    return img[istart[0]:iend[0], istart[1]:iend[1]]


def cropcenter(sz, img=None):
    """
    if no img, then return crop function
    :param sz:
    :param img:
    :return:
    """
    l = len(sz)
    sz = np.array(sz)

    def wrapped(im):
        imsz = np.array(im.shape)
        s = (imsz[:l] - sz) / 2  # start index
        to = s + sz  # end index

        # img[s[0]:to[0], ... s[end]:to[end], ...]
        slices = [slice(s, e) for s, e in zip(s, to)]

        return im[slices]

    if img is not None:
        return wrapped(img)

    return wrapped


def cropcenter_batch(sz, img=None):
    return cropcenter_dim(sz, start_axis=1, img=img)


def cropcenter_dim(sz, start_axis=0, img=None):
    l = len(sz)
    sz = np.array(sz)

    def wrapped(im):
        imsz = np.array(im.shape)
        s = (imsz[start_axis:l+start_axis] - sz) / 2
        to = s + sz
        # img[:, s[0]:to[0], ... s[end]:to[end], ...]
        slices = [slice(None)] * start_axis + [slice(s, e) for s, e in zip(s, to)]
        return im[slices]

    if img is not None:
        return wrapped(img)

    return wrapped


def pad_if_need(sz_atleast, img, mode='constant'):
    # fixme : function or ....
    """
    pad img if need to guarantee minumum size
    :param sz_atleast: [H,W] at least
    :param img: image np.array [H,W, ...]
    :param mode: str, padding mode
    :return: padded image or asis if enought size
    """
    # sz_atleast = np.asarray(sz_atleast)
    imsz = img.shape[:2]  # assume img [H,W, ...]
    padneed = np.asarray((sz_atleast[0] - imsz[0], sz_atleast[1] - imsz[1]))
    if np.any(padneed > 0):
        # need padding
        padding = np.zeros((img.ndim, 2), dtype='int16')
        padneed = np.maximum(padneed, 0)
        padding[:2, 0] = padneed/2
        padding[:2, 1] = padneed - padneed/2
        img = np.pad(img, padding, mode=mode)

    return img


def canny(img, threshold1=255/3, threshold2=255, **kwargs):
    """ canny edge """
    import cv2
    # edges=None, apertureSize=None, L2gradient=None
    if img.ndim <= 3:
        edge = cv2.Canny(img, threshold1, threshold2, **kwargs)
        if edge.ndim == 2:
            edge = np.expand_dims(edge, 2)
    elif img.ndim == 4:
        # batch
        edge = np.asarray([cv2.Canny(i, threshold1, threshold2, **kwargs) for i in img])
        if edge.ndim == 3:
            edge = np.expand_dims(edge, 3)
    else:
        raise ValueError('above 5d?')
    return edge


def _convert_uint8(im):
    if im.dtype != np.uint8:
        im = np.uint8(im * 255)
    return im


@tuple_args
def alpha_composite(images):
    from PIL import Image

    # images

    # backward
    # images = reversed(images)
    # out = Image
    out = Image.fromarray(_convert_uint8(images[0]))
    for img in images[1:]:
        img = Image.fromarray(_convert_uint8(img))
        # check same size?
        # Image.alpha_composite(bg, fg)
        # alpha composite im2 over im1
        out = Image.alpha_composite(img, out)

    return np.asarray(out, dtype='float32') / 255.

