from __future__ import division

import cv2
import numpy as np


def resize_image_1(img, min_side=800, max_side=1333):
    """ Resizes a numpy.ndarray of the format HWC such that the
    smallest side >= min_side and largest side <= max_side
    In the event that scaling is not possible to meet both conditions, only
    largest side <= max_side will be satisfied
    """
    rows, cols = img.shape[:2]

    smallest_side = min(rows, cols)

    # rescale the image so the smallest side is min_side
    scale = min_side / smallest_side

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_side = max(rows, cols)
    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    img = cv2.resize(img, None, fx=scale, fy=scale)

    return img, scale

def pad_to(img, target_shape):
    """ Takes an numpy.ndarray image of the format HWC and pads it to the target_shape
    The original image will be placed in the top left corner
    """

    ndim = len(img.shape)
    pad = [None] * ndim

    for i in range(ndim):
        pad[i] = (0, target_shape[i] - img.shape[i])

    return np.pad(img, pad, 'constant')
