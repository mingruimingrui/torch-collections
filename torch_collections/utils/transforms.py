from __future__ import division

import math
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


def pad_img_to(img, target_hw, location='upper-left', mode='constant'):
    """ Takes an numpy.ndarray image of the format HWC and pads it to the target_hw

    Args
        img       : numpy.ndarray image of the format HWC or HW
        target_hw : target height width (list-like of size 2)
        location  : location of original image after padding, option of 'upper-left' and 'center'
        mode      : mode of padding in https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.pad.html
    Returns
        padded image

    The original image will be placed in the top left corner
    """
    if len(img.shape) == 3:
        pad = [None, None, (0, 0)]
    else:
        pad = [None, None]

    if location == 'upper-left':
        for i in range(2):
            pad[i] = (0, target_hw[i] - img.shape[i])

    elif location == 'center':
        for i in range(2):
            excess = target_hw[i] - img.shape[i]
            x1 = math.ceil(excess / 2)
            x2 = excess - x1
            pad[i] = (x1, x2)

    else:
        raise ValueError('{} is not a valid location argument'.format(location))

    return np.pad(img, pad, mode=mode)
