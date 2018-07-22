""" Script storing all anchor related functions uses tensor operations instead of numpy """

import torch


def meshgrid2d(x, y):
    xx = x.repeat(len(y), 1)
    yy = y.repeat(len(x), 1).permute(1, 0)
    return xx, yy


def generate_anchors_at_window(
    base_size=16,
    ratios=[0.5, 1., 2.],
    scales=[2. ** 0., 2. ** (1. / 3.), 2. ** (2. / 3.)]
):
    """ Generate anchors based on a size a set of ratios and scales
    w.r.t a reference window
    """
    if not isinstance(base_size, torch.Tensor):
        base_size = torch.Tensor([base_size]).reshape(1)
    if not isinstance(ratios, torch.Tensor):
        ratios = torch.Tensor(ratios)
    if not isinstance(scales, torch.Tensor):
        scales = torch.Tensor(scales)

    num_anchors = len(ratios) * len(scales)
    tiled_scales = scales.repeat(3)
    repeated_ratios = torch.stack([ratios] * 3).transpose(0, 1).reshape(-1)

    # initialize output anchors
    anchors = torch.zeros(num_anchors, 4)
    anchors[:, 2] = base_size * tiled_scales
    anchors[:, 3] = base_size * tiled_scales

    # compute areas of anchors
    areas = anchors[:, 2] * anchors[:, 3]

    # correct for ratios
    anchors[:, 2] = torch.sqrt(areas / repeated_ratios)
    anchors[:, 3] = anchors[:, 2] * repeated_ratios

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] -= anchors[:, 2:3] / 2
    anchors[:, 1::2] -= anchors[:, 3:4] / 2

    return anchors


def shift_anchors(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size """
    shift_x = torch.arange(0 + 0.5, shape[1] + 0.5, step=1) * stride
    shift_y = torch.arange(0 + 0.5, shape[0] + 0.5, step=1) * stride

    shift_x, shift_y = meshgrid2d(shift_x, shift_y)

    shifts = torch.stack([
        shift_x.reshape(-1), shift_y.reshape(-1),
        shift_x.reshape(-1), shift_y.reshape(-1)
    ], dim=1)

    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = anchors.reshape(1, A, 4) + shifts.reshape(1, K, 4).permute(1, 0, 2)
    all_anchors = all_anchors.reshape(K * A, 4)

    return all_anchors


def bbox_transform(anchors, gt_boxes, mean, std):
    """ Compute bounding-box regression targets for an image """
    anchor_widths  = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

    import pdb; pdb.set_trace()

    targets = torch.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    targets = targets.T

    targets = (targets - mean) / std

    return targets


def bbox_transform_inv(boxes, deltas, mean, std):
    """ Applies deltas (usually regression results) to boxes (usually anchors).
    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.
    Args
        boxes : torch.Tensor of shape (B, N, 4), where B is the batch size, N the number of boxes and 4 values for (x1, y1, x2, y2).
        deltas: torch.Tensor of same shape as boxes. These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).
    Returns
        A torch.Tensor of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    width  = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = torch.stack([x1, y1, x2, y2], dim=2)

    return pred_boxes
