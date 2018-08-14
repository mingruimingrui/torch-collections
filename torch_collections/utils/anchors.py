""" Script storing all anchor related functions uses tensor operations instead of numpy """
from __future__ import division

import torch


def meshgrid2d(x, y):
    xx = x.repeat(len(y), 1)
    yy = y.repeat(len(x), 1).permute(1, 0)
    return xx, yy


def compute_overlap(a, b):
    """
    Parameters
    ----------
    a: (N, 4) Tensor
    b: (K, 4) Tensor
    Returns
    -------
    overlaps: (N, K) Tensor of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(a[:, 2:3], b[:, 2]) - torch.max(a[:, 0:1], b[:, 0])
    ih = torch.min(a[:, 3:4], b[:, 3]) - torch.max(a[:, 1:2], b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = (a[:, 2:3] - a[:, 0:1]) * (a[:, 3:4] - a[:, 1:2]) + area - iw * ih
    ua = torch.clamp(ua, min=1e-15)

    intersection = iw * ih

    return intersection / ua


def generate_anchors_at_window(
    base_size=16,
    ratios=[0.5, 1., 2.],
    scales=[2. ** 0., 2. ** (1. / 3.), 2. ** (2. / 3.)]
):
    """ Generate anchors based on a size a set of ratios and scales
    w.r.t a reference window
    """
    # if not isinstance(base_size, torch.Tensor):
    #     base_size = torch.Tensor([base_size]).reshape(1)
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
    anchors[:, 3] = anchors[:, 2].clone() * repeated_ratios

    # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
    anchors[:, 0::2] = anchors[:, 0::2].clone() - anchors[:, 2:3].clone() / 2
    anchors[:, 1::2] = anchors[:, 1::2].clone() - anchors[:, 3:4].clone() / 2

    return anchors


def shift_anchors(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size """
    shift_x = torch.arange(0 + 0.5, shape[1] + 0.5, step=1) * stride
    shift_y = torch.arange(0 + 0.5, shape[0] + 0.5, step=1) * stride
    if anchors.is_cuda:
        device_idx = torch.cuda.device_of(anchors).idx
        shift_x = shift_x.cuda(device_idx)
        shift_y = shift_y.cuda(device_idx)

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

    targets = torch.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2), dim=1)
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

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std + mean) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std + mean) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std + mean) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std + mean) * height

    pred_boxes = torch.stack([x1, y1, x2, y2], dim=2)

    return pred_boxes


def anchor_targets_bbox(
    anchors,
    annotations,
    num_classes,
    mask_shape=None,
    negative_overlap=0.4,
    positive_overlap=0.5
):
    """ Generate anchor targets for bbox detection.
    Args
        anchors: torch.Tensor of annotations of shape (N, 4) for (x1, y1, x2, y2).
        annotations: torch.Tensor of shape (N, 5) for (x1, y1, x2, y2, label).
        num_classes: Number of classes to predict.
        mask_shape: If the image is padded with zeros, mask_shape can be used to mark the relevant part of the image.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).
    Returns
        labels: torch.Tensor of shape (A, num_classes) where a row consists of 0 for negative and 1 for positive for a certain class.
        annotations: torch.Tensor of shape (A, 5) for (x1, y1, x2, y2, label) containing the annotations corresponding to each anchor or 0 if there is no corresponding anchor.
        anchor_states: torch.Tensor of shape (N,) containing the state of an anchor (-1 for ignore, 0 for bg, 1 for fg).
    """
    # anchor states: 1 is positive, 0 is negative, -1 is dont care
    anchor_states = torch.zeros_like(anchors[:, 0])
    labels        = torch.stack([anchor_states] * num_classes, dim=1)

    if annotations.shape[0] > 0:
        # obtain indices of gt annotations with the greatest overlap
        overlaps             = compute_overlap(anchors, annotations)
        argmax_overlaps_inds = torch.argmax(overlaps, dim=1)
        max_overlaps         = overlaps[range(overlaps.shape[0]), argmax_overlaps_inds]

        # assign "dont care" labels
        positive_indices                = max_overlaps >= positive_overlap
        ignore_indices                  = (max_overlaps > negative_overlap) & ~positive_indices
        anchor_states[ignore_indices]   = -1
        anchor_states[positive_indices] = 1

        # compute box regression targets
        annotations = annotations[argmax_overlaps_inds]

        # compute target class labels
        if torch.sum(positive_indices) > 0:
            labels[positive_indices] = labels[positive_indices].scatter(
                1, annotations[positive_indices, 4:5].long(), 1
            )
    else:
        annotations = torch.stack([anchor_states] * 5, dim=1)

    # ignore annotations outside of image
    if mask_shape is not None:
        anchors_centers        = (anchors[:, :2] + anchors[:, 2:]) / 2
        indices                = (anchors_centers[:, 0] >= mask_shape[-1]) | (anchors_centers[:, 1] >= mask_shape[-2])
        anchor_states[indices] = -1

    return labels, annotations, anchor_states


def box_nms(bboxes, scores, threshold=0.5, mode='union'):
    '''Non maximum suppression.
    Args
        bboxes: (tensor) bounding boxes, sized [N,4].
        scores: (tensor) bbox scores, sized [N,].
        threshold: (float) overlap threshold.
        mode: (str) 'union' or 'min'.
    Returns
        keep: (tensor) selected indices.
    Reference
        https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
        https://github.com/kuangliu/pytorch-retinanet/blob/master/utils.py
    '''
    x1 = bboxes[:,0]
    y1 = bboxes[:,1]
    x2 = bboxes[:,2]
    y2 = bboxes[:,3]

    areas = (x2-x1) * (y2-y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        if not len(order.shape):
            print('WTF')
            break

        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        ovr = compute_overlap(bboxes[order[0]:order[0]+1], bboxes[order[1:]])[0]
        ids = (ovr<=threshold).nonzero().squeeze()
        order = order[ids + 1]

    return torch.LongTensor(keep)
