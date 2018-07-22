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
    """ Produce shifted anchors based on shape of the map and stride size
    This is for ordinary np.array
    """
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


class Anchors(torch.nn.Module):
    def __init__(
        self,
        size,
        stride,
        ratios=[0.5, 1., 2.],
        scales=[2. ** 0., 2. ** (1. / 3.), 2. ** (2. / 3.)]
    ):
        super(Anchors, self).__init__()
        self.stride = stride
        self.num_anchors = len(ratios) * len(scales)
        self.anchors = generate_anchors_at_window(
            base_size=size,
            ratios=ratios,
            scales=scales,
        )

    def forward(self, batch_size, feature_shape):
        # x.shape       = [-1, C, H, W]
        # anchors.shape = [H*W*num_anchors, 4]
        anchors = shift_anchors(feature_shape, self.stride, self.anchors)
        all_anchors = anchors.repeat(batch_size, 1, 1)
        return all_anchors
