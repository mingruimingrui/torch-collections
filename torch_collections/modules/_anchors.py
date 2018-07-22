import torch
from ..utils import anchors as utils_anchors


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
        self.anchors = utils_anchors.generate_anchors_at_window(
            base_size=size,
            ratios=ratios,
            scales=scales,
        )

    def forward(self, batch_size, feature_shape):
        # x.shape       = [-1, C, H, W]
        # anchors.shape = [H*W*num_anchors, 4]
        anchors = utils_anchors.shift_anchors(feature_shape, self.stride, self.anchors)
        all_anchors = anchors.repeat(batch_size, 1, 1)
        return all_anchors
