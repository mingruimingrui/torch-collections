import torch
from ..utils import anchors as utils_anchors


class RegressBoxes(torch.nn.Module):
    def __init__(self, mean=0.0, std=0.2):
        super(RegressBoxes, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, anchors, regression):
        return utils_anchors.bbox_transform_inv(anchors, regression, self.mean, self.std)
