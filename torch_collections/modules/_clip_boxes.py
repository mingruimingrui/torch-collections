import torch
from ..utils import anchors as utils_anchors


class ClipBoxes(torch.nn.Module):
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, image_shape, boxes):
        x1 = torch.clamp(boxes[:, :, 0], 0, image_shape[-1])
        y1 = torch.clamp(boxes[:, :, 1], 0, image_shape[-2])
        x2 = torch.clamp(boxes[:, :, 2], 0, image_shape[-1])
        y2 = torch.clamp(boxes[:, :, 3], 0, image_shape[-2])

        return torch.stack([x1, y1, x2, y2], dim=2)
