import torch
from ..utils import anchors as utils_anchors


class FilterDetections(torch.nn.Module):
    def __init__(
        self,
        nms_threshold   = 0.5,
        score_threshold = 0.05,
        max_detections  = 300,
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.
        Args
            nms_threshold   : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold : Threshold used to prefilter the boxes with.
            max_detections  : Maximum number of detections to keep.
        """
        super(FilterDetections, self).__init__()
        self.nms_threshold   = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections

    def forward(self, image_shape, boxes):
        import pdb; pdb.set_trace()
        x1 = torch.clamp(boxes[:, :, 0], 0, image_shape[-1])
        y1 = torch.clamp(boxes[:, :, 1], 0, image_shape[-2])
        x2 = torch.clamp(boxes[:, :, 2], 0, image_shape[-1])
        y2 = torch.clamp(boxes[:, :, 3], 0, image_shape[-2])

        return torch.stack([x1, y1, x2, y2], dim=2)
