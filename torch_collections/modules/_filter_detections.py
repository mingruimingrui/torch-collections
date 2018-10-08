from operator import itemgetter

import torch
import numpy as np

from ..utils import anchors as utils_anchors
from ..utils import cpu_nms as utils_nms


class FilterDetections(torch.nn.Module):
    def __init__(
        self,
        nms_threshold=0.5,
        score_threshold=0.05,
        max_detections=300,
        apply_nms=True,
        type='hard',
        use_cpu=False
    ):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.
        Args
            nms_threshold   : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold : Threshold used to prefilter the boxes with.
            max_detections  : Maximum number of detections to keep.
            apply_nms       : Flag to raise if NMS should be applied (will ignore nms_threshold and max_detections)
            type            : Option of ['hard', 'soft'], hard is the traditional nms, soft implements https://arxiv.org/abs/1704.04503
            use_cpu         : Performs nms on CPU if True
        """
        assert type in ['soft', 'hard'], "type must be in ['hard', 'soft'], got {}".format(type)
        super(FilterDetections, self).__init__()
        self.nms_threshold   = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.apply_nms       = apply_nms
        self.type            = type
        self.use_cpu         = use_cpu

    def forward(self, boxes_batch, classification_batch):
        all_results = []

        for boxes, classification in zip(boxes_batch, classification_batch):
            # Perform per class filtering
            all_keep = []
            for c in range(classification.shape[1]):
                scores = classification[:, c]
                indices_keep = scores > self.score_threshold

                if not torch.sum(indices_keep):
                    continue

                filtered_boxes  = boxes[indices_keep]
                filtered_scores = scores[indices_keep]

                if self.apply_nms:
                    if self.use_cpu:
                        filtered_dets = torch.cat((filtered_boxes, filtered_scores.reshape(-1, 1)), dim=1)
                        filtered_dets = filtered_dets.cpu().data.numpy()
                        if self.type == 'soft':
                            nms_indices = utils_nms.cpu_soft_nms(filtered_dets)
                        else:
                            nms_indices = utils_nms.cpu_nms(filtered_dets, thresh=self.nms_threshold)
                            import pdb; pdb.set_trace()
                    else:
                        nms_indices = utils_anchors.box_nms(
                            filtered_boxes,
                            filtered_scores,
                            threshold=self.nms_threshold
                        )
                else:
                    _, nms_indices = filtered_scores.sort(dim=0, descending=True)

                nms_indices = nms_indices[:self.max_detections]

                for index in nms_indices:
                    all_keep.append({
                        'box'  : filtered_boxes[index],
                        'score': filtered_scores[index],
                        'label': torch.IntTensor([c])[0]
                    })

            if not len(all_keep):
                all_results.append({
                    'boxes' : torch.Tensor(),
                    'scores': torch.Tensor(),
                    'labels': torch.Tensor()
                })
                continue

            # Select the top detections
            all_keep.sort(key=itemgetter('score'))
            all_keep = all_keep[::-1][:self.max_detections]

            # Gather into arrays
            boxes  = torch.stack([k['box']   for k in all_keep], dim=0)
            scores = torch.stack([k['score'] for k in all_keep], dim=0)
            labels = torch.stack([k['label'] for k in all_keep], dim=0)

            # Gather into result
            all_results.append({
                'boxes' : boxes,
                'scores': scores,
                'labels': labels
            })

        return all_results
