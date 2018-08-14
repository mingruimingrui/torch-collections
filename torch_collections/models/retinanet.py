""" Retinanet implementation in torch """

import torch

# Default configs
from ._retinanet_configs import make_configs

# Backbone loader functions
from ._backbone import (
    build_backbone_model,
    get_backbone_channel_sizes,
    build_fpn_feature_shape_fn
)

# RetinaNet submodels
from ._retinanet import (
    FeaturePyramidSubmodel,
    DynamicRegressionModel,
    DynamicClassificationModel,
    ComputeAnchors,
    RetinaNetLoss
)

# Common detection submodules
from ..modules import RegressBoxes, ClipBoxes, FilterDetections


class RetinaNet(torch.nn.Module):
    def __init__(self, num_classes, **kwargs):
        """ Constructs a RetinaNet model
        Args
            Refer to torch_collections.models._retinanet_configs.py
        Returns
            A RetinaNet model whos inputs/outputs are as follows

            Training
                - Input of this model is a dict of the following format
                    {
                        'images'      : Tensor of images in the format NCHW with torch standard preprocessing,
                        'annotations' : list of tensors representing annotations
                                        each annotation tensor is of the shape [n_annotations, 5]
                                        the columns are [x1, y1, x2, y2, class_id]
                    }
                - Output of this model is a single tensor for loss

            Eval
                - Input of this model is a dict of the following format
                    {'images'      : Tensor of images in the format NCHW with torch standard preprocessing}
                - Output of this model is a list of detections, each detection is a dictionary of the format
                    {
                        'boxes'  : (num_detections, 4) shaped tensor for [x1, y1, x2, y2] of all detections,
                        'scores' : (num_detections) shaped tensor for scores of all detections,
                        'labels' : (num_detections) shaped tensor for labels of all detections
                    }
        """
        super(RetinaNet, self).__init__()

        # Make config
        kwargs['num_classes'] = num_classes
        self.configs = make_configs(**kwargs)

        # Make helper functions and modules
        self.fpn_feature_shape_fn = build_fpn_feature_shape_fn(self.configs['backbone'])
        self.build_modules()

    def build_modules(self):
        """ Build all modules and submodels for RetinaNet """
        # Make backbone model
        self.backbone_model = build_backbone_model(self.configs['backbone'])
        self.feature_pyramid_submodel = FeaturePyramidSubmodel(
            backbone_channel_sizes=get_backbone_channel_sizes(self.configs['backbone']),
            feature_size=self.configs['pyramid_feature_size']
        )

        # Make regression and classification models
        self.regression_submodel = DynamicRegressionModel(
            self.configs['num_anchors'],
            pyramid_feature_size=self.configs['pyramid_feature_size'],
            regression_feature_size=self.configs['regression_feature_size'],
            growth_rate=self.configs['regression_growth_rate'],
            num_layers=self.configs['regression_num_layers'],
            block_type=self.configs['regression_block_type']
        )
        self.classification_submodel = DynamicClassificationModel(
            self.configs['num_classes'],
            self.configs['num_anchors'],
            pyramid_feature_size=self.configs['pyramid_feature_size'],
            classification_feature_size=self.configs['classification_feature_size'],
            growth_rate=self.configs['classification_growth_rate'],
            num_layers=self.configs['classification_num_layers'],
            block_type=self.configs['classification_block_type']
        )

        # Create function to compute anchors based on feature shapes
        self.compute_anchors = ComputeAnchors(
            sizes=self.configs['anchor_sizes'],
            strides=self.configs['anchor_strides'],
            ratios=self.configs['anchor_ratios'],
            scales=self.configs['anchor_scales']
        )

        # Create loss function
        self.loss_fn = RetinaNetLoss(
            num_classes=self.configs['num_classes'],
            fpn_feature_shape_fn=self.fpn_feature_shape_fn,
            compute_anchors=self.compute_anchors
        )

        # Create functions to inverse the bbox transform
        self.regress_boxes = RegressBoxes()
        self.clip_boxes    = ClipBoxes()

        # Create funciton to apply NMS
        self.filter_detections = FilterDetections(
            nms_threshold=self.configs['nms_threshold'],
            score_threshold=self.configs['score_threshold'],
            max_detections=self.configs['max_detections']
        )

    def forward(self, image, annotations=None):
        if self.training:
            assert annotations is not None

        # Calculate features
        C3, C4, C5 =  self.backbone_model(image)[-3:]
        features = self.feature_pyramid_submodel(C3, C4, C5)

        # Apply regression and classification submodels on each feature
        regression_outputs     = [self.regression_submodel(f)     for f in features]
        classification_outputs = [self.classification_submodel(f) for f in features]

        # Concat outputs
        regression     = torch.cat(regression_outputs    , 1)
        classification = torch.cat(classification_outputs, 1)

        # Train on regression and classification
        if self.training:
            return self.loss_fn(regression, classification, image, annotations)

        # Collect batch information
        feature_shapes = self.fpn_feature_shape_fn(image.shape)[-5:]

        # Compute base anchors
        anchors = self.compute_anchors(image.shape[0], feature_shapes)

        # Apply predicted regression to anchors
        boxes = self.regress_boxes(anchors, regression)
        boxes = self.clip_boxes(image.shape, boxes)

        detections = self.filter_detections(boxes, classification)

        return detections
