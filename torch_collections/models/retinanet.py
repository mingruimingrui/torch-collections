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
    DefaultRegressionModel,
    DefaultClassificationModel,
    ComputeAnchors,
    RetinaNetLoss,
    build_collate_container
)

# Common detection submodules
from ..modules import RegressBoxes, ClipBoxes, FilterDetections


class RetinaNet(torch.nn.Module):
    def __init__(self, num_classes, **kwargs):
        """ Construct a RetinaNet model for training

        Args
            Refer to torch_collections.models._retinanet_configs.py
        Returns
            training_model : a RetinaNet training model
                - Outputs of this model are [anchor_regressions, anchor_classifications]
                - Shapes would be [(batch_size, num_anchors, 4), (batch_size, num_anchors, num_classes)]
        """
        super(RetinaNet, self).__init__()

        # Make config file
        kwargs['num_classes'] = num_classes
        self.configs = make_configs(**kwargs)

        # Make helper functions and variables
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
        self.regression_submodel = DefaultRegressionModel(
            self.configs['num_anchors'],
            pyramid_feature_size=self.configs['pyramid_feature_size'],
            regression_feature_size=self.configs['regression_feature_size']
        )
        self.classification_submodel = DefaultClassificationModel(
            self.configs['num_classes'],
            self.configs['num_anchors'],
            pyramid_feature_size=self.configs['pyramid_feature_size'],
            classification_feature_size=self.configs['classification_feature_size']
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
        self.filter_detections = FilterDetections()

    def forward(self, batch):
        # Calculate features
        C3, C4, C5 =  self.backbone_model(batch['image'])
        features = self.feature_pyramid_submodel(C3, C4, C5)

        # Apply regression and classificatio submodels on each feature
        regression_outputs     = [self.regression_submodel(f)     for f in features]
        classification_outputs = [self.classification_submodel(f) for f in features]

        # Concat outputs
        regression     = torch.cat(regression_outputs    , 1)
        classification = torch.cat(classification_outputs, 1)

        # Train on regression and classification
        if self.training:
            outputs = {
                'regression'     : regression,
                'classification' : classification
            }
            return self.loss_fn(outputs, batch)

        # Collect batch information
        feature_shapes = self.fpn_feature_shape_fn(batch['image'].shape)

        # Compute base anchors
        anchors = self.compute_anchors(batch['image'].shape[0], feature_shapes)

        # Apply predicted regression to anchors
        boxes = self.regress_boxes(anchors, regression)
        boxes = self.clip_boxes(batch['image'].shape, boxes)

        detections = self.filter_detections(boxes, classification)

        return detections

    ###########################################################################
    #### Start of collate_fn

    build_collate_container = build_collate_container
