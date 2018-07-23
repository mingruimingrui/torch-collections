""" Retinanet implementation in torch """

import torch
import torchvision

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
    collate_fn
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
        self.to_tensor = torchvision.transforms.ToTensor()
        self.normalize = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        self.regression_mean = torch.Tensor([0, 0, 0, 0])
        self.regression_std  = torch.Tensor([0.2, 0.2, 0.2, 0.2])

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

        # Create functions to inverse the bbox transform
        self.regress_boxes = RegressBoxes(mean=self.regression_mean, std=self.regression_std)
        self.clip_boxes    = ClipBoxes()

        # Create funciton to apply NMS
        self.filter_detections = FilterDetections()

    def forward(self, x):
        # Calculate features
        C3, C4, C5 =  self.backbone_model(x)
        features = self.feature_pyramid_submodel(C3, C4, C5)

        # Apply regression and classificatio submodels on each feature
        regression_outputs     = [self.regression_submodel(f)     for f in features]
        classification_outputs = [self.classification_submodel(f) for f in features]

        # Concat outputs
        regression     = torch.cat(regression_outputs    , 1)
        classification = torch.cat(classification_outputs, 1)

        # Train on regression and classification
        if self.training:
            return regression, classification

        # Collect batch information
        current_batch_size = x.shape[0]
        current_batch_image_shape = torch.Tensor(list(x.shape))
        feature_shapes = self.fpn_feature_shape_fn(current_batch_image_shape)

        # Compute base anchors
        anchors = self.compute_anchors(current_batch_size, feature_shapes)

        # Apply predicted regression to anchors
        boxes = self.regress_boxes(anchors, regression)
        boxes = self.clip_boxes(current_batch_image_shape, anchors)

        detections = self.filter_detections(boxes, classification)

        return detections

    ###########################################################################
    #### Start of collate_fn

    collate_fn = collate_fn
