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
    ComputeAnchors
)

# Common detection submodules
from ..modules import RegressBoxes, ClipBoxes

# Other utility functions
from ..utils import transforms
from ..utils import anchors as utils_anchors


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

        # detections = self.filter_detections(boxes, classification)
        detections = boxes

        return detections

    ###########################################################################
    #### Start of collate_fn

    def collate_fn(self, sample_group):
        """ Collate fn requires datasets which returns samples as a dict in the following format
        sample = {
            'image'       : Image in HWC RGB format as a numpy.ndarray,
            'annotations' : Annotations of shape (num_annotations, 5) also numpy.ndarray
                - each row will represent 1 detection target of the format
                (x1, y1, x2, y2, class_id)
        }
        """
        # Gather image and annotations group
        image_group       = [sample['image'] for sample in sample_group]
        annotations_group = [sample['annotations'] for sample in sample_group]

        # Preprocess individual samples
        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            image, scale = transforms.resize_image_1(
                image,
                min_side=self.configs['image_min_side'],
                max_side=self.configs['image_max_side']
            )
            annotations[:, :4] *= scale
            image_group[index] = image
            annotations_group[index] = annotations

        # Augment samples
        #TODO: Implement functions for image augmentation

        # Compile samples into batches
        max_image_shape = tuple(max(image.shape[x] for image in image_group) for x in range(2))
        feature_shapes = self.fpn_feature_shape_fn(torch.Tensor(max_image_shape))
        anchors = self.compute_anchors(1, feature_shapes)[0]

        image_batch = []
        regression_batch = []
        labels_batch = []

        for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
            image = transforms.pad_to(image, max_image_shape + (3,))
            image = self.to_tensor(image)
            image = self.normalize(image)

            labels, annotations, anchor_states = utils_anchors.anchor_targets_bbox(
                anchors,
                torch.Tensor(annotations),
                torch.Tensor([self.configs['num_classes']]),
                mask_shape=image.shape
            )
            regression = utils_anchors.bbox_transform(
                anchors,
                annotations,
                mean=self.regression_mean,
                std=self.regression_std
            )
            anchor_states = anchor_states.reshape(-1, 1)
            regression = torch.cat([regression, anchor_states], dim=1)
            labels     = torch.cat([labels    , anchor_states], dim=1)

            image_batch.append(image)
            regression_batch.append(regression)
            labels_batch.append(labels)

        image_batch      = torch.stack(image_batch     , dim=0)
        regression_batch = torch.stack(regression_batch, dim=0)
        labels_batch     = torch.stack(labels_batch    , dim=0)

        return {
            'image'         : image_batch,
            'regression'    : regression_batch,
            'classification': labels_batch
        }
