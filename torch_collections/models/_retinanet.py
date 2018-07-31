""" RetinaNet submodules """

import math

import torch

from ..modules import Anchors
from ..losses import DetectionFocalLoss, DetectionSmoothL1Loss
from ..utils import anchors as utils_anchors


def compute_targets(
    batch,
    num_classes,
    fpn_feature_shape_fn,
    compute_anchors,
    regression_mean,
    regression_std
):
    # Compute anchors given image shape
    image_shape = batch['image'].shape
    feature_shapes = fpn_feature_shape_fn(image_shape)
    anchors = compute_anchors(1, feature_shapes)[0]

    # Create blobs to store anchor informations
    regression_batch = []
    labels_batch = []
    states_batch = []

    for annotations in batch['annotations']:
        labels, annotations, anchor_states = utils_anchors.anchor_targets_bbox(
            anchors,
            annotations,
            num_classes=num_classes,
            mask_shape=image_shape
        )
        regression = utils_anchors.bbox_transform(
            anchors,
            annotations,
            mean=regression_mean,
            std=regression_std
        )

        regression_batch.append(regression)
        labels_batch.append(labels)
        states_batch.append(anchor_states)

        regression_batch = torch.stack(regression_batch, dim=0)
        labels_batch     = torch.stack(labels_batch    , dim=0)
        states_batch     = torch.stack(states_batch    , dim=0)

    return {
        'regression'     : regression_batch,
        'classification' : labels_batch,
        'anchor_states'  : states_batch
    }


class FeaturePyramidSubmodel(torch.nn.Module):
    def __init__(self, backbone_channel_sizes, feature_size=256):
        super(FeaturePyramidSubmodel, self).__init__()
        C3_size, C4_size, C5_size = backbone_channel_sizes

        self.relu           = torch.nn.ReLU(inplace=False)

        self.conv_C5_reduce = torch.nn.Conv2d(C5_size     , feature_size, kernel_size=1, stride=1, padding=0)
        self.conv_P5        = torch.nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.conv_C4_reduce = torch.nn.Conv2d(C4_size     , feature_size, kernel_size=1, stride=1, padding=0)
        self.conv_P4        = torch.nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.conv_C3_reduce = torch.nn.Conv2d(C3_size     , feature_size, kernel_size=1, stride=1, padding=0)
        self.conv_P3        = torch.nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        self.conv_P6        = torch.nn.Conv2d(C5_size     , feature_size, kernel_size=3, stride=2, padding=1)
        self.conv_P7        = torch.nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, C3, C4, C5):
        # upsample C5 to get P5 from the FPN paper
        P5           = self.conv_C5_reduce(C5)
        P5_upsampled = torch.nn.functional.upsample(P5, size=C4.shape[-2:], mode='bilinear', align_corners=False)
        P5           = self.conv_P5(P5)

        # add P5 elementwise to C4
        P4           = self.conv_C4_reduce(C4)
        P4           = P5_upsampled + P4
        P4_upsampled = torch.nn.functional.upsample(P4, size=C3.shape[-2:], mode='bilinear', align_corners=False)
        P4           = self.conv_P4(P4)

        # add P4 elementwise to C3
        P3 = self.conv_C3_reduce(C3)
        P3 = P4_upsampled + P3
        P3 = self.conv_P3(P3)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        P6 = self.conv_P6(C5)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        P7 = self.relu(P6)
        P7 = self.conv_P7(P7)

        return P3, P4, P5, P6, P7


class DefaultRegressionModel(torch.nn.Module):
    def __init__(
        self,
        num_anchors,
        pyramid_feature_size=256,
        regression_feature_size=256
    ):
        super(DefaultRegressionModel, self).__init__()

        # Make all layers
        self.relu  = torch.nn.ReLU(inplace=False)
        self.conv1 = torch.nn.Conv2d(pyramid_feature_size   , regression_feature_size, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(regression_feature_size, regression_feature_size, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(regression_feature_size, regression_feature_size, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(regression_feature_size, regression_feature_size, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(regression_feature_size, num_anchors * 4        , kernel_size=3, stride=1, padding=1)

        # Initialize weights
        for i in range(1, 6):
            # kernel ~ normal(mean=0.0, std=0.01)
            # bias   ~ 0.0
            kernel = getattr(self, 'conv{}'.format(i)).weight
            bias   = getattr(self, 'conv{}'.format(i)).bias
            torch.nn.init.normal_(kernel, 0.0, 0.01)
            bias.data.fill_(0.0)

    def forward(self, x):
        for i in range(1, 5):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = self.relu(x)
        x = self.conv5(x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, 4)


class DefaultClassificationModel(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        num_anchors,
        pyramid_feature_size=256,
        prior_probability=0.01,
        classification_feature_size=256
    ):
        super(DefaultClassificationModel, self).__init__()
        self.num_classes = num_classes

        # Make all layers
        self.relu    = torch.nn.ReLU(inplace=False)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv1   = torch.nn.Conv2d(pyramid_feature_size       , classification_feature_size, kernel_size=3, stride=1, padding=1)
        self.conv2   = torch.nn.Conv2d(classification_feature_size, classification_feature_size, kernel_size=3, stride=1, padding=1)
        self.conv3   = torch.nn.Conv2d(classification_feature_size, classification_feature_size, kernel_size=3, stride=1, padding=1)
        self.conv4   = torch.nn.Conv2d(classification_feature_size, classification_feature_size, kernel_size=3, stride=1, padding=1)
        self.conv5   = torch.nn.Conv2d(classification_feature_size, num_anchors * num_classes  , kernel_size=3, stride=1, padding=1)

        # Initialize weights
        for i in range(1, 5):
            # kernel ~ normal(mean=0.0, std=0.01)
            # bias   ~ 0.0
            kernel = getattr(self, 'conv{}'.format(i)).weight
            bias   = getattr(self, 'conv{}'.format(i)).bias
            torch.nn.init.normal_(kernel, 0.0, 0.01)
            bias.data.fill_(0.0)

        # Initialize classification output differently
        # kernel ~ 0.0
        # bias   ~ -log((1 - 0.01) / 0.01)
        #     So that output is 0.01 after sigmoid
        kernel = self.conv5.weight
        bias   = self.conv5.bias
        kernel.data.fill_(0.0)
        bias.data.fill_(-math.log((1 - 0.01) / 0.01))

    def forward(self, x):
        for i in range(1,5):
            x = getattr(self, 'conv{}'.format(i))(x)
            x = self.relu(x)
        x = self.conv5(x)
        x = self.sigmoid(x)
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, self.num_classes)


class ComputeAnchors(torch.nn.Module):
    def __init__(self, sizes, strides, ratios, scales):
        super(ComputeAnchors, self).__init__()
        assert len(sizes) == 5
        assert len(strides) == 5
        self.levels = [3, 4, 5, 6, 7]

        for level, size, stride in zip(self.levels, sizes, strides):
            setattr(self, 'anchor_P{}'.format(level), Anchors(
                size=size,
                stride=stride,
                ratios=ratios,
                scales=scales
            ))

    def forward(self, batch_size, feature_shapes):
        all_anchors = []

        for level, feature_shape in zip(self.levels, feature_shapes):
            anchors = getattr(self, 'anchor_P{}'.format(level))(batch_size, feature_shape[-2:])
            all_anchors.append(anchors)

        return torch.cat(all_anchors, dim=1)


class RetinaNetLoss(torch.nn.Module):
    def __init__(
        self,
        num_classes,
        fpn_feature_shape_fn,
        compute_anchors,
        focal_alpha=0.25,
        focal_gamma=2.0,
        huber_sigma=3.0,
        regression_mean=0.0,
        regression_std=0.2
    ):
        super(RetinaNetLoss, self).__init__()
        self.num_classes = num_classes
        self.fpn_feature_shape_fn = fpn_feature_shape_fn
        self.compute_anchors = compute_anchors

        self.regression_mean = regression_mean
        self.regression_std  = regression_std

        self.focal_loss_fn = DetectionFocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.huber_loss_fn = DetectionSmoothL1Loss(sigma=huber_sigma)

    def forward(self, outputs, batch):
        # Compute targets
        targets = compute_targets(
            batch,
            num_classes=self.num_classes,
            fpn_feature_shape_fn=self.fpn_feature_shape_fn,
            compute_anchors=self.compute_anchors,
            regression_mean=self.regression_mean,
            regression_std=self.regression_std
        )

        # Calculate losses
        classification_loss = self.focal_loss_fn(
            outputs['classification'],
            targets['classification'],
            targets['anchor_states']
        )

        regression_loss = self.huber_loss_fn(
            outputs['regression'],
            targets['regression'],
            targets['anchor_states']
        )

        # Return None if all anchors are too be ignored
        # Provides an easy way to skip back prop
        if classification_loss is None:
            return None

        # Regression loss defaults to 0 in the event that there is no positive anchors
        if regression_loss is None:
            regression_loss = 0.0

        return classification_loss + regression_loss
