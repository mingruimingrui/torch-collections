import torch
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
    image_shape = torch.Tensor(list(batch['image'].shape))
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


def compute_focal_loss(
    cls_inputs,
    cls_targets,
    anchor_states,
    alpha,
    gamma,
):
    torch.nn.modules.loss._assert_no_grad(cls_targets)

    # Filter out ignore anchors
    indices    = anchor_states != -1
    cls_inputs  = cls_inputs[indices]
    cls_targets = cls_targets[indices]

    if torch.sum(indices) == 0:
        # Return None if ignore all
        return None

    # compute focal loss
    bce = -(cls_targets * torch.log(cls_inputs) + (1.0 - cls_targets) * torch.log(1.0 - cls_inputs))

    alpha_factor = cls_targets.clone()
    alpha_factor[cls_targets == 1] = alpha
    alpha_factor[cls_targets != 1] = 1 - alpha

    focal_weight = cls_inputs.clone()
    focal_weight[cls_targets == 1] = 1 - cls_inputs[cls_targets == 1]

    focal_weight = alpha_factor * focal_weight ** gamma
    cls_loss = focal_weight * bce

    # Compute the normalizing factor: number of positive anchors
    normalizer = torch.sum(anchor_states == 1).float().clamp(min=10)

    return torch.sum(cls_loss) / normalizer


def compute_huber_loss(
    rgs_inputs,
    rgs_targets,
    anchor_states,
    sigma_squared
):
    torch.nn.modules.loss._assert_no_grad(rgs_targets)

    # filter out "ignore" anchors
    indices     = anchor_states == 1
    rgs_inputs  = rgs_inputs[indices]
    rgs_targets = rgs_targets[indices]

    if torch.sum(indices) == 0:
        # Return None if ignore all
        return None

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    rgs_diff = rgs_inputs - rgs_targets
    rgs_diff = torch.abs(rgs_diff)

    indices_smooth = rgs_diff < 1.0 / sigma_squared
    rgs_loss = rgs_diff.clone()
    rgs_loss[indices_smooth] = 0.5 * sigma_squared * rgs_diff[indices_smooth] ** 2
    rgs_loss[~indices_smooth] = rgs_diff[~indices_smooth] - 0.5 / sigma_squared

    # compute the normalizer: the number of positive anchors
    normalizer = torch.sum(anchor_states == 1).float().clamp(min=1)

    return torch.sum(rgs_loss) / normalizer


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

        self.alpha = focal_alpha
        self.gamma = focal_gamma
        self.sigma_squared = huber_sigma ** 2

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
        classification_loss = compute_focal_loss(
            outputs['classification'],
            targets['classification'],
            targets['anchor_states'],
            alpha=self.alpha,
            gamma=self.gamma,
        )

        regression_loss = compute_huber_loss(
            outputs['regression'],
            targets['regression'],
            targets['anchor_states'],
            sigma_squared=self.sigma_squared
        )

        # Return None if all anchors are too be ignored
        # Provides an easy way to skip back prop
        if classification_loss is None:
            return None

        # Regression loss defaults to 0 in the event that there is no positive anchors
        if regression_loss is None:
            regression_loss = 0.0

        return classification_loss + regression_loss
