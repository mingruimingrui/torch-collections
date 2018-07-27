import torch


class DetectionFocalLoss(torch.nn.Module):
    """ Focal Loss for detection with anchor states
    alpha and gamma configured on initialization

    Returns None in the event that loss should not be computed
    (Done so that backprop can be easily ommitted in such events)

    When calling forward,
        cls_inputs and cls_targets should be (num_batch, num_anchors, num_classes) shaped
        anchor_states should be (num_batch, num_anchors) shaped and contains values of
        either -1, 0 or 1 such that
            -1 represents ignore anchor
            0 represents negative anchor (no class)
            1 represents positive anchor (has class)

    Loss will be normalized across postive anchors
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(DetectionFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, cls_inputs, cls_targets, anchor_states):
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
        alpha_factor[cls_targets == 1] = self.alpha
        alpha_factor[cls_targets != 1] = 1 - self.alpha

        focal_weight = cls_inputs.clone()
        focal_weight[cls_targets == 1] = 1 - cls_inputs[cls_targets == 1]

        focal_weight = alpha_factor * focal_weight ** self.gamma
        cls_loss = focal_weight * bce

        # Compute the normalizing factor: number of positive anchors
        normalizer = torch.sum(anchor_states == 1).float().clamp(min=10)

        return torch.sum(cls_loss) / normalizer
