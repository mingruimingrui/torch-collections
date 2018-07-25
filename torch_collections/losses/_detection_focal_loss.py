import torch


class DetectionFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(DetectionFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, classification, cls_target):
        torch.nn.modules.loss._assert_no_grad(cls_target)

        # Gather anchor states from cls_target
        # Anchor state is used to check how loss should be calculated
        # -1: ignore, 0: negative, 1: positive
        anchor_state = cls_target[:, :, -1]
        cls_target   = cls_target[:, :, :-1]

        # Filter out ignore anchors
        indices        = anchor_state != -1
        classification = classification[indices]
        cls_target     = cls_target[indices]

        if torch.sum(indices) == 0:
            # Return None if ignore all
            return None

        # compute focal loss
        bce = -(cls_target * torch.log(classification) + (1.0 - cls_target) * torch.log(1.0 - classification))

        alpha_factor = cls_target.clone()
        alpha_factor[cls_target == 1] = self.alpha
        alpha_factor[cls_target != 1] = 1 - self.alpha

        focal_weight = classification.clone()
        focal_weight[cls_target == 1] = 1 - classification[cls_target == 1]

        focal_weight = alpha_factor * focal_weight ** self.gamma
        cls_loss = focal_weight * bce

        # Compute the normalizing factor: number of positive anchors
        normalizer = torch.sum(anchor_state == 1).float()
        normalizer = max(normalizer, 1)

        return torch.sum(cls_loss) / normalizer
