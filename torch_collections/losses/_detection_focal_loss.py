import torch


class DetectionFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(DetectionFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, classification, target):
        torch.nn.modules.loss._assert_no_grad(target)

        # Gather ancho states from target
        anchor_state = target[:, :, -1]
        target       = target[:, :, :-1]

        # Filter out ignore anchors
        indices = anchor_state != -1
        if torch.sum(indices) == 0:
            # Return 0 if ignore all
            return torch.zeros_like(classification[0, 0, 0])
        classification   = classification[indices]
        target  = target[indices]

        # compute focal loss
        bce = -(target * torch.log(classification) + (1.0 - target) * torch.log(1.0 - classification))

        alpha_factor = torch.ones_like(target)
        alpha_factor = alpha_factor * self.alpha
        alpha_factor[target != 1] = 1 - self.alpha

        focal_weight = classification
        focal_weight[target == 1] = 1 - focal_weight[target == 1]
        focal_weight = alpha_factor * focal_weight ** self.gamma

        cls_loss = focal_weight * bce

        # Compute the normalizing factor: number of positive anchors
        normalizer = torch.sum(anchor_state == 1).float()
        normalizer = max(normalizer, 1)

        return torch.sum(cls_loss) / normalizer
