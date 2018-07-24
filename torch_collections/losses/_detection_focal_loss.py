import torch


class DetectionFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(DetectionFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, input, target):
        torch.nn.modules.loss._assert_no_grad(target)

        # Gather ancho states from target
        anchor_state = target[:, :, -1]
        target       = target[:, :, :-1]

        # Filter out ignore anchors
        indices = anchor_state != -1
        input   = input[indices]
        target  = target[indices]

        # compute focal loss
        alpha_factor = torch.ones_like(target)
        alpha_factor = alpha_factor * self.alpha
        alpha_factor[target != 1] = 1 - self.alpha

        focal_weight = input
        focal_weight[target == 1] = 1 - focal_weight[target == 1]
        focal_weight = alpha_factor * focal_weight ** self.gamma

        bce = -(target * torch.log(input) + (1.0 - target) * torch.log(1.0 - input))

        cls_loss = focal_weight * bce

        # Compute the normalizing factor: number of positive anchors
        normalizer = torch.sum(anchor_state == 1).float()
        normalizer = max(normalizer, 1)

        import pdb; pdb.set_trace()

        return torch.sum(cls_loss) / normalizer
