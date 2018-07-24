import torch


class DetectionSmoothL1Loss(torch.nn.Module):
    def __init__(self, sigma=3.0):
        super(DetectionSmoothL1Loss, self).__init__()
        self.sigma_squared = sigma ** 2

    def forward(self, regression, target):
        regression_target = target[:, :, :4]
        anchor_state      = target[:, :, 4]

        # filter out "ignore" anchors
        indices           = anchor_state == 1
        if torch.sum(indices) == 0:
            # Return 0 if ignore all
            return torch.zeros_like(regression[0, 0, 0])
        regression        = regression[indices]
        regression_target = regression_target[indices]

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = torch.abs(regression_diff)

        to_smooth = regression_diff < 1.0 / self.sigma_squared
        regression_loss = torch.zeros_like(regression_diff)
        regression_loss[to_smooth] = 0.5 * self.sigma_squared * regression_diff[to_smooth].clone() ** 2
        regression_loss[to_smooth == 0] = regression_diff[to_smooth == 0].clone() - 0.5 / self.sigma_squared

        # compute the normalizer: the number of positive anchors
        normalizer = torch.sum(anchor_state == 1).float()
        normalizer = max(normalizer, 1)

        return torch.sum(regression_loss) / normalizer
