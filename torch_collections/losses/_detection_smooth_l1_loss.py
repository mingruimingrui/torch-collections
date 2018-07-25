import torch


class DetectionSmoothL1Loss(torch.nn.Module):
    """ Huber Loss for detection with anchor states
    sigma configured on initialization

    Returns None in the event that loss should not be computed
    (Done so that backprop can be easily ommitted in such events)

    When calling forward,
        model output should be (num_batch, num_anchors, 4) shaped
        model target should be (num_batch, num_anchors, 4 + 1) shaped

    The + 1 is the index for anchor state
        -1 represents ignore anchor
        0 represents negative anchor (no class)
        1 represents positive anchor (has class)

    Each anchor for target will be of the format (x1, y1, x2, y2, anchor_state)

    Loss will be normalized across postive anchors
    """
    def __init__(self, sigma=3.0):
        super(DetectionSmoothL1Loss, self).__init__()
        self.sigma_squared = sigma ** 2

    def forward(self, regression, rgs_target):
        torch.nn.modules.loss._assert_no_grad(rgs_target)

        # Gather anchor states from cls_target
        # Anchor state is used to check how loss should be calculated
        # -1: ignore, 0: ignore, 1: positive
        anchor_state = rgs_target[:, :, -1]
        rgs_target   = rgs_target[:, :, :-1]

        # filter out "ignore" anchors
        indices    = anchor_state == 1
        regression = regression[indices]
        rgs_target = rgs_target[indices]

        if torch.sum(indices) == 0:
            # Return None if ignore all
            return None

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - rgs_target
        regression_diff = torch.abs(regression_diff)

        indices_smooth = regression_diff < 1.0 / self.sigma_squared
        regression_loss = regression_diff.clone()
        regression_loss[indices_smooth] = 0.5 * self.sigma_squared * regression_diff[indices_smooth] ** 2
        regression_loss[~indices_smooth] = regression_diff[~indices_smooth] - 0.5 / self.sigma_squared

        # compute the normalizer: the number of positive anchors
        normalizer = torch.sum(anchor_state == 1).float().clamp(min=1)

        return torch.sum(regression_loss) / normalizer
