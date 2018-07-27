import torch


class DetectionSmoothL1Loss(torch.nn.Module):
    """ Huber Loss for detection with anchor states
    sigma configured on initialization

    Returns None in the event that loss should not be computed
    (Done so that backprop can be easily ommitted in such events)

    When calling forward,
        rgs_inputs and rgs_targets should be (num_batch, num_anchors, 4) shaped
        anchor_states should be (num_batch, num_anchors) shaped and contains values of
        either -1, 0 or 1 such that
            -1 represents ignore anchor
            0 represents negative anchor (no class)
            1 represents positive anchor (has class)

    Loss will be normalized across postive anchors
    """
    def __init__(self, sigma=3.0):
        super(DetectionSmoothL1Loss, self).__init__()
        self.sigma_squared = sigma ** 2

    def forward(self, rgs_inputs, rgs_targets, anchor_states):
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

        indices_smooth = rgs_diff < 1.0 / self.sigma_squared
        rgs_loss = rgs_diff.clone()
        rgs_loss[indices_smooth] = 0.5 * self.sigma_squared * rgs_diff[indices_smooth] ** 2
        rgs_loss[~indices_smooth] = rgs_diff[~indices_smooth] - 0.5 / self.sigma_squared

        # compute the normalizer: the number of positive anchors
        normalizer = torch.sum(anchor_states == 1).float().clamp(min=1)

        return torch.sum(rgs_loss) / normalizer
