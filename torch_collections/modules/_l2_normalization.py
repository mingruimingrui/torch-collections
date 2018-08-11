""" Copied from  """

import torch


class L2Normalization(torch.nn.Module):
    def __init__(self, alpha=1.0):
        super(L2Normalization, self).__init__()
        self.alpha = alpha

    def forward(self, v):
        # v_ = v / norm * alpha
        # norm = sqrt(norm_sq)
        # norm_sq = sum(v ** 2)
        v_sq= torch.pow(v, 2)

        norm_sq = torch.sum(v_sq, dim=1).add(1e-5)
        norm = torch.sqrt(norm_sq).view(-1,1).expand_as(v)

        v_ = v.div(norm).mul(self.alpha)

        return v_
