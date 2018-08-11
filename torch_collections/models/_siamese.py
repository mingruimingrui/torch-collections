import torch


class DynamicPairDistance(torch.nn.Module):
    def __init__(self, dist_type='euclidean', p=2.0):
        assert dist_type in ['euclidean', 'cosine']
        super(DynamicPairDistance, self).__init__()
        self.dist_type = dist_type
        self.p = p

    def forward(self, A):
        if self.dist_type == 'euclidean':
            # dist = [(Ai - Aj)**2]**(1/p)
            # dist = [Ai**2 + Aj**2 - 2*Ai*Aj]**(1/p)
            return (
                -2 * A.mm(A.transpose(1, 0)) +
                A.pow(2).sum(dim=1).view(1, -1) +
                A.pow(2).sum(dim=1).view(-1, 1)
            ).clamp(min=1e-5).pow(1/self.p)

        elif self.dist_type == 'cosine':
            # dist = A.dot(A.T) / (||A|| * ||A||)
            A_mag = A.pow(2).sum(dim=1).sqrt()
            AA = A_mag.view(-1, 1).mm(A_mag.view(1, -1)).clamp(min=1e-5)
            return 1 - A.mm(A.transpose(1, 0)).div(AA)

        else:
            raise ValueError('unexpected distance type, got {}, only accepts ["euclidean", "cosine"]'.format(self.dist_type))


class DynamicDistanceFunction(torch.nn.Module):
    def __init__(self, dist_type='euclidean', p=2.0):
        assert dist_type in ['euclidean', 'cosine']
        super(DynamicDistanceFunction, self).__init__()
        self.dist_type = dist_type
        self.p = p

    def forward(self, A, B):
        if self.dist_type == 'euclidean':
            return torch.nn.functional.pairwise_distance(A, B, p=self.p)

        elif self.dist_type == 'cosine':
            return 1 - torch.nn.functional.cosine_similarity(A, B)

        else:
            raise ValueError('unexpected distance type, got {}, only accepts ["euclidean", "cosine"]'.format(self.dist_type))


class DynamicTripletLoss(torch.nn.Module):
    def __init__(self, margin, dist_fn):
        super(DynamicTripletLoss, self).__init__()
        self.margin = margin
        self.dist_fn = dist_fn

    def forward(self, anchor, positive, negative):
        distance_positive = self.dist_fn(anchor, positive)
        distance_negative = self.dist_fn(anchor, negative)
        losses = distance_positive.sub(distance_negative).add(self.margin).clamp(min=0)
        return losses.mean()


class DynamicContrastiveLoss(torch.nn.Module):
    def __init__(self, margin, dist_fn):
        super(DynamicContrastiveLoss, self).__init__()
        self.margin = margin
        self.dist_fn = dist_fn

    def forward(self, A, B, target):
        distances = self.dist_fn(A, B)
        losses = torch.where(
            target == 1,
            distances,
            (self.margin - distances).clamp(min=0.0)
        )
        return losses.mean()
