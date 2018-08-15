""" Referenced from https://github.com/adambielski/siamese-triplet/blob/master/utils.py """

import itertools

import numpy as np
import torch


def random_hard_negative(loss_values, margin):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


def hardest_negative(loss_values, margin):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


class TripletSelector(torch.nn.Module):
    def __init__(self, negative_mining_type, margin, pdist, cpu=True):
        super(TripletSelector, self).__init__()
        self.margin = margin
        self.pdist = pdist
        self.cpu = cpu
        if negative_mining_type == 'random':
            self.negative_selection_fn = random_hard_negative
        elif negative_mining_type == 'semihard':
            self.negative_selection_fn = semihard_negative
        elif negative_mining_type == 'hard':
            self.negative_selection_fn = hardest_negative

    def forward(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = self.pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(itertools.combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values, self.margin)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


class PairSelector(torch.nn.Module):
    def __init__(self):
        super(PairSelector, self).__init__()
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
