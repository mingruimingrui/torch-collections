""" Siamese network mostly inspired by https://arxiv.org/abs/1503.03832
However is more flexible
"""

import torch

# Default configs
from ._siamese_configs import make_configs

# Backbone loader functions
from ._backbone import build_backbone_model

# Siamese submodels
from ._siamese import (
    DynamicPairDistance,
    DynamicDistanceFunction,
    DynamicTripletLoss,
    DynamicContrastiveLoss
)

# Other modules
from ..modules import L2Normalization


class Siamese(torch.nn.Module):
    def __init__(self, **kwargs):
        """ Constructs a Siamese network for image similarity by comparison of embeddings

        Args
            Refer to torch_collections.models._siamese_configs.py

        Returns
            A siamese network whos inputs/outputs are as follows
            - Input is an image tensor in the format NCHW, normalized to pytorch standard
            - Output is an embedding with l2_norm applied

        One thing to note is that center loss is not implemented here as output is expected
        to be an embedding
        """
        super(Siamese, self).__init__()

        # Make configs
        self.configs = make_configs(**kwargs)

        # Make helper functions and modules
        self.build_modules()

    def build_modules(self):
        # Build backbone model
        self.backbone_model = build_backbone_model(self.configs['backbone'])

        # Retrieve deepest backbone_model output shape
        dummy_input = torch.Tensor(1, 3, self.configs['input_size'][0], self.configs['input_size'][1])
        dummy_output = self.backbone_model(dummy_input)[-1]
        dummy_output_shape = dummy_output.shape
        del dummy_input
        del dummy_output

        # Build avgool and fc layers based on the backbone output shape
        self.avgpool = torch.nn.AvgPool2d(dummy_output_shape[-2:], stride=1)
        self.fc = torch.nn.Linear(dummy_output_shape[1], self.configs['embedding_size'])

        # L2 normalization layer as suggested by the facenet paper
        # https://arxiv.org/abs/1503.03832
        self.l2_norm = L2Normalization(alpha=self.configs['l2_norm_alpha'])

        # Distance functions
        self.pdist_fn = DynamicPairDistance(dist_type=self.configs['dist_type'], p=self.configs['p_norm'])
        self.dist_fn = DynamicDistanceFunction(dist_type=self.configs['dist_type'], p=self.configs['p_norm'])

        # loss functions
        self.triplet_loss = DynamicTripletLoss(margin=self.configs['margin'], dist_fn=self.dist_fn)
        self.contrastive_loss = DynamicContrastiveLoss(margin=self.configs['margin'], dist_fn=self.dist_fn)

    def forward(self, x):
        x =  self.backbone_model(x)[-1]
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.l2_norm(x)
        return x
