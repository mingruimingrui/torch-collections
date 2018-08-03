""" Simple encoder for feature extraction """

import torch

# Default configs
from ._encoder_configs import make_configs

# Backbone loader functions
from ._backbone import (
    build_backbone_model,
    get_backbone_channel_sizes,
    build_fpn_feature_shape_fn
)


class Encoder(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

        # Make config
        self.configs = make_configs(**kwargs)

        # Make helper functions and modules
        self.build_modules()

    def build_modules(self):
        self.backbone_model = build_backbone_model(self.configs['backbone'])
        feature_size = get_backbone_channel_sizes(self.configs['backbone'])[2]
        feature_shapes = build_fpn_feature_shape_fn(self.configs['backbone'])(self.configs['input_size'])[2]
        self.avgpool = torch.nn.AvgPool2d(feature_shapes, stride=1)
        self.fc = torch.nn.Linear(feature_size, self.configs['embedding_size'])

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)

        return output

    def forward(self, image):
        # Calculate features
        _, _, x =  self.backbone_model(image)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        features = self.fc(x)
        features = self.l2_norm(features)
        features = features * 10

        return features
