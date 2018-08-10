

import torch

# Default configs
from ._siamese_configs import make_configs

# Backbone loader functions
from ._backbone import (
    build_backbone_model,
    get_backbone_channel_sizes,
    build_fpn_feature_shape_fn
)


class Siamese(torch.nn.Module):
    def __init__(self, **kwargs):
        super(Siamese, self).__init__()

        # Make config
        self.configs = make_configs(**kwargs)

        # Make helper functions and modules
        self.build_modules()

    def build_modules(self):
        self.backbone_model = build_backbone_model(self.configs['backbone'])

        feature_shape = build_fpn_feature_shape_fn(self.configs['backbone'])(self.configs['input_size'])[2]
        self.avgpool = torch.nn.AvgPool2d(feature_shape, stride=1)

        feature_size = get_backbone_channel_sizes(self.configs['backbone'])[2]
        self.fc = torch.nn.Linear(feature_size, self.configs['embedding_size'])

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        output = output * self.configs['l2_norm_alpha']

        return output

    def forward(self, x):
        _, _, x =  self.backbone_model(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        features = self.fc(x)
        features = self.l2_norm(features)

        return features
