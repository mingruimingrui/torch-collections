""" Unet for image segmentation https://arxiv.org/pdf/1505.04597.pdf """

import math
import torch

# Default configs
from ._unet_configs import make_configs

# Backbone loader functions
from ._backbone import build_backbone_model, get_backbone_channel_sizes

# Other modules
from ..modules import ConvBlock2d


class Unet(torch.nn.Module):
    def __init__(self, num_classes, **kwargs):
        """ Unet model for image segmentation """
        super(Unet, self).__init__()

        # Make config
        kwargs['num_classes'] = num_classes
        self.configs = make_configs(**kwargs)

        # Make helper functions and modules
        self.build_modules()

    def build_modules(self):
        # Build backbone model
        backbone_model = build_backbone_model(
            self.configs['backbone'],
            freeze_backbone=self.configs['freeze_backbone']
        )
        # Get backbone channel sizes
        channel_sizes = get_backbone_channel_sizes(self.configs['backbone'])

        # Misc layers
        # self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Extract layers from backbone model
        self.conv_C1 = torch.nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu,
            backbone_model.maxpool
        )
        self.conv_C2 = backbone_model.layer1
        self.conv_C3 = backbone_model.layer2
        self.conv_C4 = backbone_model.layer3
        self.conv_C5 = backbone_model.layer4

        # Create deconvolution layers
        self.conv_P4 = ConvBlock2d(
            input_feature_size=channel_sizes[-2] + channel_sizes[-1],
            internal_feature_size=256,
            output_feature_size=128,
            num_layers=2,
            batch_norm=True,
            bias=False,
        )

        self.conv_P3 = ConvBlock2d(
            input_feature_size=channel_sizes[-3] + 128,
            internal_feature_size=128,
            output_feature_size=64,
            num_layers=2,
            batch_norm=True,
            bias=False,
        )

        self.conv_P2 = ConvBlock2d(
            input_feature_size=channel_sizes[-4] + 64,
            internal_feature_size=64,
            output_feature_size=64,
            num_layers=2,
            batch_norm=True,
            bias=False,
        )

        self.conv_P1 = ConvBlock2d(
            input_feature_size=channel_sizes[-4] + 64,
            internal_feature_size=64,
            output_feature_size=128,
            num_layers=2,
            batch_norm=True,
            bias=False,
        )

        self.conv_P0 = ConvBlock2d(
            input_feature_size=3 + 128,
            internal_feature_size=64,
            output_feature_size=128,
            num_layers=2,
            batch_norm=True,
            bias=False,
        )

        self.classifier = torch.nn.Conv2d(
            128,
            self.configs['num_classes'],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        # Initialize classification output to 0.01
        # kernel ~ 0.0
        # bias   ~ -log((1 - 0.01) / 0.01)  So that output is 0.01 after sigmoid
        kernel = self.classifier.weight
        bias   = self.classifier.bias
        kernel.data.fill_(0.0)
        bias.data.fill_(-math.log((1 - 0.01) / 0.01))

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, C0):
        C1 = self.conv_C1(C0)
        C2 = self.conv_C2(C1)
        C3 = self.conv_C3(C2)
        C4 = self.conv_C4(C3)
        C5 = self.conv_C5(C4)

        C5_upsampled = torch.nn.functional.interpolate(C5, size=C4.shape[-2:], mode='bilinear', align_corners=False)
        P4 = self.conv_P4(torch.cat((C5_upsampled, C4), dim=1))
        P4_upsampled = torch.nn.functional.interpolate(P4, size=C3.shape[-2:], mode='bilinear', align_corners=False)
        P3 = self.conv_P3(torch.cat((P4_upsampled, C3), dim=1))
        P3_upsampled = torch.nn.functional.interpolate(P3, size=C2.shape[-2:], mode='bilinear', align_corners=False)
        P2 = self.conv_P2(torch.cat((P3_upsampled, C2), dim=1))
        P2_upsampled = torch.nn.functional.interpolate(P2, size=C1.shape[-2:], mode='bilinear', align_corners=False)
        P1 = self.conv_P1(torch.cat((P2_upsampled, C1), dim=1))
        P1_upsampled = torch.nn.functional.interpolate(P1, size=C0.shape[-2:], mode='bilinear', align_corners=False)
        P0 = self.conv_P0(torch.cat((P1_upsampled, C0), dim=1))
        output = self.classifier(P0)

        return self.sigmoid(output)

        # self.C4b(torch.cat())



        # center = self.center(self.pool(C5))
        #
        # deC4 = self.deC4(torch.cat([center, C5], 1))
        # import pdb; pdb.set_trace()
        # deC3 = self.deC3(torch.cat([deC4, C4], 1))
        # deC2 = self.deC2(torch.cat([deC3, C3], 1))
        # deC1 = self.deC1(torch.cat([deC2, C2], 1))
        # deconv0 = self.deconv0(deC1)

        # return self.final(dec0)
