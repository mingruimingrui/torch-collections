import math
import torch
import torchvision


def resnet_fpn_feature_shape_fn(img_shape):
    """ Takes an image_shape as an input to calculate the FPN output sizes
    Ensure that img_shape is of the format (..., H, W)

    Args
        img_shape : image shape as torch.Tensor not torch.Size should have
            H, W as last 2 axis
    Returns
        P3_shape, P4_shape, P5_shape, P6_shape, P7_shape : as 5 (2,) Tensors
    """
    C0_shape = img_shape[-2:]

    C1_shape = (math.ceil(C0_shape[0] / 2), math.ceil(C0_shape[1] / 2))
    C2_shape = (math.ceil(C1_shape[0] / 2), math.ceil(C1_shape[1] / 2))

    P3_shape = (math.ceil(C2_shape[0] / 2), math.ceil(C2_shape[1] / 2))
    P4_shape = (math.ceil(P3_shape[0] / 2), math.ceil(P3_shape[1] / 2))
    P5_shape = (math.ceil(P4_shape[0] / 2), math.ceil(P4_shape[1] / 2))

    P6_shape = (math.ceil(P5_shape[0] / 2), math.ceil(P5_shape[1] / 2))
    P7_shape = (math.ceil(P6_shape[0] / 2), math.ceil(P6_shape[1] / 2))

    return C2_shape, P3_shape, P4_shape, P5_shape, P6_shape, P7_shape


def get_resnet_backbone_channel_sizes(backbone_name):
    if backbone_name in ['resnet18', 'resnet34']:
        sizes = [64, 128, 256, 512]
    elif backbone_name in ['resnet50', 'resnet101', 'resnet152']:
        sizes = [256, 512, 1024, 2048]
    else:
        raise Exception('{} is not a valid resnet backbone'.format(backbone_name))

    return sizes


class ResNetBackbone(torch.nn.Module):
    def __init__(self, backbone_name, freeze_backbone=False):
        super(ResNetBackbone, self).__init__()

        # Load a pretrained resnet model
        resnet_model = getattr(torchvision.models, backbone_name)(pretrained=True)

        # Copy layers with weights
        self.conv1   = resnet_model.conv1
        self.bn1     = resnet_model.bn1
        self.relu    = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1  = resnet_model.layer1
        self.layer2  = resnet_model.layer2
        self.layer3  = resnet_model.layer3
        self.layer4  = resnet_model.layer4

        # Delete unused layers
        del resnet_model.avgpool
        del resnet_model.fc
        del resnet_model

        # Freeze batch norm
        for layer in self.modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                for param in layer.parameters():
                    param.requires_grad = False

        # Freeze backbone if flagged
        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Get layers which outputs C3, C4, C5
        C1 = self.conv1(x)
        C1 = self.bn1(C1)
        C1 = self.relu(C1)

        C2 = self.maxpool(C1)
        C2 = self.layer1(C2)

        C3 = self.layer2(C2)

        C4 = self.layer3(C3)

        C5 = self.layer4(C4)

        return C2, C3, C4, C5
