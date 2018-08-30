import math
import torch
import torchvision


def squeezenet1_0_fpn_feature_shape_fn(img_shape):
    """ Takes an image_shape as an input to calculate the FPN output sizes
    Ensure that img_shape is of the format (..., H, W)

    Args
        img_shape : image shape as torch.Tensor not torch.Size should have
            H, W as last 2 axis
    Returns
        P3_shape, P4_shape, P5_shape, P6_shape, P7_shape : as 5 (2,) Tensors
    """
    C0_shape = img_shape[-2:]

    C1_shape = (math.floor((C0_shape[0] - 5) / 2), math.floor((C0_shape[1] - 5) / 2))
    C2_shape = (math.ceil((C1_shape[0] - 1) / 2), math.ceil((C1_shape[1] - 1) / 2))

    P3_shape = (math.ceil((C2_shape[0] - 1) / 2), math.ceil((C2_shape[1] - 1) / 2))
    P4_shape = (math.ceil((P3_shape[0] - 1) / 2), math.ceil((P3_shape[1] - 1) / 2))
    P5_shape = (math.ceil((P4_shape[0] - 1) / 2), math.ceil((P4_shape[1] - 1) / 2))

    P6_shape = (math.ceil(P5_shape[0] / 2), math.ceil(P5_shape[1] / 2))
    P7_shape = (math.ceil(P6_shape[0] / 2), math.ceil(P6_shape[1] / 2))

    return C2_shape, P3_shape, P4_shape, P5_shape, P6_shape, P7_shape


def squeezenet1_1_fpn_feature_shape_fn(img_shape):
    """ Takes an image_shape as an input to calculate the FPN output sizes
    Ensure that img_shape is of the format (..., H, W)

    Args
        img_shape : image shape as torch.Tensor not torch.Size should have
            H, W as last 2 axis
    Returns
        P3_shape, P4_shape, P5_shape, P6_shape, P7_shape : as 5 (2,) Tensors
    """
    C0_shape = img_shape[-2:]

    C1_shape = (math.floor((C0_shape[0] - 1) / 2), math.floor((C0_shape[1] - 1) / 2))
    C2_shape = (math.ceil((C1_shape[0] - 1) / 2), math.ceil((C1_shape[1] - 1) / 2))

    P3_shape = (math.ceil((C2_shape[0] - 1) / 2), math.ceil((C2_shape[1] - 1) / 2))
    P4_shape = (math.ceil((P3_shape[0] - 1) / 2), math.ceil((P3_shape[1] - 1) / 2))
    P5_shape = (math.ceil((P4_shape[0] - 1) / 2), math.ceil((P4_shape[1] - 1) / 2))

    P6_shape = (math.ceil(P5_shape[0] / 2), math.ceil(P5_shape[1] / 2))
    P7_shape = (math.ceil(P6_shape[0] / 2), math.ceil(P6_shape[1] / 2))

    return C2_shape, P3_shape, P4_shape, P5_shape, P6_shape, P7_shape


def get_squeezenet_backbone_channel_sizes(backbone_name):
    if backbone_name == 'squeezenet1_0':
        sizes = [64, 128, 256, 512]
    elif backbone_name == 'squeezenet1_1':
        sizes = [96, 256, 512, 512]
    else:
        raise Exception('{} is not a valid squeezenet backbone'.format(backbone_name))

    return sizes


class SqueezeNetBackbone(torch.nn.Module):
    def __init__(self, backbone_name, freeze_backbone=False):
        super(SqueezeNetBackbone, self).__init__()
        self.backbone_name = backbone_name

        # Load a pretrained squeezenet model
        squeezenet_model = getattr(torchvision.models, self.backbone_name)(pretrained=True)

        # Copy layers with weights
        self.conv1   = squeezenet_model.features[0]
        self.relu    = squeezenet_model.features[1]
        self.maxpool = squeezenet_model.features[2]

        if self.backbone_name == 'squeezenet1_0':
            self.fire1 = squeezenet_model.features[3]
            self.fire2 = squeezenet_model.features[4]
            self.fire3 = squeezenet_model.features[5]
            self.fire4 = squeezenet_model.features[7]
            self.fire5 = squeezenet_model.features[8]
            self.fire6 = squeezenet_model.features[9]
            self.fire7 = squeezenet_model.features[10]
            self.fire8 = squeezenet_model.features[12]

        elif self.backbone_name == 'squeezenet1_1':
            self.fire1 = squeezenet_model.features[3]
            self.fire2 = squeezenet_model.features[4]
            self.fire3 = squeezenet_model.features[6]
            self.fire4 = squeezenet_model.features[7]
            self.fire5 = squeezenet_model.features[9]
            self.fire6 = squeezenet_model.features[10]
            self.fire7 = squeezenet_model.features[11]
            self.fire8 = squeezenet_model.features[12]

        else:
            raise Exception('{} is not a valid squeezenet backbone'.format(self.backbone_name))

        # Delete uneeded tensors
        del squeezenet_model

        # Freeze backbone if flagged
        if freeze_backbone:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Get layers which outputs C3, C4, C5
        C1 = self.conv1(x)
        self.relu(C1)

        C2 = self.maxpool(C1)
        if self.backbone_name == 'squeezenet1_0':
            C2 = self.fire1(C2)
            C2 = self.fire2(C2)
            C2 = self.fire3(C2)

            C3 = self.maxpool(C2)
            C3 = self.fire4(C3)
            C3 = self.fire5(C3)
            C3 = self.fire6(C3)
            C3 = self.fire7(C3)

            C4 = self.maxpool(C3)
            C4 = self.fire8(C4)

        elif self.backbone_name == 'squeezenet1_1':
            C2 = self.fire1(C2)
            C2 = self.fire2(C2)

            C3 = self.maxpool(C2)
            C3 = self.fire3(C3)
            C3 = self.fire4(C3)

            C4 = self.maxpool(C3)
            C4 = self.fire5(C4)
            C4 = self.fire6(C4)
            C4 = self.fire7(C4)
            C4 = self.fire8(C4)

        else:
            raise Exception('{} is not a valid squeezenet backbone'.format(self.backbone_name))

        C5 = self.maxpool(C4)

        return C2, C3, C4, C5
