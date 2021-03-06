""" To ensure that FPN can be built from as many backbones are possible, there
will be no FPN class. Instead there will be functions which will build a FPN
based on the backbone_name (along with getters for other neccessary backbone
specific variables and functions).
"""

def build_backbone_model(backbone_name, freeze_backbone=False):
    if 'resnet' in backbone_name:
        from ._resnet import ResNetBackbone as Backbone
    elif 'squeezenet' in backbone_name:
        from ._squeezenet import SqueezeNetBackbone as Backbone
    else:
        raise Exception('{} has not been implemented yet'.format(backbone_name))

    return Backbone(backbone_name, freeze_backbone=freeze_backbone)

def get_backbone_channel_sizes(backbone_name):
    if 'resnet' in backbone_name:
        from ._resnet import get_resnet_backbone_channel_sizes
        backbone_channel_sizes = get_resnet_backbone_channel_sizes(backbone_name)
    elif 'squeezenet' in backbone_name:
        from ._squeezenet import get_squeezenet_backbone_channel_sizes
        backbone_channel_sizes = get_squeezenet_backbone_channel_sizes(backbone_name)
    else:
        raise Exception('{} has not been implemented yet'.format(backbone_name))

    return backbone_channel_sizes

def build_fpn_feature_shape_fn(backbone_name):
    if 'resnet' in backbone_name:
        from ._resnet import resnet_fpn_feature_shape_fn as fpn_feature_shape_fn
    elif backbone_name == 'squeezenet1_0':
        from ._squeezenet import squeezenet1_0_fpn_feature_shape_fn as fpn_feature_shape_fn
    elif backbone_name == 'squeezenet1_1':
        from ._squeezenet import squeezenet1_1_fpn_feature_shape_fn as fpn_feature_shape_fn
    else:
        raise Exception('{} has not been implemented yet'.format(backbone_name))

    return fpn_feature_shape_fn
