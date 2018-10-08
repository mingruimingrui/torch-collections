from __future__ import division
from copy import deepcopy

from ..utils.collections import AttrDict

ACCEPTED_FEATURE_LEVELS = set([2, 3, 4, 5, 6, 7])

# Define default parameters
_c = AttrDict()

################################################################################
#### Start of configurable parameters

#### Model configs
_c.name        = 'retinanet'
_c.num_classes = None

_c.backbone        = 'resnet50'
_c.freeze_backbone = False  # Does not actually work at the moment will implement in future
_c.return_loss     = True   # If True, returns loss during training

_c.pyramid_feature_levels = [3, 4, 5, 6, 7]  # Any combination of [2, 3, 4, 5, 6, 7] in list form
_c.pyramid_feature_size   = 256

_c.anchor_sizes   = [32, 64, 128, 256, 512]  # Same number of values as feature levels
_c.anchor_strides = [8, 16, 32, 64, 128]     # Same number of values as feature levels
_c.anchor_ratios  = [0.5, 1., 2.]
_c.anchor_scales  = [2. ** 0., 2. ** (1. / 3.), 2. ** (2. / 3.)]

_c.regression_block_type   = 'fc'  # one of ['fc', 'dense']
_c.regression_num_layers   = 4
_c.regression_feature_size = 256  # Regression model internal channel size (only for 'fc')
_c.regression_growth_rate  = 64   # Regression model channel growth rate (only for 'dense')

_c.classification_block_type   = 'fc'  # one of ['fc', 'dense']
_c.classification_num_layers   = 4
_c.classification_feature_size = 256  # Classification model internal channel size (only for 'fc')
_c.classification_growth_rate  = 64   # Classification model channel growth rate (only for 'dense')

_c.apply_nms       = True
_c.nms_threshold   = 0.5
_c.score_threshold = 0.05
_c.max_detections  = 300
_c.nms_type        = 'hard'
_c.nms_use_cpu     = True

################################################################################
#### End of configurable parameters

# Set default configs to be immutable
_c.immutable(True)

def validate_configs(configs):
    assert isinstance(configs.num_classes, int), 'num_classes must be specified'
    assert 'resnet' in configs.backbone, 'only resnet backbones supported'

    assert isinstance(configs.pyramid_feature_levels, (list, tuple)), 'feature levels should be a list'
    assert all([(l in ACCEPTED_FEATURE_LEVELS) for l in configs.pyramid_feature_levels])

    num_levels = len(configs.pyramid_feature_levels)
    assert len(configs.anchor_sizes) == num_levels, 'number of anchor_sizes should be same as feature levels'
    assert len(configs.anchor_strides) == num_levels, 'number of anchor_strides should be same as feature levels'

    assert configs.regression_block_type in ['fc', 'dense'], "regression_block_type should be in ['fc', 'dense']"
    assert configs.classification_block_type in ['fc', 'dense'], "classification_block_type should be in ['fc', 'dense']"

    configs.num_anchors = len(configs.anchor_ratios) * len(configs.anchor_scales)

def make_configs(**kwargs):
    configs = deepcopy(_c)
    configs.immutable(False)

    # Update default configs with user provided ones
    for arg, value in kwargs.items():
        configs[arg] = value

    validate_configs(configs)
    configs.immutable(True)

    return configs
