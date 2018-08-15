from __future__ import division
from copy import deepcopy

from ..utils.collections import AttrDict

# Define default parameters
_c = AttrDict()

################################################################################
#### Start of configurable parameters

#### Model configs
_c.name        = 'retinanet'
_c.num_classes = None

_c.backbone        = 'resnet50'
_c.freeze_backbone = False  # Does not actually work at the moment will implement in future

_c.anchor_sizes   = [32, 64, 128, 256, 512]
_c.anchor_strides = [8, 16, 32, 64, 128]
_c.anchor_ratios  = [0.5, 1., 2.]
_c.anchor_scales  = [2. ** 0., 2. ** (1. / 3.), 2. ** (2. / 3.)]

_c.pyramid_feature_size = 256

_c.regression_block_type   = 'fc'  # one of ['fc', 'dense']
_c.regression_num_layers   = 4
_c.regression_feature_size = 256  # Regression model internal channel size (only for 'fc')
_c.regression_growth_rate  = 64   # Regression model channel growth rate (only for 'dense')

_c.classification_block_type   = 'fc'  # one of ['fc', 'dense']
_c.classification_num_layers   = 4
_c.classification_feature_size = 256  # Classification model internal channel size (only for 'fc')
_c.classification_growth_rate  = 64   # Classification model channel growth rate (only for 'dense')

_c.nms_threshold   = 0.5
_c.score_threshold = 0.05
_c.max_detections  = 300

################################################################################
#### End of configurable parameters

# Set default configs to be immutable
_c.immutable(True)

def validate_configs(configs):
    assert isinstance(configs.num_classes, int), 'num_classes must be specified'
    configs.num_anchors = len(configs.anchor_ratios) * len(configs.anchor_scales)

def make_configs(**kwargs):
    configs = deepcopy(_c)
    configs.immutable(False)

    # Update default configs with user provided ones
    for arg, value in kwargs.items():
        configs[arg] = value

    # Validate
    validate_configs(configs)

    configs.immutable(True)

    return configs
