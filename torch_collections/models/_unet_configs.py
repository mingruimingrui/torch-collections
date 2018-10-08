from __future__ import division
from copy import deepcopy

from ..utils.collections import AttrDict

# Define default parameters
_c = AttrDict()

################################################################################
#### Start of configurable parameters

#### Model configs
_c.name        = 'unet'
_c.num_classes = None

_c.backbone        = 'resnet18'
_c.freeze_backbone = False

################################################################################
#### End of configurable parameters

# Set default configs to be immutable
_c.immutable(True)

def validate_configs(configs):
    assert isinstance(configs.num_classes, int), 'num_classes must be specified'
    assert 'resnet' in configs.backbone, 'Currently only resnet has been implemented for backbone'

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
