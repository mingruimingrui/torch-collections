from __future__ import division
from copy import deepcopy

from ..utils.collections import AttrDict

# Define default parameters
_c = AttrDict()

################################################################################
#### Start of configurable parameters

#### Model configs
_c.name           = 'encoder'
_c.input_size     = [224, 224]
_c.embedding_size = 256

_c.backbone        = 'resnet18'
_c.freeze_backbone = False

################################################################################
#### End of configurable parameters

# Set default configs to be immutable
_c.immutable(True)

def make_configs(**kwargs):
    configs = deepcopy(_c)
    configs.immutable(False)

    # Update default configs with user provided ones
    for arg, value in kwargs.items():
        configs[arg] = value

    configs.immutable(True)

    return configs
