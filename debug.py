
import time
import numpy as np

import torch
from torchsummary import summary

from torch_collections.models.retinanet import RetinaNet

backbone = 'resnet18'
retinanet = RetinaNet(backbone=backbone, num_classes=80)

# inputs = np.random.rand(1, 3, 224, 224).astype('float32')
inputs = np.random.rand(1, 3, 224, 224).astype('float32') * 0
inputs = torch.from_numpy(inputs)

t1 = time.time()
outputs = retinanet(inputs)
print('{:.2f}'.format(time.time() - t1))

print(backbone)

import pdb; pdb.set_trace()
# print([np.array(o.shape) for o in outputs])
# print(outputs[0][0, 0])

# torch.save(retinanet, 'retinanet_resnet18.pth')
