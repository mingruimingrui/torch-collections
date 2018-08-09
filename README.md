
__torch-collections__ is a collection of popular deep learning models that uses the torch high-level APIs. These models are intended for fast iterative train-test cycles.

Most models are implemented based on their papers but additional options are inserted to improve customizability and the codebase are optimized for fast performance. Some pretrained models are provided for the popular open-source datasets.

## Common Attributes
Each model comes with a custom config file
(eg.
  [retinanet.py](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/retinanet.py) and
  [\_retinanet_configs.py](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/_retinanet_configs.py)
)

These config files contain the customizable parameters for the model which can be defined upon initialization by using keyword arguments.
(eg. `RetinaNet(num_classes=80, backbone='resnet101')`) Configs can be retrieved using the `model.configs` attribute and will be stored in an immutable `AttrDict`.

These are mainly measures taken to prevent the edition of model parameters post initialization and during training as such actions are highly inadvisable.

## List of models
- [RetinaNet](https://github.com/mingruimingrui/torch-collections/blob/master/docs/reinanet.md)

## License
[MIT License](https://github.com/mingruimingrui/torch-collections/blob/master/LICENSE)
