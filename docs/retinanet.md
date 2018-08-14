<p align="center">
  <img src="https://github.com/mingruimingrui/torch-collections/blob/feature/improve_docs/images/retinanet_example_1.jpg" height="400px"/>
</p>

[![api-source](https://img.shields.io/badge/api-source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/retinanet.py)

## `torch_collections.models.retinanet.RetinaNet`

The `RetinaNet` is a state of the art object detection model, implemented based on [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).

A pretrained model trained on the coco dataset can be downloaded from this repository's [release page](https://github.com/mingruimingrui/torch-collections/releases).

This page serves as a documentation for the various functionalities of this implementation of `RetinaNet`.

On a side note, this implementation of `RetinaNet` only supports feature levels of 3 - 7. There are no short term plans to add this feature.

<br>


## API reference

The `RetinaNet` is a `torch.nn.Module` object and share all of it's functions.

- [`RetinaNet.__init__`](#retinanet__init__num_classes-kwargs)
- [`RetinaNet.forward`](#retinanetforwardimage-annotationsnone-)
- [`RetinaNet.configs`](#retinanetconfigs)

<br>


### `RetinaNet.__init__(num_classes, **kwargs)`

The initialization of the `RetinaNet` will store all parameters in `RetinaNet.configs` as a dictionary. Reference to it for the model settings.

| Arguments | Descriptions |
| --- | --- |
| `num_classes (required)` | `type: int` <br> The number of classes this model is expected to detect. |
| `backbone` | `type: string` `default: 'resnet50'` `option: ['resnet18', 'resnet34', ...]` <br> The backbone to be used in the RetinaNet model. Only resnet backbones has been implemented so far. |
| `anchor_sizes` | `type: list` `default: [32, 64, 128, 256, 512]` <br> The sizes at which anchors should be generated at each feature level. |
| `anchor_strides` | `type: list` `default: [8, 16, 32, 64, 128]` <br> The strides at which anchors should be generated at each feature level. |
| `anchor_ratios` | `type: list` `default: [0.5, 1, 2]` <br> The ratios at which anchors should be generated at each moving window. |
| `anchor_scales` | `type: list` `default: [2 ** 0, 2 ** (1/3), 2 ** (2/3)]` <br> The scales at which anchors should be generated at each moving window. |
| `pyramid_feature_size` | `type: int` `default: 256` <br> The channel size of features output by the FPN. |
| `regression_block_type` | `type: string` `default: 'fc'` `option: ['fc', 'dense']` <br> The type of regression model to use. |
| `regression_num_layers` | `type: int` `default: 4` <br> The number of layers in the regression model. |
| `regression_feature_size` | `type: int` `default: 256` <br> The internal channel size of the regression model (only for `'fc'`). |
| `regression_growth_rate` | `type: int` `default: 64` <br> The channel growth rate of the regression model (only for `'dense'`). |
| `classification_block_type` | `type: string` `default: 'fc'` `option: ['fc', 'dense']` <br> The type of classification model to use. |
| `classification_num_layers` | `type: int` `default: 4` <br> The number of layers in the classification model. |
| `classification_feature_size` | `type: int` `default: 256` <br> The internal channel size of the classification model (only for `'fc'`). |
| `classification_growth_rate` | `type: int` `default: 64` <br> The channel growth rate of the classification model (only for `'dense'`). |

<br>


### `RetinaNet.forward(image, annotations=None)`

The inference function expects both `image` and `annotations` during training. However only `image` is required for evaluation.
Do note that the outputs for training and evaluation mode are also different.

| Arguments | Descriptions |
| --- | --- |
| `image` | `type: tensor` <br> The input image tensor formatted to NCHW and normalized to pytorch standard |
| `annotations` <br> *training only* | `type: list` <br> A list of annotations, there should be N (batch_size) annotations in this list. <br> Each annotation is a tensor in the shape (num_detections, 5), where each detection should be in the format (x1, y2, x2, y2, class_id). <br> As annotations cannot be expected to have similar shapes, they have to be stored in a list. <br> This variable is only needed for training. |

| Returns | Descriptions |
| --- | --- |
| `loss` <br> *training only* | `type: tensor` <br> The mean loss of this batch. Backprop ready. |
| `detections` <br> *evaluation only* | `type: list` <br> A list of length N (batch_size). Each entry is a dictionary in the format shown below. |
```
{
  'boxes'  : A tensor of the shape (num_detections, 4) where each box is in the (x1, y1, x2, y2) format
  'labels' : A tensor of the shape (num_detections,) representing the individual class_id of each detection
  'scores' : A tensor of the shape (num_detections,) representing the confidence score of each detection
}
```

<br>


### `RetinaNet.configs`

An `AttrDict` containing all the parameters called upon initialization. The `AttrDict` is immutable which is done to ensure that model parameters are not changed accidentally during training.
