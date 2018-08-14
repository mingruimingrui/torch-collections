<!-- Insert examples in the future -->

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
- [`RetinaNet.configs`](#)

<br>

### `RetinaNet.__init__(num_classes, **kwargs)`

| Arguments | Descriptions |
| --- | --- |
| `num_classes (required)` | `int` <br> The number of classes this model is expected to detect. |
| `backbone` | `string` `default: 'resnet50'` `option: ['resnet18', 'resnet34', ...]` <br> The backbone to be used in the RetinaNet model. Only resnet backbones has been implemented so far. |
| `anchor_sizes` | `list` `default: [32, 64, 128, 256, 512]` <br> The sizes at which anchors should be generated at each feature level. |
| `anchor_strides` | `list` `default: [8, 16, 32, 64, 128]` <br> The strides at which anchors should be generated at each feature level. |
| `anchor_ratios` | `list` `default: [0.5, 1, 2]` <br> The ratios at which anchors should be generated at each moving window. |
| `anchor_scales` | `list` `default: [2 ** 0, 2 ** (1/3), 2 ** (2/3)]` <br> The scales at which anchors should be generated at each moving window. |
| `pyramid_feature_size` | `int` `default: 256` <br> The channel size of features output by the FPN. |
| `regression_block_type` | `string` `default: 'fc'` `option: ['fc', 'dense']` <br> The type of regression model to use. |
| `regression_num_layers` | `int` `default: 4` <br> The number of layers in the regression model. |
| `regression_feature_size` | `int` `default: 256` <br> The internal channel size of the regression model (only for `'fc'`). |
| `regression_growth_rate` | `int` `default: 64` <br> The channel growth rate of the regression model (only for `'dense'`). |
| `classification_block_type` | `string` `default: 'fc'` `option: ['fc', 'dense']` <br> The type of classification model to use. |
| `classification_num_layers` | `int` `default: 4` <br> The number of layers in the classification model. |
| `classification_feature_size` | `int` `default: 256` <br> The internal channel size of the classification model (only for `'fc'`). |
| `classification_growth_rate` | `int` `default: 64` <br> The channel growth rate of the classification model (only for `'dense'`). |

<br>









### `RetinaNet.forward(image, annotations=None)` [![source](https://img.shields.io/badge/source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/retinanet.py#L116)

**Args**

`annotations` will not be needed for evaluation.

> `image :tensor:`
The input image tensor formatted to NCHW and normalized to pytorch standard

> `annotations :list of tensor: (training only)`
A list of annotations, there should be N annotations (batch_size) in this list.
Each annotation is a tensor in the shape (num_detections, 5),
where each detection should be in the format (x1, y2, x2, y2, class_id).
As annotations cannot be expected to have similar shapes, they have to be stored in a list

**Returns**

The returning item will be different for training and evaluation

*training*
> `loss :tensor:`
The mean loss of this batch, ready for backprop.

*evaluation*
> `detections :list:`
A list of length N (batch_size).
Each entry is a dictionary in following format
```
{
  'boxes'  : A tensor of the shape (num_detections, 4) where each box is in the (x1, y1, x2, y2) format
  'labels' : A tensor of the shape (num_detections,) representing the individual class_id of each detection
  'scores' : A tensor of the shape (num_detections,) representing the confidence score of each detection
}
```
