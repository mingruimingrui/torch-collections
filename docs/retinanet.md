<!-- Insert examples in the future -->

[![api-source](https://img.shields.io/badge/api-source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/retinanet.py)

## `class torch_collections.models.retinanet.RetinaNet(torch.nn.Module)`

The `RetinaNet` is a state of the art object detection model, implemented based on [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).

A pretrained model trained on the coco dataset can be downloaded from this repository's [release page](https://github.com/mingruimingrui/torch-collections/releases).

This page serves as a documentation for the various functionalities of this implementation of `RetinaNet`.

On a side note, this implementation of `RetinaNet` only supports feature levels of 3 - 7. There are no short term plans to add this feature.

<br>

## API reference

The `RetinaNet` is a `torch.nn.Module` object and share all of it's functions.

- [`RetinaNet.__init__`](#)
- [`RetinaNet.forward`](#)
- [`RetinaNet.configs`](#)

<br>

### `RetinaNet.__init__(num_classes, **kwargs)`

| Arguments | Descriptions |
| --- | --- |
| `num_classes (required)` | `int` The number of classes this model is expected to detect. |
| `backbone` | `string` The backbone to be used in the RetinaNet model. Only resnet backbones has been implemented so far. Default of `'resnet50'`. Option of `['resnet18', 'resnet34', ...]`. |
| `anchor_sizes` | `list` The sizes at which anchors should be generated at each feature level. Default of `[32, 64, 128, 256, 512]`. |
| `anchor_strides` | `list` The strides at which anchors should be generated at each feature level. Default of `[8, 16, 32, 64, 128]`. |
| `anchor_ratios` | `list` The ratios at which anchors should be generated at each moving window. Default of `[0.5, 1, 2]`. |
| `anchor_scales` | `list` The scales at which anchors should be generated at each moving window. Default of `[2 ** 0, 2 ** (1/3), 2 ** (2/3)]`. |
| `pyramid_feature_size` | `int` The channel size of features output by the FPN. Default of `256`. |
| `regression_block_type` | `string` The type of regression model to use. Default of `'fc'`. Option of `['fc', 'dense']`. |
| `regression_num_layers` | `int` The number of layers in the regression model. Default of `4`. |
| `regression_feature_size` | `int` The internal channel size of the regression model (only for `'fc'`). Default of `256`. |
| `regression_growth_rate` | `int` The channel growth rate of the regression model (only for `'dense'`). Default of `64`. |
| `classification_block_type` | `string` The type of classification model to use. Default of `'fc'`. Option of `['fc', 'dense']`. |
| `classification_num_layers` | `int` The number of layers in the classification model. Default of `4`. |
| `classification_feature_size` | `int` The internal channel size of the classification model (only for `'fc'`). Default of `256`. |
| `classification_growth_rate` | `int` The channel growth rate of the classification model (only for `'dense'`). Default of `64`. |

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
