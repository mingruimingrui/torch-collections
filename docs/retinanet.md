
[source](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/retinanet.py)

## torch_collections.models.RetinaNet

The `RetinaNet` is a state of the art object detection model, implemented based on [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).

A pretrained model trained on the coco dataset can be downloaded from this repository's [release page](https://github.com/mingruimingrui/torch-collections/releases).

This page serves as a documentation for the various functionalities of this implementation of `RetinaNet`.


`RetinaNet.forward(image, annotations=None)` [source](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/retinanet.py#L116)

**Args**

`annotations` will not be needed for evaluation.

> `image :tensor:` The input image tensor formatted to NCHW and normalized to pytorch standard

> `annotations :list of tensor: (training only)`
A list of annotations, there should be N annotations (batch_size) in this list.
Each annotation is a tensor in the shape (num_detections, 5),
where each detection should be in the format (x1, y2, x2, y2, class_id).
As annotations cannot be expected to have similar shapes, they have to be stored in a list

**Returns**

The returning item will be different for training and evaluation

*training*
> `loss :tensor:` The mean loss of this batch.

*evaluation*
> `detections :list:` A list of length N (batch_size).
Each entry is a dictionary in following format
```
{
  'boxes'  : A tensor of the shape (num_detections, 4) where each box is in the (x1, y1, x2, y2) format
  'labels' : A tensor of the shape (num_detections,) representing the individual class_id of each detection
  'scores' : A tensor of the shape (num_detections,) representing the confidence score of each detection
}
```


`RetinaNet.__init__(num_classes, **kwargs)` [source](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/retinanet.py#L29)

All valid kwargs are listed below.

> `num_classes :int:` The number of classes the RetinaNet model is expected to detect
