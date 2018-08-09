
[source](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/retinanet.py)

## torch_collections.models.RetinaNet

The `RetinaNet` is a state of the art object detection model, implemented based on [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).

A pretrained model trained on the coco dataset can be downloaded from this repository's [release page](https://github.com/mingruimingrui/torch-collections/releases).

This page serves as a documentation for the various functionalities of this implementation of `RetinaNet`.

`RetinaNet.__init__(num_classes, **kwargs)` [source](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/retinanet.py#L29)
> The model expects at least the num_classes argument. All other customization to the `RetinaNet` model can be viewed below

`RetinaNet.forward(batch)` [source](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/retinanet.py#L116)
> The `RetinaNet` model expects a dict-like object `batch` as an input.
> Do take note that the inputs and outputs expected during training and evaluation are different

> **Training** a training batch must be a sample in the following format
```
batch = {
  'image' : The input image tensor formatted to NCHW and normalized to pytorch standard,
  'annotations' : A list of annotations, there should be N annotations (batch size) in this list
                  Each annotation is a tensor in the shape (Number of detections, 5)
                  Where each detection should be in the following format (x1, y1, x2, y2, class_id)
                  As annotations cannot be expected to have similar shapes, they have to be stored in a list
}
```

> **Evaluation** an evaluation batch must be a sample in the following format
```
batch = {
  'image' : The input image tensor formatted to NCHW and normalized to pytorch standard
}
```



## customizable options
