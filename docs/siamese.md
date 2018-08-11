
[![api-source](https://img.shields.io/badge/api-source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/siamese.py)

## `torch_collections.models.siamese.Siamese`

The `Siamese` network is designed for triplet and contrastive loss, popularized by the task of facial recognition. The features of this network is mostly implemented based on [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) as well as a series of other popular papers which proposes improvements to the original `FaceNet` model.

A notable omission is [A Discriminative Deep Feature Learning Approach for Face Recognition](https://ydwen.github.io/papers/WenECCV16.pdf) mainly due to the massive differences in expected outputs as well as training procedures.

This page serves as a documentation for the various functionalities of this implementation of `Siamese` network.

<br>

### `Siamese.__init__(**kwargs)` [![source](https://img.shields.io/badge/source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/siamese.py#L26)

All valid kwargs are listed below.

> `input_size :list: default [160, 160]`
The expected height and width of input image tensors

> `embedding_size :int: default 128`
The size of the embedded features

> `backbone :string: default 'resnet50'`
The backbone to use in the RetinaNet option of `['resnet18', 'resnet34', 'resnet50', ...]`,
only resnet backbones have been implemented so far.

> `freeze_backbone :bool: default False`
The flag to raise if the backbone model should be frozen

> `l2_norm_alpha :float: default 10.0`
The alpha to apply to the normalized embeddings https://arxiv.org/pdf/1703.09507.pdf

> `margin :float: default 0.5`
The margin to use for both triplet and contrastive loss

> `dist_type :string: default "euclidean"`
Option of `['euclidean', 'cosine']`, The distance function to utilize

> `p_norm :float: default 2.0`
The normalization degree in euclidean distance,

<br>

### `Siamese.forward(image)` [![source](https://img.shields.io/badge/source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/siamese.py#L90)

There is no difference in training and evaluation of the `Siamese` network.

**Args**

> `image :tensor:`
The input image tensor formatted to NCHW and normalized to pytorch standard

**Returns**

> `embeddings :tensor:`
The output embedding tensor of the shape (batch_size, embedding_size)

<br>

### `Siamese.dist_fn(A, B)`
[![source](https://img.shields.io/badge/source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/_siamese.py#L31)

Computes the difference between two sets of embeddings, A and B

**Args**

> `A and B :tensor:`
Embedding tensors of the shape (batch_size, embedding_size).

**Returns**

> `distances :tensor:`
A distance tensor of the shape(batch_size,). The nth element in this tensor will represent
the distance between nth corresponding embeddings in each of A and B.

<br>

### `Siamese.pdist_fn(A)`
[![source](https://img.shields.io/badge/source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/_siamese.py#L4)

Computes the difference between every possible pair of embeddings and collate in a matrix.

**Args**

> `A :tensor:`
Embedding tensor of the shape (batch_size, embedding_size).

**Returns**

> `distance_matrix :tensor:`
A distance_matrix tensor of the shape(batch_size, batch_size).
The element in the [i, j] position represents the distance between the A[i]
and A[j] embedding

<br>

### `Siamese.triplet_loss(anchor, positive, negative)`
[![source](https://img.shields.io/badge/source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/_siamese.py#L49)

Computes the triplet loss for a given set of anchor, positive and negative embeddings.

**Args**

> `anchor, positive, negative :tensor:`
Embedding tensors of the 3 images. Each tensor is of the shape (batch_size, embedding_size)

**Returns**

> `loss :tensor:`
A single tensor indicating the loss for the triplet batch.

<br>

### `Siamese.contrastive_loss(A, B, target)`
[![source](https://img.shields.io/badge/source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/_siamese.py#L62)

Computes the contrastive loss for a given set of embeddings and target.

**Args**

> `A, B :tensor:`
Embedding tensors of the 2 images. Each tensor is of the shape (batch_size, embedding_size)

> `target :tensor:`
Indicator for each image pair. 1 for same class, 0 for different class

**Returns**

> `loss :tensor:`
A single tensor indicating the loss for the pair wise batch.

<br>

## Notes
At the moment the `Siamese` network has not been tested yet. There is still tuning to be done to optimize default values.
