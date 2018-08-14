
[![api-source](https://img.shields.io/badge/api-source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/siamese.py)

## `torch_collections.models.siamese.Siamese`

The `Siamese` network is designed for triplet and contrastive loss, popularized by the task of facial recognition. The features of this network is mostly implemented based on [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) as well as a series of other popular papers which proposes improvements to the original `FaceNet` model.

A notable omission is [A Discriminative Deep Feature Learning Approach for Face Recognition](https://ydwen.github.io/papers/WenECCV16.pdf) mainly due to the massive differences in expected outputs as well as training procedures.

This page serves as a documentation for the various functionalities of this implementation of `Siamese` network.

On a side note, the `Siamese` network has not been tested yet. There is still tuning to be done to optimize default values.

<br>


## API reference

The `Siamese` network is a `torch.nn.Module` object and share all of it's functions.

- [`Siamese.__init__`](#siamese__init__kwargs)
- [`Siamese.forward`](#siameseforwardimage)
- [`Siamese.configs`](#siameseconfigs)
- [`Siamese.dist_fn`](#siamesedist_fn)
- [`Siamese.pdist_fn`](#siamesepdist_fn)
- [`Siamese.triplet_loss`](#siamesetriplet_loss)
- [`Siamese.contrastive_loss`](#siamesecontrastive_loss)

<br>


### `Siamese.__init__(**kwargs)` [![source](https://img.shields.io/badge/source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/siamese.py#L26)

| Arguments | Descriptions |
| --- | --- |
| `input_size` | `type: list` `default: [160, 160]` <br> The expected height and width of input image tensors. |
| `embedding_size` | `type: int` `default: 128` <br> The size of the embedded features. |
| `backbone` | `type: string` `default: 'resnet50'` `option: ['resnet18', 'resnet34', ...]` <br> The backbone to be used in the RetinaNet model. Only resnet backbones has been implemented so far. |
| `freeze_backbone` | `type: bool` `default: False` <br> The flag to raise if the backbone model should be frozen. |
| `l2_norm_alpha` | `type: float` `default: 10.0` <br> The alpha to apply to the normalized embeddings https://arxiv.org/pdf/1703.09507.pdf. |
| `margin` | `type: float` `default: 0.5` <br> The margin to use for both triplet and contrastive loss |
| `dist_type` | `type: string` `default: 'euclidean'` `option: ['euclidean', 'cosine']` <br> The distance function to utilize. |
| `p_norm` | `type: float` `default: 2.0` <br> The normalization degree in euclidean distance. |

<br>


### `Siamese.forward(image)` [![source](https://img.shields.io/badge/source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/siamese.py#L90)

There is no difference in training and evaluation of the `Siamese` network.

| Arguments | Descriptions |
| --- | --- |
| `image` | `type: tensor` <br> The input image tensor formatted to NCHW and normalized to pytorch standard. |

| Returns | Descriptions |
| --- | --- |
| `embeddings` | `type: tensor` <br> The output embedding tensor of the shape (batch_size, embedding_size). |

<br>


### `Siamese.dist_fn(A, B)`
[![source](https://img.shields.io/badge/source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/_siamese.py#L31)

Computes the difference between two sets of embeddings, A and B

| Arguments | Descriptions |
| --- | --- |
| `A, B` | `type: tensor` <br> Embedding tensors of the shape (batch_size, embedding_size). |

| Returns | Descriptions |
| --- | --- |
| `distances` | `type: tensor` <br> A distance tensor of the shape(batch_size,). The nth element in this tensor will represent the distance between nth corresponding embeddings in each of A and B. |

<br>


### `Siamese.pdist_fn(A)`
[![source](https://img.shields.io/badge/source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/_siamese.py#L4)

Computes the difference between every possible pair of embeddings and collate in a matrix.

| Arguments | Descriptions |
| --- | --- |
| `A` | `type: tensor` <br> Embedding tensor of the shape (batch_size, embedding_size). |

| Returns | Descriptions |
| --- | --- |
| `distance_matrix` | `type: tensor` <br> A distance_matrix tensor of the shape(batch_size, batch_size). The element in the [i, j] position represents the distance between the A[i] and A[j] embedding. |

<br>


### `Siamese.triplet_loss(anchor, positive, negative)`
[![source](https://img.shields.io/badge/source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/_siamese.py#L49)

Computes the triplet loss for a given set of anchor, positive and negative embeddings.

| Arguments | Descriptions |
| --- | --- |
| `anchor, positive, negative` | `type: tensor` <br> Embedding tensors of the 3 images. Each tensor is of the shape (batch_size, embedding_size). |

| Returns | Descriptions |
| --- | --- |
| `loss` | `type: tensor` <br> A single tensor indicating the loss for the triplet batch. |

<br>


### `Siamese.contrastive_loss(A, B, target)`
[![source](https://img.shields.io/badge/source-blue.svg)](https://github.com/mingruimingrui/torch-collections/blob/master/torch_collections/models/_siamese.py#L62)

Computes the contrastive loss for a given set of embeddings and target.

| Arguments | Descriptions |
| --- | --- |
| `A, B` | `type: tensor` <br> Embedding tensors of the 2 images. Each tensor is of the shape (batch_size, embedding_size). |
| `target` | `type: tensor` <br> Indicator for each image pair. 1 for same class, 0 for different class. |

| Returns | Descriptions |
| --- | --- |
| `loss` | `type: tensor` <br> A single tensor indicating the loss for the pair wise batch. |

<br>
