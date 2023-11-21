# Why Look at Every Frame in Video Instance Segmentation
This is the official implementation of TRN on MinVIS. 

<div align="center">
  <img src=framework.jpg width="100%" height="100%"/>
</div>

### Features
* Temporal Reduce Network to Speed Up Video Instance Segmentation models.
* Support major video instance segmentation datasets: YouTubeVIS 2019/2021, Occluded VIS (OVIS).

## Installation

See [installation instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for TRN](datasets/README.md).

See [Getting Started with TRN](GETTING_STARTED.md).

## Model Zoo

Trained models are available for download in the [TRN Model Zoo](MODEL_ZOO.md).

## License

The majority of TRN is made available under the [Nvidia Source Code License-NC](LICENSE). The trained models in the [TRN Model Zoo](MODEL_ZOO.md) are made available under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

Portions of the project are available under separate license terms: Mask2Former is licensed under a [MIT License](https://github.com/facebookresearch/Mask2Former/blob/main/LICENSE). Swin-Transformer-Semantic-Segmentation is licensed under the [MIT License](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/LICENSE), Deformable-DETR is licensed under the [Apache-2.0 License](https://github.com/fundamentalvision/Deformable-DETR/blob/main/LICENSE).

## Acknowledgement

This repo is largely based on Mask2Former (https://github.com/facebookresearch/Mask2Former) and MinVIS (https://github.com/NVlabs/MinVIS).
