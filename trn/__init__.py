# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.

# config
from .config import add_minvis_config

# models
from .video_maskformer_model import VideoMaskFormer_frame
from .tarvis_video_model import VideoMaskFormer_frame_tarvis
from .mem_video_model import VideoMaskFormer_frame_mem
from .video_mask2former_transformer_decoder import VideoMultiScaleMaskedTransformerDecoder_frame
from .tarvis_video_mask2former_transformer_decoder import VideoMultiScaleMaskedTransformerDecoder_frame_tarvis
from .mem_video_mask2former_transformer_decoder import VideoMultiScaleMaskedTransformerDecoder_frame_mem
from .share_mask_fpn import MSSharePixelDecoder
from .share_mask_former_head import ShareMaskFormerHead
from .share_video_maskformer_model import ShareVideoMaskFormer_frame
from .share_video_mask2former_transformer_decoder import ShareVideoMultiScaleMaskedTransformerDecoder_frame

# video
from .data_video import (
    YTVISDatasetMapper,
    YTVISEvaluator,
    build_detection_train_loader,
    build_detection_test_loader,
    get_detection_dataset_dicts,
)
