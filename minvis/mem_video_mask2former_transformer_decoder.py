# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine
from mask2former_video.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine3D

from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import VideoMultiScaleMaskedTransformerDecoder, CrossAttentionLayer, SelfAttentionLayer
import einops


@TRANSFORMER_DECODER_REGISTRY.register()
class VideoMultiScaleMaskedTransformerDecoder_frame_mem(VideoMultiScaleMaskedTransformerDecoder):

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        # video related
        num_frames,
    ):
        super().__init__(
            in_channels=in_channels,
            mask_classification=mask_classification,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            pre_norm=pre_norm,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            num_frames=num_frames,
        )

        # use 2D positional embedding
        N_steps = hidden_dim // 2
        #self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.pe_layer = PositionEmbeddingSine3D(N_steps, normalize=True)


        '''
        self.pre_memory_embed_k = nn.Linear(hidden_dim, hidden_dim)
        self.pre_memory_embed_v = nn.Linear(hidden_dim, hidden_dim)

        self.pre_query_embed_k = nn.Linear(hidden_dim, hidden_dim)
        self.pre_query_embed_v = nn.Linear(hidden_dim, hidden_dim)

        self.pre_attn = CrossAttentionLayer(
                            d_model=hidden_dim,
                            nhead=nheads,
                            dropout=0.0,
                            normalize_before=pre_norm,
                        )

        # learnable query p.e.
        self.mem_query_embed = nn.Embedding(num_queries, 2 * hidden_dim)

        self.pre_attn = SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
        '''

    def forward(self, x, mask_features, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        bt, c_m, h_m, w_m = mask_features.shape
        t = 2
        pred_add = bt % t
        if bt % t == 0:
            bs = bt // t
        else:
            bs = bt // t + 1
            repeat_mask_features = mask_features[-t:-pred_add]
            mask_features = torch.cat((mask_features, repeat_mask_features))

        if self.training:
            nc = self.num_frames // t
        else:
            nc = bs

        mask_features = mask_features.view(bs, t, c_m, h_m, w_m)

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            if pred_add:
                repeat_x = x[i][-t:-pred_add]
                add_x = torch.cat((x[i], repeat_x))
            else:
                add_x = x[i]
            size_list.append(add_x.shape[-2:])
            pos.append(self.pe_layer(add_x.view(bs, t, -1, size_list[-1][0], size_list[-1][1]), None).flatten(3))
            src.append(self.input_proj[i](add_x).flatten(2) + self.level_embed.weight[i][None, :, None])

            # NTxCxHW => NxTxCxHW => (TxHW)xNxC
            _, c, hw = src[-1].shape
            pos[-1] = pos[-1].view(bs, t, c, hw).permute(1, 3, 0, 2).flatten(0, 1)
            src[-1] = src[-1].view(bs, t, c, hw).permute(1, 3, 0, 2).flatten(0, 1)

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        #mem_query_embed = self.mem_query_embed.weight.view(100, 2, c)

        pre_memory = None
        if pre_memory is not None:
            output = self.pre_attn(
                output, pre_m,
                memory_mask=None,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=query_embed, query_pos=query_embed
            )

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output = self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            '''
            for j in range(output.shape[1]-1):
                output[:, j:j+2] = self.pre_attn(
                        output[:, j:j+2].transpose(0, 1), tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=mem_query_embed.transpose(0,1)
                ).transpose(0, 1)
            '''

            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.num_layers + 1

        # expand BT to B, T
        for i in range(len(predictions_mask)):
            if self.training:
                predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b nc) q ct h w -> b q (nc ct) h w', nc=nc)
            else:
                predictions_mask[i] = einops.rearrange(predictions_mask[i], '(b nc) q ct h w -> b q nc ct h w', nc=nc)

        for i in range(len(predictions_class)):
            predictions_class[i] = einops.rearrange(predictions_class[i], '(b nc) q c -> b nc q c', nc=nc)
            if self.training:
                predictions_class[i] = einops.repeat(predictions_class[i], 'b nc q c -> b (repeat nc) q c', repeat=2)

        pred_embds = self.decoder_norm(output)

        pred_embds = einops.rearrange(pred_embds, 'q (b nc) c -> b c nc q', nc=nc)

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            'pred_embds': pred_embds,
            'pred_add': pred_add,
        }

        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,btchw->bqthw", mask_embed, mask_features)
        b, q, t, _, _ = outputs_mask.shape

        # NOTE: prediction is of higher-resolution
        # [B, Q, T, H, W] -> [B, Q, T*H*W] -> [B, h, Q, T*H*W] -> [B*h, Q, T*HW]
        attn_mask = F.interpolate(outputs_mask.flatten(0, 1), size=attn_mask_target_size, mode="bilinear", align_corners=False).view(
            b, q, t, attn_mask_target_size[0], attn_mask_target_size[1])
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask
