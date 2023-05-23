# TRN Model Zoo

## Introduction

This file documents a collection of trained TRN models.
The numbers in this page are for the specific checkpoints and are different from the paper, which are averaged from multiple runs. The "Config" column contains a link to the config file. Running `train_net_video.py --num-gpus $num_gpus` with this config file will train a model with the same setting. ResNet-50 results are trained with 8 GPUs.

## Video Instance Segmentation
### YouTubeVIS 2019

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Mode</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">FPS</th>
<th valign="bottom">AP</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: R50 -->
 <tr><td align="center">MinVIS</td>
<td align="center">R50</td>
<td align="center">47.1</td>
<td align="center">108</td>
<td align="center"><a href="configs/youtubevis_2019/minvis_video_maskformer2_R50_bs32_8ep_frame.yaml">yaml</a></td>
<td align="center">None</a></td>
</tr>
 <tr><td align="center">TRN (B)</td>
<td align="center">R50</td>
<td align="center">141</td>
<td align="center">47.1</td>
<td align="center"><a href="configs/youtubevis_2019/trn_balance_video_maskformer2_R50_bs32_8ep_frame.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1-SQ9Oixvybpp2QDWxZOhqsHCCCXJY9J2/view?usp=sharing">model</a></td>
</tr>
 <tr><td align="center">TRN (T)</td>
<td align="center">R50</td>
<td align="center">200</td>
<td align="center">44.1</td>
<td align="center"><a href="configs/youtubevis_2019/trn_turbo_video_maskformer2_R50_bs32_8ep_frame.yaml">yaml</a></td>
<td align="center">the same as TRN (B)</a></td>
</tr>
<!-- ROW: Swin-T -->
 <tr><td align="center">MinVIS</td>
<td align="center">Swin-T</td>
<td align="center">52.2</td>
<td align="center">93</td>
<td align="center"><a href="configs/youtubevis_2019/minvis_video_maskformer2_R50_bs32_8ep_frame.yaml">yaml</a></td>
<td align="center">None</a></td>
</tr>
 <tr><td align="center">TRN (B)</td>
<td align="center">Swin-T</td>
<td align="center">114</td>
<td align="center">52.5</td>
<td align="center"><a href="configs/youtubevis_2019/swin/trn_balance_video_maskformer2_swin_tiny.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1lyrKS00ON0Ly7r062eB9IQgJdmDURfKo/view?usp=sharing">model</a></td>
</tr>
 <tr><td align="center">TRN (T)</td>
<td align="center">Swin-T</td>
<td align="center">151</td>
<td align="center">47.4</td>
<td align="center"><a href="configs/youtubevis_2019/swin/trn_turbo_video_maskformer2_swin_tiny.yaml">yaml</a></td>
<td align="center">the same as TRN (B)</a></td>
</tr>
</tbody></table>
