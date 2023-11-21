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
<td align="center">110</td>
<td align="center">47.1</td>
<td align="center">None</a></td>
<td align="center">None</a></td>
</tr>
 <tr><td align="center">TRN-Lite</td>
<td align="center">R50</td>
<td align="center">123</td>
<td align="center">48.4</td>
<td align="center"><a href="configs/youtubevis_2019/trnlite_minvis_R50_bs32_8ep.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1mdrL6QRVmoz-QizohZ7SnyGDlpAWFCVf/view?usp=sharing">model</a></td>
</tr>
 <tr><td align="center">TRN</td>
<td align="center">R50</td>
<td align="center">142</td>
<td align="center">46.7</td>
<td align="center"><a href="configs/youtubevis_2019/trn_minvis_R50_bs32_8ep.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1-SQ9Oixvybpp2QDWxZOhqsHCCCXJY9J2/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: Swin-L -->
 <tr><td align="center">MinVIS</td>
<td align="center">Swin-L</td>
<td align="center">43</td>
<td align="center">60.9</td>
<td align="center">None</a></td>
<td align="center">None</a></td>
</tr>
 <tr><td align="center">TRN-Lite</td>
<td align="center">Swin-L</td>
<td align="center">48</td>
<td align="center">60.7</td>
<td align="center"><a href="configs/youtubevis_2019/swin/trnlite_minvis_swin_large_bs16_8ep.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/12yL72Qv8OBqapgvGmLHXAb6YZv4SS_6D/view?usp=sharing">model</a></td>
</tr>
 <tr><td align="center">TRN</td>
<td align="center">Swin-L</td>
<td align="center">52</td>
<td align="center">59.4</td>
<td align="center"><a href="configs/youtubevis_2019/swin/trn_minvis_swin_large_bs16_8ep.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/18xxnf_5D0R5PajsXOU5AMsjNSfLnsJOY/view?usp=sharing">model</a></td>
</tr>
</tbody></table>

### YouTubeVIS 2021

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
<td align="center">110</td>
<td align="center">44.3</td>
<td align="center">None</a></td>
<td align="center">None</a></td>
</tr>
 <tr><td align="center">TRN-Lite</td>
<td align="center">R50</td>
<td align="center">123</td>
<td align="center">44.8</td>
<td align="center"><a href="configs/youtubevis_2021/trnlite_minvis_R50_bs32_8ep.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1efuTrDtaHKDY6924fCB2ROiY5LICt_ts/view?usp=sharing">model</a></td>
</tr>
 <tr><td align="center">TRN</td>
<td align="center">R50</td>
<td align="center">142</td>
<td align="center">42.8</td>
<td align="center"><a href="configs/youtubevis_2021/trn_minvis_R50_bs32_8ep.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1xnN7KU7rM1Ce57fAKLSJ8j94xU9oWXz0/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: Swin-L -->
 <tr><td align="center">MinVIS</td>
<td align="center">Swin-L</td>
<td align="center">43</td>
<td align="center">55.3</td>
<td align="center">None</a></td>
<td align="center">None</a></td>
</tr>
 <tr><td align="center">TRN-Lite</td>
<td align="center">Swin-L</td>
<td align="center">48</td>
<td align="center">54.9</td>
<td align="center"><a href="configs/youtubevis_2021/swin/trn_minvis_swin_large_bs16_8ep.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1j46M_NFGzpt2Ga4ptOumTRAjmr8eQb1Y/view?usp=sharing">model</a></td>
</tr>
 <tr><td align="center">TRN</td>
<td align="center">Swin-L</td>
<td align="center">52</td>
<td align="center">55.1</td>
<td align="center"><a href="configs/youtubevis_2021/swin/trn_minvis_swin_large_bs16_8ep.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1nNfy7G4EC96IGLk4Uy9dKfuhCtREQWOf/view?usp=sharing">model</a></td>
</tr>
</tbody></table>

### OVIS

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
<td align="center">110</td>
<td align="center">25.0</td>
<td align="center">None</a></td>
<td align="center">None</a></td>
</tr>
 <tr><td align="center">TRN-Lite</td>
<td align="center">R50</td>
<td align="center">123</td>
<td align="center">26.5</td>
<td align="center"><a href="configs/ovis/trnlite_minvis_R50_bs32_8ep.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1CaJhmej8ySruccKklDJcIK6lKuEskpXj/view?usp=sharing">model</a></td>
</tr>
 <tr><td align="center">TRN</td>
<td align="center">R50</td>
<td align="center">142</td>
<td align="center">25.8</td>
<td align="center"><a href="configs/ovis/trn_minvis_R50_bs32_8ep.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1q2vzpYBGwbHqlJ11asTwoITdYHn6p8qy/view?usp=sharing">model</a></td>
</tr>
<!-- ROW: Swin-L -->
 <tr><td align="center">MinVIS</td>
<td align="center">Swin-L</td>
<td align="center">43</td>
<td align="center">39.4</td>
<td align="center">None</a></td>
<td align="center">None</a></td>
</tr>
 <tr><td align="center">TRN-Lite</td>
<td align="center">Swin-L</td>
<td align="center">48</td>
<td align="center">39.1</td>
<td align="center"><a href="configs/ovis/swin/trn_minvis_swin_large_bs16_8ep.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1fkRrN8PCyhhMAb1YYOb--2K6JnpBk7oR/view?usp=sharing">model</a></td>
</tr>
 <tr><td align="center">TRN</td>
<td align="center">Swin-L</td>
<td align="center">52</td>
<td align="center">35.5</td>
<td align="center"><a href="configs/ovis/swin/trn_minvis_swin_large_bs16_8ep.yaml">yaml</a></td>
<td align="center"><a href="https://drive.google.com/file/d/1zOkXgR9F-OzlAmQo9_kNPhnONam1c_2e/view?usp=sharing">model</a></td>
</tr>
</tbody></table>
