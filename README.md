# VmatchDemo

## 基于以下开源项目

[FFmpeg](https://github.com/FFmpeg/FFmpeg)  
[ISC 图像特征提取](https://github.com/lyakaap/ISC21-Descriptor-Track-1st)  
[VSCL 数据集](https://github.com/ant-research/VCSL?tab=readme-ov-file)  
[~~TransVCL 侵权定位算法~~](https://github.com/transvcl/TransVCL)  
[Milvus 向量数据库](https://milvus.io/)  

## 探索成果

TransVCL 的定位细粒度达不到生产需求，想要精准定位要求拷贝视频分段得在 10 秒以.
但是 IscNet 的识别效果很好，可以很好的完成相似图片识别任务，只是识别速度很慢（远慢于 Resnet50），且十分依赖 GPU 的计算加速。
所以本仓库最终选择使用 IscNet + milvus 完成视频拷贝定位.

识别粒度目前可达 3 秒片段，识别得分 80 分及以上为绝对准确结果，30 分以下为不准确结果，30 分～80 分为近似结果。

## 目录说明

[video_features_dnagen](./src/video_features_dnagen.py): 特征提取  
[iscnet_match](./src/iscnet_match.py): 基于 Iscnet + milvus 的特征匹配  
