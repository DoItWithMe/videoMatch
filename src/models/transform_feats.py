import torch
import numpy as np
from loguru import logger as log


def _feat_paddding(feat: torch.Tensor, axis: int, new_size: int, fill_value: int = 0):
    pad_shape = list(feat.shape)
    pad_shape[axis] = max(0, new_size - pad_shape[axis])

    # 只有当需要填充时才进行填充操作
    if pad_shape[axis] > 0:
        pad = torch.full(pad_shape, fill_value, dtype=feat.dtype, device=feat.device)
        return torch.cat([feat, pad], dim=axis)

    # 如果没有需要填充的部分，直接返回原始张量
    return feat


def trans_isc_features_to_transVCL_fromat(
    sample_feats: np.ndarray,
    ref_feats: np.ndarray,
    title: str,
    segment_length: int,
    device: str = "cuda",
):
    """
    Feature transformer for ISCNet features to TransVCL features format

    Args:
        sample_feats (`NDArray[numpy[n,m], dtype=np.float32]`):
            the NDArray of imgs features\n
            n = len(imgs_path_list)\n
            m = model output features dim\n

        ref_feats (`NDArray[numpy[n,m], dtype=np.float32]`):
            the NDArray of imgs features\n
            n = len(imgs_path_list)\n
            m = model output features dim\n

        title (`str`):
            compare task name

        device (`str="cuda"`):
            Devices for model inference, must be same as the model use.

        segment_length (`int`)
            frames number of each segment, it's ok if real frame length is lesser than segment_length
    """

    sample_list = [
        sample_feats[i * segment_length : (i + 1) * segment_length]
        for i in range(len(sample_feats) // segment_length)
    ]
    ref_list = [
        ref_feats[j * segment_length : (j + 1) * segment_length]
        for j in range(len(ref_feats) // segment_length)
    ]

    # 添加剩余部分
    if len(sample_feats) % segment_length != 0:
        sample_list.append(
            sample_feats[(len(sample_feats) // segment_length) * segment_length :]
        )

    if len(ref_feats) % segment_length != 0:
        ref_list.append(
            ref_feats[(len(ref_feats) // segment_length) * segment_length :]
        )

    batch_list = []
    for i, sample in enumerate(sample_list):
        for j, ref in enumerate(ref_list):
            # 获取有效长度
            sample_valid_len = len(sample)
            ref_valid_len = len(ref)

            # 创建填充后的张量
            sample_feat_padding = _feat_paddding(
                torch.tensor(sample, device=device), 0, segment_length
            )
            ref_feat_padding = _feat_paddding(
                torch.tensor(ref, device=device), 0, segment_length
            )

            # 移动到 CPU 并保存结果
            batch_list.append(
                (
                    sample_feat_padding.cpu(),
                    ref_feat_padding.cpu(),
                    torch.tensor(
                        [True] * sample_valid_len
                        + [False] * (segment_length - sample_valid_len)
                    ).cpu(),
                    torch.tensor(
                        [True] * ref_valid_len
                        + [False] * (segment_length - ref_valid_len)
                    ).cpu(),
                    [
                        torch.tensor([sample_valid_len], device=device).cpu(),
                        torch.tensor([ref_valid_len], device=device).cpu(),
                    ],
                    title,
                    i,
                    j,
                )
            )

    return batch_list


def trans_isc_features_to_transVCL_fromat2(
    sample_frame_offset: int,
    sample_feat: np.ndarray,
    ref_frame_offset: int,
    ref_feat_list: list[np.ndarray],
    device: str = "cuda",
):
    batch_list = []
    segment_length = sample_feat.shape[0]

    for ref_seg_id, ref_feats in enumerate(ref_feat_list):
        sample_valid_len = len(sample_feat)
        ref_valid_len = len(ref_feats)

        # 创建填充后的张量
        sample_feat_padding = _feat_paddding(
            torch.tensor(sample_feat, device=device), 0, segment_length
        )
        ref_feat_padding = _feat_paddding(
            torch.tensor(ref_feats, device=device), 0, segment_length
        )
        # 移动到 CPU 并保存结果
        batch_list.append(
            (
                sample_feat_padding.cpu(),
                ref_feat_padding.cpu(),
                torch.tensor(
                    [True] * sample_valid_len
                    + [False] * (segment_length - sample_valid_len)
                ).cpu(),
                torch.tensor(
                    [True] * ref_valid_len + [False] * (segment_length - ref_valid_len)
                ).cpu(),
                [
                    torch.tensor([sample_valid_len], device=device).cpu(),
                    torch.tensor([ref_valid_len], device=device).cpu(),
                ],
                sample_frame_offset,
                ref_seg_id * segment_length + ref_frame_offset,
            )
        )

    return batch_list
