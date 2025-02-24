import torch
import torchvision
import torch.nn as nn
import json
import math
from loguru import logger as log

from typing import Any
from .transvcl.yolo_pafpn import YOLOPAFPN
from .transvcl.yolo_head import YOLOXHead
from .transvcl.transvcl_model import TransVCL
from collections import defaultdict


def _hhmmss_to_milliseconds(hhmmss: str) -> int:
    # Split the input string by ':'
    hours, minutes, seconds = map(int, hhmmss.split(":"))

    # Convert everything to milliseconds
    total_milliseconds = (hours * 3600 + minutes * 60 + seconds) * 1000

    return total_milliseconds


def _milliseconds_to_hhmmss(ms: int):
    # calculate total seconds
    total_seconds = ms / 1000.0

    # calculate hour, minute, seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # format
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))


class VideoSegment:
    def __init__(
        self,
        sample_start_time: str,
        sample_end_time: str,
        ref_start_time: str,
        ref_end_time: str,
        score: float,
        ref_title: str,
        sample_title: str,
        sample_seg_id: int,
        ref_seg_id: int,
        sample_start_frame: int,
        sample_end_frame: int,
        ref_start_frame: int,
        ref_end_frame: int,
    ):
        self.sample_title = sample_title
        self.sample_seg_id = sample_seg_id
        self.sample_start_time = sample_start_time
        self.sample_end_time = sample_end_time
        self.sample_start_frame = sample_start_frame
        self.sample_end_frame = sample_end_frame

        self.ref_title = ref_title
        self.ref_seg_id = ref_seg_id
        self.ref_start_time = ref_start_time
        self.ref_end_time = ref_end_time
        self.ref_start_frame = ref_start_frame
        self.ref_end_frame = ref_end_frame

        self.sample_start_ms = _hhmmss_to_milliseconds(sample_start_time)
        self.sample_end_ms = _hhmmss_to_milliseconds(sample_end_time)
        self.ref_start_ms = _hhmmss_to_milliseconds(ref_start_time)
        self.ref_end_ms = _hhmmss_to_milliseconds(ref_end_time)

        self.score = score
        self.next_video_segment: Any = None
        self.choiced: bool = False

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "VideoSegment":
        data = json.loads(json_str)
        return cls(**data)

    def __repr__(self) -> str:
        info = f"VideoSegment("
        info += f"sample {self.sample_title} | {_milliseconds_to_hhmmss(self.sample_start_ms)} - {_milliseconds_to_hhmmss(self.sample_end_ms)} | {self.sample_start_frame} -- {self.sample_end_frame} <-> "
        info += f"ref {self.ref_title} | {_milliseconds_to_hhmmss(self.ref_start_ms)} - {_milliseconds_to_hhmmss(self.ref_end_ms)} | {self.ref_start_frame} -- {self.ref_end_frame}, "
        info += f"score={self.score}, choiced: {self.choiced}"
        info += f")"
        return info

    def get_sample_duraion(self):
        return self.sample_end_ms - self.sample_start_ms

    def get_ref_duration(self):
        return self.ref_end_ms - self.ref_start_ms

    def get_ref_frame_len(self):
        return self.ref_end_frame - self.ref_start_frame + 1


def _postprocess(prediction, num_classes, conf_thre, nms_thre, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5 : 5 + num_classes], 1, keepdim=True
        )

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections  # type: ignore
        else:
            output[i] = torch.cat((output[i], detections))  # type: ignore

    return output


def create_transvcl_model(
    weight_file_path: str,
    depth: float = 0.33,
    width: float = 0.50,
    act: str = "silu",
    num_classes: int = 1,
    vta_config: dict[str, Any] = dict(),
    device: str = "cuda",
    is_training: bool = False,
):
    """
    Create a transvcl model which basied on YOLO for image copy-detection task.

    Args:
        weight_file_path (`str=None`):
            Weight file path.
        depth (`float=0.33`):
            the depth of network.
        width (`float=0.50`):
            the width of network.
        act (`str="silu"`):
            Types of activation functions.
        num_classes (`int=1`):
            The number of categories in the classification task.
        vta_config (`dict[str, Any]`):
            some args used by TransVCL, see class TransVCL for the details.
        device (`str='cuda'`):
            Device to load the model.
        is_training (`bool=False`):
            Whether to load the model for training.

    Returns:
        model:
            TransVCL model.
    """
    if len(vta_config) == 0:
        vta_config = {
            "d_model": 256,
            "nhead": 8,
            "layer_names": ["self", "cross"] * 1,
            "attention": "linear",
            "match_type": "dual_softmax",
            "dsmax_temperature": 0.1,
            "keep_ratio": False,
            "unsupervised_weight": 0.5,
        }

    def init_transvcl(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    head = YOLOXHead(num_classes, width, in_channels=in_channels, act=act)
    model = TransVCL(vta_config, backbone, head)

    model.apply(init_transvcl)
    model.head.initialize_biases(1e-2)

    model.to(device).train(is_training)

    if device == "cuda":
        ckpt = torch.load(weight_file_path)
    else:
        ckpt = torch.load(weight_file_path, map_location="cpu")

    model.load_state_dict(ckpt["model"])

    if device == "cuda":
        model = torch.nn.DataParallel(model.cuda())

    return model


def gen_match_segments_by_transVCL(
    model: nn.Module,
    transvcl_batch_feats: list[Any],
    confthre: float,
    nmsthre: float,
    img_size: tuple[int, int],
    segment_length: int,
    frame_interval: float,
    device: str = "cuda",
) -> list[VideoSegment]:
    """
    Using TransVCL for video copy positioning

    Args:
        model (`nn.Module`):
            TransVCL model

        transvcl_batch_feats (`list[Any]`):
            transvcl batch features, generate by func:: transform_feats.trans_isc_features_to_transVCL_fromat

        confthre (`float`):
            conf threshold of copied segments

        nmsthre (`float`):
            nms threshold of copied segments

        img_size (`tuple[int, int]`):
            length for copied localization module

        segment_length (`int`):
            frames number of each segment

        frame_interval (`float`):
            the frame interval(millisecond)

        device (`str="cuda"`):
            Devices for model inference, must be same as the model use.


    Returns:
        result (`defaultdict(list)`):
        content:
        result ==>\n
        {'SampleFileName-RefFileName': [['00:00:00', '00:00:00', '00:02:30', '00:02:30', 0.9887829422950745]]}\n
        titile :[[sample start time, ref start time , samle end time , ref end time, confirm score]\n
    """
    if isinstance(model, nn.DataParallel):
        if not isinstance(model.module, TransVCL):
            raise RuntimeError(f"unknown model: {type(model)} -- {type(model.module)}")
    else:
        if not isinstance(model, TransVCL):
            raise RuntimeError(f"unknown model: {type(model)}")

    # batch_feat_result: dict[str, list[Any]] = {}
    outputs_list = list()

    for idx, batch_feat in enumerate(transvcl_batch_feats):
        (
            sample_feat,
            ref_feat,
            mask1,
            mask2,
            img_info,
            sample_frame_offset,
            ref_frame_offset,
        ) = batch_feat

        sample_feat, ref_feat, mask1, mask2 = (
            sample_feat.unsqueeze(0).to(device),
            ref_feat.unsqueeze(0).to(device),
            mask1.unsqueeze(0).to(device),
            mask2.unsqueeze(0).to(device),
        )

        # log.info(f"query file: {title}")
        # log.info(f"sample_feat: {sample_feat.shape}")
        # log.info(f"ref_feat: {ref_feat.shape}")
        # log.info(f"mask1: {mask1.shape}")
        # log.info(f"mask2: {mask2.shape}")

        with torch.no_grad():
            model_outputs = model(
                sample_feat,
                ref_feat,
                mask1,
                mask2,
                [
                    "",
                ],
                img_info,
            )

            outputs = _postprocess(
                model_outputs[1],
                1,
                confthre,
                nmsthre,
                class_agnostic=True,
            )

            for idx2, output in enumerate(outputs):
                if output is not None:
                    bboxes = output[:, :5].cpu()

                    scale1, scale2 = (
                        img_info[0] / img_size[0],
                        img_info[1] / img_size[1],
                    )
                    bboxes[:, 0:4:2] *= scale2[idx2]
                    bboxes[:, 1:4:2] *= scale1[idx2]
                    outputs_list.append(
                        [
                            "",
                            sample_frame_offset,
                            ref_frame_offset,
                            bboxes[:, (1, 0, 3, 2, 4)].tolist(),
                        ]
                    )

    # result = defaultdict(list)

    result: list[VideoSegment] = list()

    for output in outputs_list:
        # log.info(f"titile: {titile}")
        titile = output[0]
        # sample_title, ref_title = titile.split(",")
        sample_title = ""
        ref_title = ""
        # i is sample frame offset
        # j is reference frame offset
        i = output[1]
        j = output[2]

        for r in output[3]:
            sample_start_frame = math.floor((r[0] + i))
            ref_start_frame = math.floor((r[1] + j))

            sample_end_frame = math.ceil((r[2] + i))
            ref_end_frame = math.ceil((r[3] + j))
            
            sample_start_frame = max(sample_start_frame, 0)
            ref_start_frame = max(ref_start_frame, 0)
            sample_end_frame = max(sample_end_frame, 0)
            ref_end_frame = max(ref_end_frame, 0)
            # log.info(f"sample seg: {i}, sample: {sample_start_frame} -- {sample_end_frame} || ref seg: {j}, ref: {ref_start_frame} -- {ref_end_frame}")

            sample_start_timeformat = _milliseconds_to_hhmmss(
                int(sample_start_frame * frame_interval)
            )
            ref_start_timeformat = _milliseconds_to_hhmmss(
                int(ref_start_frame * frame_interval)
            )
            sample_end_timeformat = _milliseconds_to_hhmmss(
                int(sample_end_frame * frame_interval)
            )
            ref_end_timeformat = _milliseconds_to_hhmmss(int(ref_end_frame * frame_interval))
            confirm_score = round(r[4] * 100.000, 2)

            video_match_segment = VideoSegment(
                sample_start_time=sample_start_timeformat,
                sample_end_time=sample_end_timeformat,
                ref_start_time=ref_start_timeformat,
                ref_end_time=ref_end_timeformat,
                score=confirm_score,
                sample_title=sample_title,
                ref_title=ref_title,
                sample_seg_id=-1,
                ref_seg_id=-1,
                sample_start_frame=sample_start_frame,
                sample_end_frame=sample_end_frame,
                ref_start_frame=ref_start_frame,
                ref_end_frame=ref_end_frame,
            )

            result.append(video_match_segment)

    return result
