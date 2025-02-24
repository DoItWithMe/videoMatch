import sys
import os

_project_dipath: str = os.path.dirname((os.path.abspath(__file__)))
sys.path.append(_project_dipath)

import argparse
import torch
import numpy as np
import shutil
import concurrent.futures
import multiprocessing
import copy
import torch.nn as nn

from models.iscnet_utils import create_isc_model, gen_img_feats_by_ISCNet
from models.transvcl_utils import (
    create_transvcl_model,
    gen_match_segments_by_transVCL,
    VideoSegment,
)
from models.transform_feats import (
    trans_isc_features_to_transVCL_fromat,
    trans_isc_features_to_transVCL_fromat2,
)
from ffmpeg.ffmpeg import extract_imgs
from loguru import logger as log
from utils.time_utils import TimeRecorder
from typing import Any

import torch.multiprocessing as mp
import math


DEVICE_LIST = ["cpu", "cuda"]
MIN_SEGMENT_LENGTH = 100


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ffmpeg",
        type=str,
        help="ffmpeg bin file path",
        default="./assets/ffmpeg",
        required=False,
    )

    parser.add_argument("--device", type=str, help="cpu or cuda", default="cpu")

    parser.add_argument(
        "--output-dir",
        type=str,
        help="output dir path, auto-create it if not exisit",
        default="./output",
        required=False,
    )

    parser.add_argument(
        "--isc-weight",
        type=str,
        help="isc weight path",
        default="./assets/models/isc_ft_v107.pth.tar",
        required=False,
    )

    parser.add_argument(
        "--transVCL-weight",
        type=str,
        help="transVCL weight path",
        default="./assets/models/tarnsVCL_model_1.pth",
        required=False,
    )

    parser.add_argument(
        "--conf-thre",
        type=float,
        default=0.6,
        help="transVCL: conf threshold of copied segments    ",
    )

    parser.add_argument(
        "--nms-thre",
        type=float,
        default=0.3,
        help="transVCL: nms threshold of copied segments",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="transVCL: length for copied localization module",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=1,
        help="output fps when converting video to images",
    )

    parser.add_argument(
        "--segment-duration",
        type=int,
        default=200,
        help="transVCL: segment duration in milliseconds",
    )

    parser.add_argument(
        "--sample-videos-dir",
        "-s",
        type=str,
        help="input sample videos dir",
        required=True,
    )

    parser.add_argument(
        "--reference-videos-dir",
        "-r",
        type=str,
        help="input reference videos dir",
        required=True,
    )

    parser.add_argument(
        "--extract-imgs",
        type=bool,
        help="extract images of input videos",
        default=False,
    )

    parser.add_argument(
        "--extract-feats",
        type=bool,
        help="extract feats of input videos",
        default=False,
    )

    return parser.parse_args()


def check_args(args):
    if args.device == "cuda":
        if not torch.cuda.is_available():
            log.warning("gpu is not available, use cpu")
            args.device = "cpu"

    if args.device not in DEVICE_LIST:
        log.error(
            f"unkown device: {args.device}, only thess is available: {DEVICE_LIST}"
        )
        exit(-1)

    if args.segment_duration < MIN_SEGMENT_LENGTH:
        log.error(f"segment duration can not smaller than {MIN_SEGMENT_LENGTH}")
        exit(-1)


def init_log():
    log.remove()
    log.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
    )


def _generate_imgs(
    ffmpeg_bin_file_path: str,
    videos_dir_path: str,
    output_dir: str,
    fps: int,
    executor_num: int,
):
    videos_files_path = [
        os.path.join(videos_dir_path, img_name)
        for img_name in os.listdir(videos_dir_path)
    ]

    args_list = list()

    for video_file_path in videos_files_path:
        args_list.append((ffmpeg_bin_file_path, video_file_path, output_dir, fps))

    time_recorder = TimeRecorder()
    time_recorder.start_record()

    with concurrent.futures.ThreadPoolExecutor(max_workers=executor_num) as executor:
        executor.map(
            lambda args: extract_imgs(args[0], args[1], args[2], args[3]),
            args_list,
        )

    time_recorder.end_record()
    log.info(
        f"generate imgs of {len(videos_files_path)} videos cost {time_recorder.get_total_duration_miliseconds()} ms"
    )


def _save_feats(feats_save_path, feats):
    tmp_dir = os.path.dirname(feats_save_path)
    os.makedirs(tmp_dir, exist_ok=True)
    log.info(f"save feats: {feats.shape} to {feats_save_path} ")
    np.save(feats_save_path, feats)


def _generate_feats(
    imgs_output_dir: str, feats_output_dir: str, isc_weight_path: str, device: str
):
    isc_model, isc_processer = create_isc_model(
        weight_file_path=isc_weight_path, device=device, is_training=False
    )

    feats_time_recorder = TimeRecorder()
    save_time_recorder = TimeRecorder()
    imgs_count = 0

    tmp_imgs_output_dir = [
        os.path.join(imgs_output_dir, title) for title in os.listdir(imgs_output_dir)
    ]

    for img_dir_path in tmp_imgs_output_dir:
        imgs_list = os.listdir(img_dir_path)
        # sort imgs for listdir may not return by the number order...
        # print(imgs_list)

        tmp_imgs_list = sorted(imgs_list, key=lambda x: int(x.split(".")[0]))

        imgs_list.clear()
        imgs_list = [os.path.join(img_dir_path, img_name) for img_name in tmp_imgs_list]
        imgs_count += len(imgs_list)

        log.info(f"start gen feats, imgs len: {len(imgs_list)}")
        # it's very slow when use cpu to generate image feats....
        feats_time_recorder.start_record()
        ref_isc_feats = gen_img_feats_by_ISCNet(
            imgs_list, isc_model, isc_processer, device
        )
        feats_time_recorder.end_record()
        log.info(f"get feats: {ref_isc_feats.shape}")

        feats_file_path = os.path.join(
            feats_output_dir,
            f"{os.path.splitext(os.path.basename(img_dir_path))[0]}.npy",
        )

        save_time_recorder.start_record()
        _save_feats(feats_file_path, ref_isc_feats)
        save_time_recorder.end_record()

    log.info(
        f"generate iscNet feats by {device} of {imgs_count} imgs cost {feats_time_recorder.get_total_duration_miliseconds()} ms, save it cost {save_time_recorder.get_total_duration_miliseconds()} ms"
    )


def _load_feats(feats_store_dir_path: str):
    time_recorder = TimeRecorder()
    feat_dict: dict[str, Any] = dict()
    feats_files_list = [
        os.path.join(feats_store_dir_path, file_name)
        for file_name in os.listdir(feats_store_dir_path)
    ]

    time_recorder.start_record()
    count = 0
    for feats_file in feats_files_list:
        feat_dict[os.path.splitext(os.path.basename(feats_file))[0]] = np.load(
            feats_file
        )
        count += 1
        # if count >= 1:
        #     break
    time_recorder.end_record()
    log.info(
        f"load feats of {feats_store_dir_path} cost {time_recorder.get_total_duration_miliseconds()} ms"
    )

    return feat_dict


def _milliseconds_to_hhmmss(ms: int):
    # calculate total seconds
    total_seconds = ms / 1000.0

    # calculate hour, minute, seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # format
    return "{:02}:{:02}:{:02}".format(int(hours), int(minutes), int(seconds))


def _hhmmss_to_milliseconds(hhmmss: str) -> int:
    # Split the input string by ':'
    hours, minutes, seconds = map(int, hhmmss.split(":"))

    # Convert everything to milliseconds
    total_milliseconds = (hours * 3600 + minutes * 60 + seconds) * 1000

    return total_milliseconds


def _global_search(
    device: str,
    transvcl_model: nn.Module,
    confthre: float,
    nmsthre: float,
    img_size: tuple[Any, Any],
    frame_interval: float,
    sample_frame_offset: int,
    sample_feat: np.ndarray,
    ref_feat_list: np.ndarray,
) -> list[VideoSegment]:

    segment_length = sample_feat.shape[0]

    # 这里母本的分段切割点会很影响匹配效果, 所以做个策略把中间的缝也给补上
    ref_list = [
        ref_feat_list[j * segment_length : (j + 1) * segment_length]
        for j in range(len(ref_feat_list) // segment_length)
    ]

    if len(ref_feat_list) % segment_length != 0:
        ref_list.append(
            ref_feat_list[(len(ref_feat_list) // segment_length) * segment_length :]
        )

    compare_batch_feats_list = trans_isc_features_to_transVCL_fromat2(
        sample_frame_offset,
        sample_feat,
        0,
        ref_list,
        device,
    )

    # 填补缝隙
    for j in range(len(ref_feat_list) // segment_length):
        tmp_offset = int((j + 0.5) * segment_length)
        tmp_ref_seg = ref_feat_list[tmp_offset : tmp_offset + segment_length]
        tmp_compare_batch_list = trans_isc_features_to_transVCL_fromat2(
            sample_frame_offset,
            sample_feat,
            tmp_offset,
            [
                tmp_ref_seg,
            ],
            device,
        )
        compare_batch_feats_list.extend(tmp_compare_batch_list)

    log.info(f"compare_batch_feats_list: {len(compare_batch_feats_list)}")

    matched_segments = gen_match_segments_by_transVCL(
        transvcl_model,
        compare_batch_feats_list,
        confthre,
        nmsthre,
        img_size,
        segment_length,
        frame_interval,
        device,
    )

    for i in matched_segments:
        log.info(i)

    return matched_segments


def _increasement_search(
    device: str,
    transvcl_model: nn.Module,
    confthre: float,
    nmsthre: float,
    img_size: tuple[Any, Any],
    frame_interval: float,
    sample_frame_offset: int,
    sample_feat: np.ndarray,
    ref_frame_offset: int,
    ref_feat: np.ndarray,
):
    segment_length = sample_feat.shape[0]
    compare_batch_feats_list = trans_isc_features_to_transVCL_fromat2(
        sample_frame_offset,
        sample_feat,
        ref_frame_offset,
        [
            ref_feat,
        ],
        device,
    )

    matched_segments = gen_match_segments_by_transVCL(
        transvcl_model,
        compare_batch_feats_list,
        confthre,
        nmsthre,
        img_size,
        segment_length,
        frame_interval,
        device,
    )
    

    return matched_segments


def _filter_and_sort_matched_segs(
    matched_segments: list[VideoSegment],
    score_limit: float,
    duration_limit: int,
):
    if len(matched_segments) == 0:
        return list()

    filtered_segs = [
        match_seg
        for match_seg in matched_segments
        if match_seg.score >= score_limit
        and match_seg.get_ref_duration() >= duration_limit
        and match_seg.get_sample_duraion() >= duration_limit
    ]

    sorted_matched_segments = sorted(
        filtered_segs,
        key=lambda seg: seg.sample_start_ms,
    )

    return sorted_matched_segments


def _could_merge(s, e, l, u) -> bool:
    if s <= l and e >= l:
        return True

    if s >= l and s <= u:
        return True

    return False


def _merge_match_segs(matched_segments: list[VideoSegment], merge_threshold: float):
    if len(matched_segments) == 0:
        return list()

    matched_segments_len = len(matched_segments)
    for i in range(0, matched_segments_len):
        current_seg = matched_segments[i]
        if current_seg.choiced:
            continue

        for j in range(i + 1, matched_segments_len):
            next_seg = matched_segments[j]
            if next_seg.choiced:
                continue

            cur_ss_ms = current_seg.sample_start_ms
            cur_se_ms = current_seg.sample_end_ms

            next_ss_ms = next_seg.sample_start_ms
            next_se_ms = next_seg.sample_end_ms

            s_duration_threshold = (cur_se_ms - cur_ss_ms) * merge_threshold

            s_lower_bound = next_ss_ms - s_duration_threshold
            s_upper_bound = next_se_ms + s_duration_threshold

            if s_lower_bound < 0:
                s_lower_bound = 0

            if not _could_merge(cur_ss_ms, cur_se_ms, s_lower_bound, s_upper_bound):
                continue

            cur_rs_ms = current_seg.ref_start_ms
            cur_re_ms = current_seg.ref_end_ms

            next_rs_ms = next_seg.ref_start_ms
            next_re_ms = next_seg.ref_end_ms

            r_duration_threshold = (cur_re_ms - cur_rs_ms) * merge_threshold

            r_lower_bound = next_rs_ms - r_duration_threshold
            r_upper_bound = next_re_ms + r_duration_threshold

            if r_lower_bound < 0:
                r_lower_bound = 0

            if not _could_merge(cur_rs_ms, cur_re_ms, r_lower_bound, r_upper_bound):
                continue

            next_seg.choiced = True
            log.info(f"\ncur: {current_seg}\nnext: {next_seg}\nmerge!\n")

            current_seg.sample_start_ms = min(cur_ss_ms, next_ss_ms)
            current_seg.sample_end_ms = max(cur_se_ms, next_se_ms)
            current_seg.ref_start_ms = min(cur_rs_ms, next_rs_ms)
            current_seg.ref_end_ms = max(cur_re_ms, next_re_ms)

            current_seg.sample_start_frame = min(
                current_seg.sample_start_frame, next_seg.sample_start_frame
            )

            current_seg.sample_end_frame = max(
                current_seg.sample_end_frame, next_seg.sample_end_frame
            )

            current_seg.ref_start_frame = min(
                current_seg.ref_start_frame, next_seg.ref_start_frame
            )

            current_seg.ref_end_frame = max(
                current_seg.ref_end_frame, next_seg.ref_end_frame
            )

            current_seg.score = (current_seg.score + next_seg.score) / 2.0

            log.info(f"cur seg update to\n{current_seg}\n\n")

    merged_matched_segments = [
        match_seg for match_seg in matched_segments if match_seg.choiced == False
    ]

    return merged_matched_segments


def _downsampled(src_data, interval: int):

    tmp_list = list()
    count = 0

    for data in src_data:
        if count % interval != 0:
            count += 1
            continue
        count += 1
        tmp_list.append(data)

    return np.array(tmp_list)


def _query_matched_videos(
    transvcl_weight_path: str,
    confthre: int,
    nmsthre: int,
    img_size: tuple[Any, Any],
    device: str,
    sample_feat_dict: dict[str, Any],
    ref_feat_dict: dict[str, Any],
    segment_length: int,
    frame_interval: float,
    worker_num: int,
):

    transform_time_recorder = TimeRecorder()
    compare_time_recorder = TimeRecorder()

    transvcl_model = create_transvcl_model(
        weight_file_path=transvcl_weight_path, device=device, is_training=False
    )

    score_limit = 80

    forward_offset_thresh = 0.3
    backward_offset_thresh = 6
    offset_round = 10

    merge_threshold = 0.5
    link_threshold = 0.5
    duration_threshold = 0.3

    interval = 1
    frame_interval *= interval
    segment_length //= interval

    segment_duration = segment_length * frame_interval

    duration_limit = duration_threshold * segment_duration
    link_duration_limit = link_threshold * segment_duration

    for sample_name, sample_feat in sample_feat_dict.items():
        downsampled_sample_feat = _downsampled(sample_feat, interval)

        # if (
        #     sample_name != "R_2_flip_jieshuo"
        #     and sample_name != "R_2_flip_jieshuo"
        # ):
        #     continue

        # 对样本进行分段
        sample_list = [
            downsampled_sample_feat[i * segment_length : (i + 1) * segment_length]
            for i in range(len(downsampled_sample_feat) // segment_length)
        ]

        if len(downsampled_sample_feat) % segment_length != 0:
            sample_list.append(
                downsampled_sample_feat[
                    (len(downsampled_sample_feat) // segment_length) * segment_length :
                ]
            )

        log.info(
            f"sample: {sample_name} have {len(sample_list)} segments, each segments have {segment_length} frames.."
        )

        for ref_name, ref_feat in ref_feat_dict.items():
            downsampled_ref_feat = _downsampled(ref_feat, interval)

            titile = f"{sample_name},{ref_name}"

            matched_segments_list: list[VideoSegment] = list()
            global_search_flag = True
            last_matched_segment_idx = 0
            sample_seg_id = 0
            sample_seg_len = len(sample_list)
            while sample_seg_id < sample_seg_len:
                sample_feat = sample_list[sample_seg_id]
                if sample_seg_id >= 999:
                    break

                # sample_s = sample_seg_id * segment_duration
                log.info(
                    f"start search sample sample_seg_id: {sample_seg_id} - {_milliseconds_to_hhmmss(int(sample_seg_id * segment_duration))} - {_milliseconds_to_hhmmss(int((sample_seg_id + 1) * segment_duration)) }"
                )

                tmp_duration = sample_feat.shape[0] * frame_interval

                if tmp_duration < duration_limit:
                    log.warning(
                        f"sample segment not contain enough frame({tmp_duration}--{sample_feat.shape[0]}/{duration_limit}), skip.. "
                    )
                    break

                src_len = len(matched_segments_list)
                # 样本第一个分段 或者 是上一分段无结果
                if global_search_flag:
                    matched_segments = _global_search(
                        device,
                        transvcl_model,
                        confthre,
                        nmsthre,
                        img_size,
                        frame_interval,
                        sample_seg_id * segment_length,
                        sample_feat,
                        downsampled_ref_feat,
                    )

                    log.info("start global")

                    sorted_matched_segments = _filter_and_sort_matched_segs(
                        matched_segments, score_limit, int(duration_limit)
                    )

                    if len(sorted_matched_segments) > 0:
                        merged_matched_segments = _merge_match_segs(
                            sorted_matched_segments, merge_threshold
                        )
                        matched_segments_list.extend(merged_matched_segments)
                        for i in merged_matched_segments:
                            log.info(f"global search got: {i}")
                else:
                    # Todo:
                    # 1. 增量搜索可能会导致当前 seg 无匹配，需要重新进行全局匹配
                    # 2. 匹配分段 merge 逻辑
                    tmp_matched_segments_list: list[VideoSegment] = list()
                    for idx in range(last_matched_segment_idx, src_len):
                        last_matched_segment = matched_segments_list[idx]
                        last_matched_segment_ref_end_frame = (
                            last_matched_segment.ref_end_frame
                        )

                        last_matched_segment_ref_len = (
                            last_matched_segment.get_ref_frame_len()
                        )

                        ref_forward_lowwer_bound = math.floor(
                            last_matched_segment_ref_end_frame
                            - forward_offset_thresh * last_matched_segment_ref_len
                        )

                        ref_forward_upper_bound = math.floor(
                            last_matched_segment_ref_end_frame
                            + backward_offset_thresh
                            * segment_duration
                            // frame_interval
                        )

                        step = (
                            ref_forward_upper_bound - ref_forward_lowwer_bound
                        ) // offset_round

                        log.info(
                            f"ref search range: {_milliseconds_to_hhmmss(int((ref_forward_lowwer_bound) * frame_interval))} -- {_milliseconds_to_hhmmss(int((ref_forward_upper_bound + step ) * frame_interval) + segment_length)}, step: {step} "
                        )

                        offset_matched_segments: list[VideoSegment] = list()
                        for ref_forward_frame in range(
                            ref_forward_lowwer_bound,
                            ref_forward_upper_bound + step,
                            step,
                        ):
                            ref_backward_frame = segment_length + ref_forward_frame - 1

                            ref_increasement_feat = downsampled_ref_feat[
                                ref_forward_frame : ref_backward_frame + 1
                            ][:]

                            tmp_duration = (
                                ref_increasement_feat.shape[0] * frame_interval
                            )
                            if tmp_duration < duration_limit:
                                log.warning(
                                    f"ref segment not contain enough frame({tmp_duration}-{ref_forward_frame}:{ref_backward_frame+1}::{downsampled_ref_feat.shape}/{duration_limit}), skip.. "
                                )
                                continue

                            log.info(
                                f"search range: {_milliseconds_to_hhmmss(int((ref_forward_frame) * frame_interval))} - {_milliseconds_to_hhmmss(int((ref_backward_frame)*frame_interval))}"
                            )

                            matched_segments = _increasement_search(
                                device,
                                transvcl_model,
                                confthre,
                                nmsthre,
                                img_size,
                                frame_interval,
                                sample_seg_id * segment_length,
                                sample_feat,
                                ref_forward_frame,
                                ref_increasement_feat,
                            )

                            sorted_matched_segments = _filter_and_sort_matched_segs(
                                matched_segments, score_limit, int(duration_limit)
                            )

                            if len(sorted_matched_segments) == 0:
                                continue

                            merged_matched_segments = _merge_match_segs(
                                sorted_matched_segments, merge_threshold
                            )
                            offset_matched_segments.extend(merged_matched_segments)

                        merged_matched_segments = _merge_match_segs(
                            offset_matched_segments, merge_threshold
                        )
                        for i in merged_matched_segments:
                            log.info(f"increasement search got: {i}")

                        for matched_seg in merged_matched_segments:
                            matched_segments_list.append(matched_seg)
                            if (
                                abs(
                                    _hhmmss_to_milliseconds(
                                        last_matched_segment.ref_end_time
                                    )
                                    - _hhmmss_to_milliseconds(
                                        matched_seg.ref_start_time
                                    )
                                )
                                <= link_duration_limit
                            ):
                                last_score = 0
                                if last_matched_segment.next_video_segment is not None:
                                    last_score = (
                                        last_matched_segment.next_video_segment.score
                                    )

                                if last_score < matched_seg.score:
                                    last_matched_segment.next_video_segment = (
                                        matched_seg
                                    )
                                elif last_score == matched_seg.score:
                                    current_len = (
                                        matched_seg.ref_end_frame
                                        - matched_seg.ref_start_frame
                                        + 1
                                    )
                                    if last_matched_segment_ref_len < current_len:
                                        last_matched_segment.next_video_segment = (
                                            matched_seg
                                        )

                new_len = len(matched_segments_list)

                if global_search_flag:
                    if new_len != src_len:
                        global_search_flag = False
                        last_matched_segment_idx = src_len
                else:
                    if new_len == src_len:
                        # 当前 sample seg 增量搜索失败，回退至全局搜索
                        global_search_flag = True
                        sample_seg_id -= 1
                    else:
                        last_matched_segment_idx = src_len

                sample_seg_id += 1

                log.info(
                    f"global_search_flag: {global_search_flag}, get new matchec: new_len:{new_len} src_len: {src_len}, diff: {new_len - src_len}, last_matched_segment_idx update to {last_matched_segment_idx}"
                )

                log.info("====\n")

            tmp_res_list = list()
            tmp_merged = _merge_match_segs(matched_segments_list, merge_threshold)
            tmp_merged2 = _filter_and_sort_matched_segs(
                tmp_merged, 85, 2 * int(duration_limit)
            )
            
            log.info(f"Results, sample_name: {sample_name}, ref: {ref_name} ")
            for i in tmp_merged2:
                if i.get_ref_duration() < 1.5 * segment_duration:
                    continue
                log.info(i)

            # for matched_seg in matched_segments_list:
            #     if matched_seg.choiced:
            #         continue

            #     tmp_matched_seg = matched_seg
            #     info = str("vvvvvvv \n")
            #     tmp_duration = 0
            #     while tmp_matched_seg is not None:
            #         info += str(tmp_matched_seg)
            #         info += "\n"
            #         tmp_duration += tmp_matched_seg.get_ref_duration()
            #         tmp_matched_seg.choiced = True
            #         tmp_matched_seg = tmp_matched_seg.next_video_segment
            #     log.info(
            #         f"tmp_duration: {tmp_duration}, duration_limit: {duration_limit * 3}"
            #     )
            #     # if tmp_duration >= duration_limit * 3:
            #     tmp_res_list.append(info)

            # k = 50
            # log.info(f"print top{k} matched results.")
            # for idx in range(0, len(tmp_res_list)):
            #     # if idx >= k:
            #     #     break
            #     log.info(tmp_res_list[idx])

    log.info(
        f"finish all, transform avg cost {transform_time_recorder.get_avg_duration_miliseconds()} ms, compare avg cost {compare_time_recorder.get_avg_duration_miliseconds()} ms"
    )


def main():
    # torch.set_num_threads(16)
    init_log()
    args = parser_args()
    check_args(args)

    ref_videos_dir_path = os.path.normpath(args.reference_videos_dir)
    sample_videos_dir_path = os.path.normpath(args.sample_videos_dir)

    device = args.device
    ffmpeg_path = args.ffmpeg
    output_dir = args.output_dir

    isc_weight_path = args.isc_weight
    transvcl_weight_path = args.transVCL_weight

    confthre = args.conf_thre
    nmsthre = args.nms_thre
    img_size = (args.img_size, args.img_size)

    segment_duration: int = args.segment_duration

    fps = args.fps
    frame_interval = 1000.0 / float(fps)

    if frame_interval > segment_duration:
        log.warning(
            f"fps is {fps}, frame interval {frame_interval} ms is bigger than segment duration: {segment_duration} ms"
        )
        segment_duration = round(frame_interval * 10)
        log.warning(f"segment duration reset to {segment_duration} ms")

    segment_length = round(segment_duration / frame_interval)
    log.info(
        f"segment_length: {segment_length}, segment_duration: {segment_duration}, frame_interval: {frame_interval}"
    )
    log.info(f"device: {device}")

    re_extract_imgs_flag = args.extract_imgs
    re_extract_imgs_feats_flag = args.extract_feats

    if re_extract_imgs_flag:
        re_extract_imgs_feats_flag = True

    # 1. 解析输入媒体文件列表，判断是否重新提取帧

    ref_output_base_name = f"{os.path.basename(ref_videos_dir_path)}_fps_{fps}"
    sample_output_base_name = f"{os.path.basename(sample_videos_dir_path)}_fps_{fps}"

    ref_imgs_output_dir = os.path.join(output_dir, "imgs", ref_output_base_name)
    sample_imgs_output_dir = os.path.join(output_dir, "imgs", sample_output_base_name)

    if re_extract_imgs_flag:
        executor_num = 2
        log.info(f"star generate ref imgs with {executor_num} worker ...")
        shutil.rmtree(ref_imgs_output_dir, ignore_errors=True)
        os.makedirs(ref_imgs_output_dir, exist_ok=True)
        _generate_imgs(
            ffmpeg_path, ref_videos_dir_path, ref_imgs_output_dir, fps, executor_num
        )

        log.info(f"star generate sample imgs with {executor_num} worker ...")
        shutil.rmtree(sample_imgs_output_dir, ignore_errors=True)
        os.makedirs(sample_imgs_output_dir, exist_ok=True)
        _generate_imgs(
            ffmpeg_path,
            sample_videos_dir_path,
            sample_imgs_output_dir,
            fps,
            executor_num,
        )
    else:
        log.info(f"use cached imgs: {ref_imgs_output_dir} and {sample_imgs_output_dir}")

    # 2. 使用 IscNet 提取特征

    ref_feats_output_dir = os.path.join(output_dir, "feats", ref_output_base_name)
    sample_feats_output_dir = os.path.join(output_dir, "feats", sample_output_base_name)

    if re_extract_imgs_feats_flag:
        log.info(f"star generate ref feats...")
        shutil.rmtree(ref_feats_output_dir, ignore_errors=True)
        os.makedirs(ref_feats_output_dir, exist_ok=True)

        _generate_feats(
            ref_imgs_output_dir, ref_feats_output_dir, isc_weight_path, device
        )

        log.info(f"star generate sample feats...")
        shutil.rmtree(sample_feats_output_dir, ignore_errors=True)
        os.makedirs(sample_feats_output_dir, exist_ok=True)

        _generate_feats(
            sample_imgs_output_dir, sample_feats_output_dir, isc_weight_path, device
        )
        log.info("all feats extract done.....")
    else:
        log.info(
            f"use cached feats: {ref_feats_output_dir} and {sample_feats_output_dir}"
        )

    # 3. 加载特征

    # log.info(f"start load feats in {ref_feats_output_dir}")
    # ref_feat_dict = _load_feats(ref_feats_output_dir)

    # log.info(f"start load feats in {sample_feats_output_dir}")
    # sample_feat_dict = _load_feats(sample_feats_output_dir)

    # 4. 查询
    # create_isc_model(isc_weight_path)
    # from isc_feature_extractor import create_model
    # worker_num = 5
    # _query_matched_videos(
    #     transvcl_weight_path,
    #     confthre,
    #     nmsthre,
    #     img_size,
    #     device,
    #     sample_feat_dict,
    #     ref_feat_dict,
    #     segment_length,
    #     frame_interval,
    #     worker_num,
    # )

if __name__ == "__main__":
    print(f"torch.get_num_threads(): {torch.get_num_threads()}")
    main()
