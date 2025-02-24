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


from models.iscnet_utils import create_isc_model, gen_img_feats_by_ISCNet
from models.transvcl_utils import create_transvcl_model, gen_match_segments_by_transVCL
from models.transform_feats import trans_isc_features_to_transVCL_fromat
from ffmpeg.ffmpeg import extract_imgs
from loguru import logger as log
from utils.time_utils import TimeRecorder
from typing import Any

import torch.multiprocessing as mp


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

    parser.add_argument("--device", type=str, help="cpu or cuda", default="cuda")

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
        rimgs_list = [
            os.path.join(img_dir_path, img_name)
            for img_name in os.listdir(img_dir_path)
        ]

        imgs_count += len(rimgs_list)

        log.info(f"start gen feats, imgs len: {len(rimgs_list)}")
        # it's very slow when use cpu to generate image feats....
        feats_time_recorder.start_record()
        ref_isc_feats = gen_img_feats_by_ISCNet(
            rimgs_list, isc_model, isc_processer, device
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


def _excute_transform(params):
    sample_name, sample_feats, ref_feats_dict, segment_length = params
    transvcl_batch_feats_list = list()
    for ref_name, ref_feats in ref_feats_dict.items():
        transvcl_batch_feats = trans_isc_features_to_transVCL_fromat(
            sample_feats, ref_feats, f"{sample_name} Vs {ref_name}", segment_length
        )

        if len(transvcl_batch_feats) == 0:
            continue

        transvcl_batch_feats_list.append(transvcl_batch_feats)
    return transvcl_batch_feats_list


def _query_matched_videos(
    transvcl_weight_path: str,
    confthre: int,
    nmsthre: int,
    img_size: tuple[Any, Any],
    device: str,
    sample_feats_dict: dict[str, Any],
    ref_feats_dict: dict[str, Any],
    segment_length: int,
    frame_interval: float,
    worker_num: int,
):

    transform_time_recorder = TimeRecorder()
    compare_time_recorder = TimeRecorder()

    transvcl_model = create_transvcl_model(
        weight_file_path=transvcl_weight_path, device=device, is_training=False
    )

    for sample_name, sample_feats in sample_feats_dict.items():
        if sample_name != "EmpressesInThePalace_2" and sample_name != "EmpressesInThePalace_3":
            continue

        for ref_name, ref_feats in ref_feats_dict.items():
            if ref_name != "EmpressesInThePalaceR_2":
                continue

            titile = f"{sample_name},{ref_name}"

            log.info(f"{titile} start spilit feats to segment....")
            transform_time_recorder.start_record()
            transvcl_batch_feats = trans_isc_features_to_transVCL_fromat(
                sample_feats, ref_feats, titile, segment_length
            )
            transform_time_recorder.end_record()

            if len(transvcl_batch_feats) == 0:
                log.warning(f"{titile} skip for no segments....")
                continue

            log.info(f"{titile} start to locate copied segment ")

            compare_time_recorder.start_record()
            matched_segments = gen_match_segments_by_transVCL(
                transvcl_model,
                transvcl_batch_feats,
                confthre,
                nmsthre,
                img_size,
                segment_length,
                frame_interval,
                frame_interval,
                device,
            )
            compare_time_recorder.end_record()

            for matched_seg_title in matched_segments:
                log.info(f"matched_segments: {matched_segments[matched_seg_title]}")

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
        shutil.rmtree(ref_imgs_output_dir, ignore_errors=True)
        shutil.rmtree(sample_imgs_output_dir, ignore_errors=True)
        os.makedirs(ref_imgs_output_dir, exist_ok=True)
        os.makedirs(sample_imgs_output_dir, exist_ok=True)
        executor_num = 2

        log.info(f"star generate ref imgs with {executor_num} worker ...")
        _generate_imgs(
            ffmpeg_path, ref_videos_dir_path, ref_imgs_output_dir, fps, executor_num
        )

        log.info(f"star generate sample imgs with {executor_num} worker ...")
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
        shutil.rmtree(ref_feats_output_dir, ignore_errors=True)
        shutil.rmtree(sample_feats_output_dir, ignore_errors=True)
        os.makedirs(ref_feats_output_dir, exist_ok=True)
        os.makedirs(sample_feats_output_dir, exist_ok=True)

        log.info(f"star generate ref feats...")
        _generate_feats(
            ref_imgs_output_dir, ref_feats_output_dir, isc_weight_path, device
        )

        log.info(f"star generate sample feats...")
        _generate_feats(
            sample_imgs_output_dir, sample_feats_output_dir, isc_weight_path, device
        )
    else:
        log.info(
            f"use cached feats: {ref_feats_output_dir} and {sample_feats_output_dir}"
        )

    # 3. 加载特征

    log.info(f"start load feats in {ref_feats_output_dir}")
    ref_feats_dict = _load_feats(ref_feats_output_dir)

    log.info(f"start load feats in {sample_feats_output_dir}")
    sample_feats_dict = _load_feats(sample_feats_output_dir)

    # 4. 查询
    worker_num = 5
    _query_matched_videos(
        transvcl_weight_path,
        confthre,
        nmsthre,
        img_size,
        device,
        sample_feats_dict,
        ref_feats_dict,
        segment_length,
        frame_interval,
        worker_num,
    )


if __name__ == "__main__":
    print(f"torch.get_num_threads(): {torch.get_num_threads()}")
    main()
