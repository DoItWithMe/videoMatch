import sys
import os
from unittest import result

_project_dipath: str = os.path.dirname((os.path.abspath(__file__)))
sys.path.append(_project_dipath)

import argparse
import torch
import numpy as np
import shutil

# import concurrent.futures
from models.iscnet_utils import create_isc_model, gen_img_feats_by_ISCNet
from ffmpeg.ffmpeg import FFmpeg
from ffmpeg.exception import FFmpegException, FFmpegEofException
from loguru import logger as log
from utils.time_utils import TimeRecorder

import time

from functools import wraps
import multiprocessing as mp
import io

DEVICE_LIST = ["cpu", "cuda"]


def exception_handler(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except FFmpegException as e:
            raise FFmpegException(f"get exception: {e}")
        except FFmpegEofException as e:
            log.info("get EOF")
            return
        except Exception as e:
            log.info(f"get FUCK ?: {e}")
            raise FFmpegException(f"get an unexpect exception: {e}")

    return wrapper


def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ffmpeg",
        type=str,
        help="ffmpeg bin file path",
        default="/data/jinzijian/VmatchDemo/assets/ffmpeg",
        required=False,
    )

    parser.add_argument("--device", type=str, help="cpu or cuda", default="cpu")

    parser.add_argument(
        "--output-dir",
        type=str,
        help="output dir path, auto-create it if not exisit",
        default="./vmatch_dnagen_output",
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
        "--fps",
        type=int,
        default=8,
        help="output fps when converting video to images",
    )

    parser.add_argument(
        "--input-videos-dir",
        "-i",
        type=str,
        help="input videos dir",
        required=True,
    )

    parser.add_argument(
        "--worker",
        type=int,
        help="ffmpeg worker num",
        default=1,
        required=False,
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

    assert os.path.exists(args.input_videos_dir)
    assert os.path.exists(args.isc_weight)


def init_log():
    log.remove()
    log.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
        enqueue=True,
    )


@exception_handler
def process_task(
    ffmpeg_bin_path,
    video_file_path,
    output_dir,
    fps,
    isc_weight_path,
    device,
) -> None:
    # (
    #     ffmpeg_bin_path,
    #     video_file_path,
    #     output_dir,
    #     fps,
    #     isc_weight_path,
    #     device,
    # ) = args

    isc_model, isc_processer = create_isc_model(
        weight_file_path=isc_weight_path, device=device, is_training=False
    )

    ffmpeg = FFmpeg(ffmpeg_bin_path, video_file_path, output_dir, fps)
    ffmpeg.start()
    img_list: list[io.BytesIO] = list()
    img_feats = list()
    while True:
        img = ffmpeg.get_img()
        if img is None:
            break
        img_list.append(img)

        if len(img_list) > 512:
            log.info(f"{video_file_path} get enough img...")
            tmp_img_feats = gen_img_feats_by_ISCNet(
                img_list[:512], isc_model, isc_processer
            )
            img_list = img_list[513:]
            img_feats.extend(tmp_img_feats)
            log.info(f"{video_file_path} get {len(tmp_img_feats)} feats...")

    tmp_img_feats = gen_img_feats_by_ISCNet(img_list, isc_model, isc_processer)
    img_feats.extend(tmp_img_feats)

    log.info(f"get {len(img_feats)} image")
    return None


def generate_imgs(
    ffmpeg_bin_path: str,
    videos_dir_path: str,
    output_dir: str,
    fps: int,
    executor_num: int,
    isc_weight_path: str,
    device: str,
):
    videos_files_path = [
        os.path.join(videos_dir_path, img_name)
        for img_name in os.listdir(videos_dir_path)
    ]

    args_list = list()

    for video_file_path in videos_files_path:
        args_list.append(
            (ffmpeg_bin_path, video_file_path, output_dir, fps, isc_weight_path, device)
        )

    args_list = args_list[:3]
    time_recorder = TimeRecorder()
    time_recorder.start_record()

    for i in range(0, len(args_list), executor_num):
        process_list: list[mp.Process] = list()
        for j in range(executor_num):
            p = mp.Process(target=process_task, args=(args_list[i]))
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()

    # process_pool = multiprocessing.Pool(processes=executor_num)
    # process_pool = mp.Pool(processes=executor_num)

    # result_async = process_pool.map_async(func=process_task, iterable=args_list)

    # result_async.wait()
    # process_pool.close()
    # process_pool.join()

    time_recorder.end_record()

    log.info(
        f"generate imgs of {len(videos_files_path)} videos cost {time_recorder.get_total_duration_miliseconds()} ms"
    )


def save_feats(feats_save_path, feats):
    tmp_dir = os.path.dirname(feats_save_path)
    os.makedirs(tmp_dir, exist_ok=True)
    log.info(f"save feats: {feats.shape} to {feats_save_path} ")
    np.save(feats_save_path, feats)


# def generate_feats(
#     imgs_output_dir: str, feats_output_dir: str, isc_weight_path: str, device: str
# ):
#     isc_model, isc_processer = create_isc_model(
#         weight_file_path=isc_weight_path, device=device, is_training=False
#     )

#     feats_time_recorder = TimeRecorder()
#     save_time_recorder = TimeRecorder()
#     imgs_count = 0

#     tmp_imgs_output_dir = [
#         os.path.join(imgs_output_dir, title) for title in os.listdir(imgs_output_dir)
#     ]

#     for img_dir_path in tmp_imgs_output_dir:
#         imgs_list = os.listdir(img_dir_path)
#         # sort imgs for listdir may not return by the number order...
#         # print(imgs_list)

#         tmp_imgs_list = sorted(imgs_list, key=lambda x: int(x.split(".")[0]))

#         imgs_list.clear()
#         imgs_list = [os.path.join(img_dir_path, img_name) for img_name in tmp_imgs_list]
#         imgs_count += len(imgs_list)

#         log.info(f"start gen feats, imgs len: {len(imgs_list)}")
#         # it's very slow when use cpu to generate image feats....
#         feats_time_recorder.start_record()
#         ref_isc_feats = gen_img_feats_by_ISCNet(
#             imgs_list, isc_model, isc_processer, device
#         )
#         feats_time_recorder.end_record()
#         log.info(f"get feats: {ref_isc_feats.shape}")

#         feats_file_path = os.path.join(
#             feats_output_dir,
#             f"{os.path.splitext(os.path.basename(img_dir_path))[0]}.npy",
#         )

#         save_time_recorder.start_record()
#         save_feats(feats_file_path, ref_isc_feats)
#         save_time_recorder.end_record()

#     log.info(
#         f"generate iscNet feats by {device} of {imgs_count} imgs cost {feats_time_recorder.get_total_duration_miliseconds()} ms, save it cost {save_time_recorder.get_total_duration_miliseconds()} ms"
#     )


def main():
    mp.set_start_method("spawn")
    init_log()
    args = parser_args()
    check_args(args)

    videos_dir_path = os.path.normpath(args.input_videos_dir)
    ffmpeg_path = args.ffmpeg
    output_dir = args.output_dir
    fps = args.fps

    isc_weight_path = args.isc_weight
    device = args.device

    output_base_name = f"{os.path.basename(videos_dir_path)}_fps_{fps}"
    imgs_output_dir = os.path.join(output_dir, "imgs", output_base_name)
    ffmpeg_worker_num = args.worker

    log.info(f"star generate imgs with {ffmpeg_worker_num} worker ...")
    shutil.rmtree(imgs_output_dir, ignore_errors=True)
    os.makedirs(imgs_output_dir, exist_ok=True)
    generate_imgs(
        ffmpeg_path,
        videos_dir_path,
        imgs_output_dir,
        fps,
        ffmpeg_worker_num,
        isc_weight_path,
        device,
    )

    # feats_output_dir = os.path.join(output_dir, "feats", output_base_name)
    # log.info(f"star generate ref feats...")
    # shutil.rmtree(feats_output_dir, ignore_errors=True)
    # os.makedirs(feats_output_dir, exist_ok=True)

    # generate_feats(feats_output_dir, feats_output_dir, isc_weight_path, device)
    # log.info("all feats extract done.....")


if __name__ == "__main__":
    main()
