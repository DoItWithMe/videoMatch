import sys
import os

from torch import embedding

_project_dipath: str = os.path.dirname((os.path.abspath(__file__)))
sys.path.append(_project_dipath)

import argparse
from configs.configs import init_server_config, get_server_config, ServerConfig
from log.log import init_logger
from loguru import logger
from milvus.milvus_manager import (
    init_milvus_client_manager,
    get_milvus_client_manager,
    MilvusClientManager,
)

# tmp
from typing import Any
import numpy as np
from numpy import ndarray
from utils.time_utils import TimeRecorder
from results_handler.query_results_handler import get_matched_video_segments, get_matched_video_segments_parallel
from milvus.schema import EmbeddingsInfo

def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        help="output dir path, auto-create it if not exisit",
        default="./output",
        required=False,
    )

    parser.add_argument(
        "--sample-videos-dir",
        "-s",
        type=str,
        help="input sample media videos dir",
        required=True,
    )

    parser.add_argument(
        "--reference-videos-dir",
        "-r",
        type=str,
        help="input reference media videos dir",
        required=True,
    )

    parser.add_argument(
        "--config-path",
        "-c",
        type=str,
        help="config path",
        required=True,
    )

    return parser.parse_args()


def load_feats(feats_store_dir_path: str) -> dict[str, Any]:
    feat_dict: dict[str, Any] = dict()
    feats_files_list: list[str] = [
        os.path.join(feats_store_dir_path, file_name)
        for file_name in os.listdir(feats_store_dir_path)
    ]

    count = 0
    for feats_file in feats_files_list:
        feat_dict[os.path.splitext(os.path.basename(feats_file))[0]] = np.load(
            feats_file
        )
        count += 1
    logger.info(f"load feats of {feats_store_dir_path}")

    return feat_dict


def add_ref(ref_feat_dict: dict[str, ndarray]):
    milvus_manager: MilvusClientManager = get_milvus_client_manager()
    time_recorder = TimeRecorder()
    for ref_name, ref_feat in ref_feat_dict.items():
        logger.info(f"start add ref embeddings of {ref_name}")
        time_recorder.start_record()
        milvus_manager.add_ref_embedding(ref_name=ref_name, ref_embeddings=ref_feat)
        time_recorder.end_record()

    logger.info(
        f"add {len(list(ref_feat_dict.keys()))} refs, avg cost: {time_recorder.get_avg_duration_miliseconds()} ms"
    )


def query(sample_feat_dict: dict[str, ndarray]):
    segment_len_limit = 8 * 3
    sample_name_list = list(sample_feat_dict.keys())
    # sample_name_list = sorted(sample_name_list, key=lambda x: int(x.split("_")[1]))
    
    milvus_manager: MilvusClientManager = get_milvus_client_manager()
    l2_dis_thresh = 1.1
    for sample_name in sample_name_list:
        # sample_name = "半熟男女11"
        
        logger.info(f"start queyr {sample_name}")
        sample_feat = sample_feat_dict[sample_name]
        
        embedding_query_recorder = TimeRecorder()
        embedding_query_recorder.start_record()
        results: list[list[EmbeddingsInfo]] = milvus_manager.get_matched_embeddings(
            sample_feat.tolist(), 3 * 3 * 8
        )
        embedding_query_recorder.end_record()
        
        matched_res_recorder = TimeRecorder()
        matched_res_recorder.start_record()
        get_matched_video_segments_parallel(
            l2_dis_thresh,
            results,
            segment_len_limit,
            sample_name,
        )
        matched_res_recorder.end_record()
        logger.info((f"query milvus cost: {embedding_query_recorder.get_total_duration_miliseconds()} ms, "
                     f"get matched results cost: {matched_res_recorder.get_total_duration_miliseconds()} ms"))
        break


def main() -> None:
    svr_args = parser_args()
    init_server_config(svr_args.config_path)
    svr_cfg: ServerConfig = get_server_config()
    init_logger(svr_cfg.get_log_config())
    init_milvus_client_manager(
        svr_cfg.get_milvus_embedding_cfg(), svr_cfg.get_milvus_media_info_cfg()
    )

    # just for test
    fps = 8
    output_dir: str = svr_args.output_dir

    ref_videos_dir_path = os.path.normpath(svr_args.reference_videos_dir)
    sample_videos_dir_path = os.path.normpath(svr_args.sample_videos_dir)

    ref_output_base_name: str = f"{os.path.basename(ref_videos_dir_path)}_fps_{fps}"
    sample_output_base_name: str = (
        f"{os.path.basename(sample_videos_dir_path)}_fps_{fps}"
    )

    ref_feat_output_dir: str = os.path.join(output_dir, "feats", ref_output_base_name)
    sample_feat_output_dir: str = os.path.join(
        output_dir, "feats", sample_output_base_name
    )

    # ref_feat_dict: dict[str, ndarray] = load_feats(ref_feat_output_dir)
    # add_ref(ref_feat_dict)

    sample_feat_dict: dict[str, ndarray] = load_feats(sample_feat_output_dir)

    query(sample_feat_dict)


if __name__ == "__main__":
    main()
