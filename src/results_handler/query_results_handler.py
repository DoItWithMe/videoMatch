from loguru import logger
from utils.time_utils import milliseconds_to_hhmmss
from milvus.milvus_manager import (
    get_milvus_client_manager,
    MilvusClientManager,
)
from .exception import exception_handler
from milvus.schema import MediaInfo, EmbeddingsInfo
from .schema import QueryResult, MediaDes

from process.process_manager import ProcessManager


class QueryResultHandler:
    def __init__(self, segment_len_limit: int):
        self.__query_result_dict: dict[str, list[QueryResult]] = dict()
        self.__quick_search_result_dict: dict[str, list[int]] = dict()

        self.__segment_len_limit: int = segment_len_limit
        self.print_flag: bool = False

    @classmethod
    def __could_merge(cls, s, e, l, u) -> bool:
        if s <= l and e >= l:
            return True

        if s >= l and s <= u:
            return True

        return False

    @classmethod
    def __merge_distance(
        cls, s1: int, e1: int, s2: int, e2: int, dis1: float, dis2: float
    ) -> float:
        len1 = e1 - s1
        len2 = e2 - s2
        if len1 == 0 or len2 == 0:
            return dis1 + dis2

        avg_dis1 = dis1 / len1
        avg_dis2 = dis2 / len2

        if e1 <= s2:
            return dis1 + dis2
        elif s1 < s2 and e1 <= e2:
            # return avg_dis1 * (e1 / 2 - s1 + s2 / 2) + avg_dis2 * (
            #     e2 - e1 / 2 - s2 / 2
            # )
            return dis1 + avg_dis2 * (e2 - e1)
        elif s1 < s2 and e1 > e2:
            # return dis2 / 2 + avg_dis1 * (len1 - len2 / 2)
            return dis1
        elif s1 >= s2 and e1 <= e2:
            # return dis1 / 2 + avg_dis2 * (len2 - len1 / 2)
            return dis2
        elif s1 >= s2 and e1 > e2:
            # return dis1 * (e1 - s1 / 2 - e2 / 2) + dis2 * (e2 / 2 + s1 / 2 - s2)
            return dis1 + avg_dis2 * (s1 - s2)
        elif s1 >= e2:
            return dis1 + dis2
        else:
            return -1.0

    def __merge_match_frame(
        self,
        ref_uuid: str,
        new_res_list: list[QueryResult],
    ) -> None:
        new_res_dict: dict[int, list[QueryResult]] = dict()
        for res in new_res_list:
            if res.ref_start_index not in new_res_dict.keys():
                new_res_dict[res.ref_start_index] = list()
            new_res_dict[res.ref_start_index].append(res)

        src_query_res_list = self.__query_result_dict[ref_uuid]
        src_quick_search_list = self.__quick_search_result_dict[ref_uuid]
        src_quick_search_remove_list = list()

        src_query_res_list_len: int = len(src_query_res_list)
        new_res_list_len: int = len(new_res_list)
        min_new_ref_start_index = -1

        if self.print_flag:
            logger.info(
                f"quick search list: {src_quick_search_list}, src_query_res_list: {len(src_query_res_list)}"
            )
            logger.info("src_res_list " * 5)
            for r in src_quick_search_list:
                r = self.__query_result_dict[ref_uuid][r]
                logger.info(
                    f"src s id: {r.sample_start_index} - {r.sample_end_index}, src ref id: {r.ref_start_index} - {r.ref_end_index}, remove: {r.remove_flag}"
                )

            logger.info("new_res_list " * 5)
            for r in new_res_list:
                logger.info(
                    f"new s id: {r.sample_start_index} - {r.sample_end_index}, new ref id: {r.ref_start_index} - {r.ref_end_index}"
                )
            logger.info("=== " * 5)

        for i in src_quick_search_list:
            src_query_res = src_query_res_list[i]
            cur_ref_id = src_query_res.ref_end_index

            cur_sample_id = src_query_res.sample_end_index
            target_sample_id = cur_sample_id + 1

            candidate_query_result = None
            target_ref_id = cur_ref_id + 0
            for j in range(1, 2):
                tmp_target_ref_id = target_ref_id + j
                # logger.info(f"tmp_target_ref_id: {tmp_target_ref_id}")
                if (
                    tmp_target_ref_id in new_res_dict.keys()
                    and new_res_dict[tmp_target_ref_id][0].sample_start_index
                    == target_sample_id
                ):
                    candidate_query_result = new_res_dict[tmp_target_ref_id][0]
                    break

            if candidate_query_result is None:
                src_quick_search_remove_list.append(i)
                continue

            src_query_res.distance += candidate_query_result.distance
            # src_query_res.ref_end_index = target_ref_id
            src_query_res.ref_end_index = target_ref_id + j
            src_query_res.sample_end_index = candidate_query_result.sample_end_index
            candidate_query_result.remove_flag = True

            if min_new_ref_start_index == -1:
                min_new_ref_start_index = target_ref_id
            else:
                min_new_ref_start_index = min(min_new_ref_start_index, target_ref_id)

        left_res_list = [
            new_res_list[i]
            for i in range(0, new_res_list_len)
            if new_res_list[i].ref_start_index > min_new_ref_start_index
            and new_res_list[i].remove_flag == False
        ]

        left_res_list_len = len(left_res_list)
        src_query_res_list.extend(left_res_list)

        for i in src_quick_search_remove_list:
            src_quick_search_list.remove(i)

        new_quick_search_list = [
            i
            for i in range(
                src_query_res_list_len, src_query_res_list_len + left_res_list_len
            )
        ]
        src_quick_search_list.extend(new_quick_search_list)

        if self.print_flag:
            logger.info("frame result " * 5)
            for r in src_quick_search_list:
                r = self.__query_result_dict[ref_uuid][r]
                logger.info(
                    f"result s id: {r.sample_start_index} - {r.sample_end_index}, result ref id: {r.ref_start_index} - {r.ref_end_index}, remove: {r.remove_flag}"
                )

    def __merge_match_segments(
        self,
        ref_uuid: str,
    ):

        merge_query_res_list: list[QueryResult] = list()
        query_result_list = self.__query_result_dict[ref_uuid]
        query_result_list_len = len(query_result_list)

        for i in range(0, query_result_list_len):
            if query_result_list[i].remove_flag == True:
                if self.print_flag:
                    logger.info(f"skip s: {query_result_list[i]}")
                continue

            merge_query_res = query_result_list[i]
            sample_threadhold = int(
                (
                    merge_query_res.sample_end_index
                    - merge_query_res.sample_start_index
                    + 1
                )
                * 0.5
            )

            if sample_threadhold <= 0:
                sample_threadhold = 5

            for j in range(i + 1, query_result_list_len):
                if query_result_list[j].remove_flag == True:
                    continue

                next_query_res = query_result_list[j]

                sample_lower_boud = (
                    next_query_res.sample_start_index - sample_threadhold
                )
                sample_upper_boud = next_query_res.sample_end_index + sample_threadhold

                if sample_lower_boud < 0:
                    sample_lower_boud = 0

                if self.print_flag:
                    logger.info(
                        f"111, src s: {merge_query_res.sample_start_index} - {merge_query_res.sample_end_index}, next s: {sample_lower_boud} - {sample_upper_boud}, next ss: {next_query_res.sample_start_index } - {next_query_res.sample_end_index}, sample_threadhold: {sample_threadhold}, could merge: {self.__could_merge(merge_query_res.sample_start_index,merge_query_res.sample_end_index,sample_lower_boud,sample_upper_boud)}"
                    )

                if self.__could_merge(
                    merge_query_res.sample_start_index,
                    merge_query_res.sample_end_index,
                    sample_lower_boud,
                    sample_upper_boud,
                ):
                    ref_threadhold = int(
                        (
                            merge_query_res.ref_end_index
                            - merge_query_res.ref_start_index
                        )
                        * 0.5
                    )
                    if ref_threadhold <= 0:
                        ref_threadhold = 5 * 10

                    ref_lower_bound = next_query_res.ref_start_index - ref_threadhold
                    ref_upper_bound = next_query_res.ref_end_index + ref_threadhold

                    if ref_lower_bound < 0:
                        ref_lower_bound = 0

                    if self.print_flag:
                        logger.info(
                            f"222, src r: {merge_query_res.ref_start_index} - {merge_query_res.ref_end_index}, next r: {ref_lower_bound} - {ref_upper_bound}, next rr: {next_query_res.ref_start_index} - {next_query_res.ref_end_index} could merge: {self.__could_merge(merge_query_res.ref_start_index,merge_query_res.ref_end_index,ref_lower_bound,ref_upper_bound)}"
                        )

                    if self.__could_merge(
                        merge_query_res.ref_start_index,
                        merge_query_res.ref_end_index,
                        ref_lower_bound,
                        ref_upper_bound,
                    ):
                        new_s_s = min(
                            merge_query_res.sample_start_index,
                            next_query_res.sample_start_index,
                        )
                        new_s_e = max(
                            merge_query_res.sample_end_index,
                            next_query_res.sample_end_index,
                        )

                        new_r_s = min(
                            merge_query_res.ref_start_index,
                            next_query_res.ref_start_index,
                        )
                        new_r_e = max(
                            merge_query_res.ref_end_index,
                            next_query_res.ref_end_index,
                        )

                        if self.print_flag:
                            logger.info(
                                f"new s: {new_s_s} - {new_s_e}, new r: {new_r_s} - {new_r_e}, new_r_e - new_r_s == (new_s_e - new_s_s) * 10: {new_r_e - new_r_s == (new_s_e - new_s_s) * 10}"
                            )

                        next_query_res.remove_flag = True
                        if self.print_flag:
                            logger.info(
                                f"res is removed: s: {next_query_res.sample_start_index} - {next_query_res.sample_end_index}, r: {next_query_res.ref_start_index} - {next_query_res.ref_end_index}"
                            )
                        merge_query_res.distance = self.__merge_distance(
                            merge_query_res.ref_start_index,
                            merge_query_res.ref_end_index,
                            next_query_res.ref_start_index,
                            next_query_res.ref_end_index,
                            merge_query_res.distance,
                            next_query_res.distance,
                        )
                        merge_query_res.sample_start_index = new_s_s
                        merge_query_res.sample_end_index = new_s_e
                        merge_query_res.ref_start_index = new_r_s
                        merge_query_res.ref_end_index = new_r_e
                        sample_threadhold = int((new_s_e - new_s_s) * 0.5)
                else:
                    if merge_query_res.sample_end_index > sample_upper_boud:
                        break

            merge_query_res_list.append(merge_query_res)

        self.__query_result_dict[ref_uuid].clear()
        self.__query_result_dict[ref_uuid].extend(merge_query_res_list)
        if self.print_flag:
            logger.info(" Merge Results -- " * 5)
            for i in merge_query_res_list:
                logger.info(
                    f"s: {i.sample_start_index} - {i.sample_end_index}, r: {i.ref_start_index} - {i.ref_end_index}"
                )

    @exception_handler
    def merge_match_frame(
        self,
        new_res_dict: dict[str, list[QueryResult]],
    ):
        for new_ref_uuid in new_res_dict.keys():
            new_res_list = new_res_dict[new_ref_uuid]
            new_res_list.sort(key=lambda res: res.ref_start_index)
            if new_ref_uuid not in self.__query_result_dict.keys():
                self.__query_result_dict[new_ref_uuid] = list()
                self.__query_result_dict[new_ref_uuid].extend(new_res_list)
                self.__quick_search_result_dict[new_ref_uuid] = [
                    i for i in range(len(self.__query_result_dict[new_ref_uuid]))
                ]
            else:
                self.__merge_match_frame(new_ref_uuid, new_res_list)

    @exception_handler
    def merge_match_segments(self):
        for ref_uuid in self.__query_result_dict.keys():
            self.__merge_match_segments(ref_uuid)

    @exception_handler
    def filter_valid_match_segment(self):
        for ref_uuid in self.__query_result_dict.keys():
            src_list = self.__query_result_dict[ref_uuid]
            if len(src_list) == 0:
                continue
            # src_list.sort(key=lambda res: res.sample_start_index)
            tmp_list = list()
            for i in range(len(src_list)):
                cur_res = src_list[i]
                if (
                    cur_res.sample_end_index - cur_res.sample_start_index + 1
                    >= self.__segment_len_limit
                ):
                    tmp_list.append(cur_res)

            self.__query_result_dict[ref_uuid][:] = []
            self.__query_result_dict[ref_uuid].extend(tmp_list)

    @exception_handler
    def just_dump(self, with_time: bool, sample_name: str, sample_total_len: int):
        logger.info(f"sample name: {sample_name} matched with:::: ")
        s_matched_duration_ms_t = 0.0
        for ref_uuid in self.__query_result_dict.keys():
            if len(self.__query_result_dict[ref_uuid]) <= 0:
                continue

            logger.info(f"ref: {ref_uuid} -----")
            s_matched_duration_ms = 0.0
            for res in self.__query_result_dict[ref_uuid]:
                if not with_time:
                    logger.info(
                        f"s: {res.sample_start_index} - {res.sample_end_index}, ref: {res.ref_start_index} - {res.ref_end_index}, remove: {res.remove_flag}"
                    )
                else:
                    s_s_t = res.sample_start_index * 125
                    s_e_t = (res.sample_end_index + 1) * 125
                    r_s_t = res.ref_start_index * 125
                    r_e_t = (res.ref_end_index + 1) * 125
                    s_matched_duration_ms += s_e_t - s_s_t

                    s_s_t_t = milliseconds_to_hhmmss(s_s_t)
                    s_e_t_t = milliseconds_to_hhmmss(s_e_t)
                    r_s_t_t = milliseconds_to_hhmmss(r_s_t)
                    r_e_t_t = milliseconds_to_hhmmss(r_e_t)

                    logger.info(
                        f"s: {s_s_t_t} - {s_e_t_t}, ref: {r_s_t_t} - {r_e_t_t}, remove: {res.remove_flag}"
                    )

            s_matched_duration_ms_t += s_matched_duration_ms
            logger.info(f"-- sample: {sample_name} matched with ref: {ref_uuid} ---")
            logger.info(
                f"sample duration: {(sample_total_len * 125 /1000.0):.3f} s, valid duration: {(s_matched_duration_ms/1000.0):.3f} -- {(s_matched_duration_ms/sample_total_len / 125.0 * 100.0 ):.2f}%\n"
            )
        logger.info(
            f"sample: {sample_name}, duration: {(sample_total_len * 125 /1000.0):.3f} s, total valid duration: {(s_matched_duration_ms_t/1000.0):.3f} -- {(s_matched_duration_ms_t/sample_total_len / 125.0 * 100.0 ):.2f}%\n\n"
        )


@exception_handler
def get_matched_video_segments(
    l2_dis_thresh: float,
    results: list[list[EmbeddingsInfo]],
    segment_len_limit: int,
    sample_name: str,
    media_des_cache_dict: dict[str, MediaDes],
    sample_offset: int,
):
    logger.info(f"start? sample_offset: {sample_offset}")
    query_result_handler = QueryResultHandler(segment_len_limit)
    tmp_query_result_dict: dict[str, list[QueryResult]] = dict()

    
    for i in range(len(results)):
        tmp_query_result_dict.clear()
        for j in range(len(results[i])):
            cur_res: EmbeddingsInfo = results[i][j]
            l2_dis = cur_res.distance
            if l2_dis > l2_dis_thresh:
                break

            ref_frame = cur_res.frame
            ref_uuid = cur_res.uuid

            media_des = None
            if ref_uuid not in media_des_cache_dict.keys():
                raise Exception("not found media des....")

            media_des = media_des_cache_dict[ref_uuid]

            res = QueryResult(
                ref_start_index=ref_frame,
                ref_end_index=ref_frame,
                distance=l2_dis,
                sample_start_index=i + sample_offset,
                sample_end_index=i + sample_offset,
            )

            if media_des.uuid not in tmp_query_result_dict.keys():
                tmp_query_result_dict[media_des.uuid] = list()
            tmp_query_result_dict[media_des.uuid].append(res)

        query_result_handler.merge_match_frame(tmp_query_result_dict)

    query_result_handler.filter_valid_match_segment()
    query_result_handler.merge_match_segments()
    query_result_handler.filter_valid_match_segment()

    samle_frame_len = len(results)
    query_result_handler.just_dump(True, sample_name, samle_frame_len)
    logger.info("finish")
    return query_result_handler, 1


@exception_handler
def get_matched_video_segments_parallel(
    l2_dis_thresh: float,
    results: list[list[EmbeddingsInfo]],
    segment_len_limit: int,
    sample_name: str,
) -> None:
    # worker_num 需要可配置
    worker_num = 1
    logger.info(f"start get matched video segments parallel, worker_num: {worker_num}")
    # 其他的 step?
    step = 1024
    args_list = list()
    results_len = len(results)
    count = 0

    milvus_manager: MilvusClientManager = get_milvus_client_manager()
    media_des_cache_dict: dict[str, MediaDes] = dict()
    for i in range(0, results_len):
        count += len(results[i])
        for cur_res in results[i]:
            ref_uuid = cur_res.uuid
            if ref_uuid not in media_des_cache_dict.keys():
                ref_info: MediaInfo = milvus_manager.get_media_info(ref_uuid)
                media_des = MediaDes(
                    name=ref_info.filename,
                    uuid=ref_info.uuid,
                    frame_total_len=ref_info.frame_total_len,
                    avg_frame_time=125.0,
                )
                media_des_cache_dict[ref_uuid] = media_des

    for i in range(0, results_len, step):
        args_list.append(
            (
                l2_dis_thresh,
                results[i : i + step],
                segment_len_limit,
                sample_name,
                media_des_cache_dict,
                i,
            )
        )

    logger.info("args list build ok, create process manager")
    process_manaer = ProcessManager(
        process_num=worker_num,
        name="vmatch dnagen worker",
        target_func=get_matched_video_segments,
        target_func_args_list=args_list,
        loglevel="INFO",
    )

    logger.info("process_manaer start")
    process_manaer.start()
    logger.info("process_manaer wait")
    process_manaer.wait()

    logger.info("process_manager finish")
    meatch_res_list = process_manaer.get_results()
    logger.info(
        f"args: {len(args_list)}, res: {len(meatch_res_list)}, handle total {count} frames.."
    )
