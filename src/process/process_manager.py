import multiprocessing as mp
import queue
import traceback
from time import sleep
from typing import Any, Callable, Optional, Tuple

from loguru import logger as log

from log.log import init_logger

from .exception import exception_handler
from configs.configs import init_server_config, get_server_config, ServerConfig


def call_func(
    args_queue: queue.Queue[Optional[Tuple]],
    results_queue: queue.Queue[Optional[Tuple | Any]],
    status_queue: queue.Queue[Optional[str]],
    target_func: Callable,
    loglevel: str,
):
    try:
        svr_cfg: ServerConfig = get_server_config()
        init_logger(svr_cfg.get_log_config())
        normal_quit = True
        while True:
            args = args_queue.get()
            if args is None:
                break
            res = target_func(*args)
            results_queue.put(res)
    except Exception as e:
        status_queue.put(traceback.format_exc())
        normal_quit = False
        log.trace(f"subprocess failed for {e}")
    finally:
        if normal_quit:
            log.trace("subprocess success")
            status_queue.put(None)


class ProcessManager:
    @exception_handler
    def __init__(
        self,
        process_num: int,
        name: Optional[str],
        target_func: Callable,
        target_func_args_list: list[Tuple],
        loglevel: str,
    ) -> None:
        self.__target_func: Callable = target_func
        self.__target_func_args_list: list[Tuple] = target_func_args_list

        self.__process_num: int = process_num
        self.__process_name: Optional[str] = name

        self.__manager = mp.Manager()

        self.__args_queue: queue.Queue[Optional[Tuple]] = self.__manager.Queue(
            maxsize=len(self.__target_func_args_list) + self.__process_num
        )

        self.__results_queue: queue.Queue[Optional[Tuple | Any]] = self.__manager.Queue(
            maxsize=len(self.__target_func_args_list)
        )

        self.__process_status_queue: queue.Queue[Optional[str]] = self.__manager.Queue(
            maxsize=self.__process_num
        )

        self.__process_list: list[mp.Process] = list()

        for _ in range(self.__process_num):
            p = mp.Process(
                target=call_func,
                name=self.__process_name,
                args=(
                    self.__args_queue,
                    self.__results_queue,
                    self.__process_status_queue,
                    self.__target_func,
                    loglevel,
                ),
            )
            p.start()
            self.__process_list.append(p)

    @exception_handler
    def start(self) -> None:
        log.trace("distribute task")
        for i in range(0, len(self.__target_func_args_list)):
            self.__args_queue.put(self.__target_func_args_list[i])

        for _ in range(self.__process_num):
            self.__args_queue.put(None)

        log.trace("distribute tasks done")
        return None

    @exception_handler
    def wait(self) -> None:
        log.trace("wait all process")
        while not self.__process_status_queue.full():
            sleep(0.5)
        log.trace("all process done")

    @exception_handler
    def get_results(self) -> list[Optional[Tuple | Any]]:
        res_list: list[Optional[Tuple | Any]] = list()
        while not self.__results_queue.empty():
            res_list.append(self.__results_queue.get())
        return res_list

    @exception_handler
    def get_running_status(self) -> list[Optional[str]]:
        status_list: list[Optional[str]] = list()
        while not self.__process_status_queue.empty():
            status_list.append(self.__process_status_queue.get())
        return status_list
