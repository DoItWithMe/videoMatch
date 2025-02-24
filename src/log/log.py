# coding: utf-8
import sys
import uuid
from contextvars import ContextVar
from typing import Optional, Literal

from loguru import logger

from configs.configs import LogConfig
from .exception import exception_handler

_x_request_id: ContextVar[str] = ContextVar("x_request_id", default="global")


@exception_handler
def set_request_id(req_id: Optional[str] = None) -> ContextVar[str]:
    """set_request_id _summary_

    Args:
        req_id (Optional[str], optional): _description_. Defaults to None.

    Returns:
        ContextVar[str]: _description_
    """
    if req_id is None:
        req_id = uuid.uuid4().hex
    _x_request_id.set(req_id)
    return _x_request_id


@exception_handler
def get_request_id() -> str:
    """get_request_id _summary_

    Returns:
        str: _description_
    """
    return _x_request_id.get()


@exception_handler
def __logger_filter(record) -> Literal[True]:
    """__logger_filter _summary_

    Args:
        record (_type_): _description_

    Returns:
        Literal[True]: _description_
    """
    request_id = _x_request_id.get()
    record["extra"]["request_id"] = request_id
    return True


@exception_handler
def init_logger(log_cfg: LogConfig) -> None:
    """init_logger _summary_

    Args:
        log_cfg (LogConfig): _description_
    """
    logger.remove()
    log_format = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {extra[request_id]} | {name}:{function}:{line} - {message}"

    if log_cfg.enable_stdout:
        logger.add(
            sys.stdout, level=log_cfg.level, filter=__logger_filter, format=log_format
        )

    if len(log_cfg.path) > 0:
        logger.add(
            log_cfg.path,
            level=log_cfg.level,
            rotation=log_cfg.rotate_bytes if log_cfg.enable_rotate else None,
            compression="zip" if log_cfg.enable_rotate else None,
            retention=log_cfg.max_files if log_cfg.enable_rotate else None,
            filter=__logger_filter,
            format=log_format,
        )

    logger.bind(name=log_cfg.name)
