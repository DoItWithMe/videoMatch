import yaml
from pydantic import BaseModel, Field
from utils.singleton import thread_safe_singleton
from .exception import exception_handler, ConfigsException
from typing import Optional


class LogConfig(BaseModel):
    name: str = Field(..., title="Logger name")
    level: str = Field(default="INFO", title="Logger level")
    path: str = Field(..., title="Log file path for all logs")
    # error_path: str = Field(default="", title="Log file path for error logs")
    enable_stdout: bool = Field(default=False, title="Enable stdout logging")
    enable_rotate: bool = Field(default=True, title="Enable log rotation")
    rotate_bytes: int = Field(default=500 * 1024 * 1024, title="Rotate bytes")
    max_files: int = Field(default=3, title="Max files")


class MilvusConfig(BaseModel):
    host: str = Field(..., title="milvus host")
    port: int = Field(default=19350, title="milvus port")
    usr: str = Field(default="", title="usr")
    password: str = Field(default="", title="password")
    token: str = Field(default="", title="milvus token")
    timeout: int = Field(default=10, title="milvus request timeout, seconds")

    db_name: str = Field(..., title="milvus database name")
    collection_name: str = Field(..., title="milvus collection name of [db]")
    index_name: str = Field(
        default="", title="milvus index name of [collection]"
    )


class ServerConfig_(BaseModel):
    log_cfg: LogConfig = Field(..., title="log config")
    milvus_embedding_cfg: MilvusConfig = Field(..., title="milvus embedding config")
    milvus_media_info_cfg: MilvusConfig = Field(..., title="milvus media config")


@exception_handler
@thread_safe_singleton
class ServerConfig:
    def __init__(self, cfg_path: str) -> None:
        """__init__ _summary_

        Args:
            cfg_path (str): _description_
        """
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
            self._svr_cfg = ServerConfig_(**cfg)

    def get_log_config(self) -> LogConfig:
        """get_log_config _summary_

        Returns:
            LogConfig: _description_
        """
        return self._svr_cfg.log_cfg

    def get_milvus_embedding_cfg(self) -> MilvusConfig:
        """get_milvus_embedding_cfg _summary_

        Returns:
            MilvusConfig: _description_
        """
        return self._svr_cfg.milvus_embedding_cfg

    def get_milvus_media_info_cfg(self) -> MilvusConfig:
        """get_milvus_media_info_cfg _summary_

        Returns:
            MilvusConfig: _description_
        """
        return self._svr_cfg.milvus_media_info_cfg


_svr_cfg: Optional[ServerConfig] = None


def init_server_config(cfg_path: str) -> None:
    """init_server_config _summary_

    Args:
        cfg_path (str): _description_
    """
    global _svr_cfg
    _svr_cfg = ServerConfig(cfg_path)


def get_server_config() -> ServerConfig:
    """get_server_config _summary_

    Raises:
        ConfigsException: _description_

    Returns:
        ServerConfig: _description_
    """
    global _svr_cfg
    if _svr_cfg == None:
        raise ConfigsException("server config need init before use")
    return _svr_cfg
