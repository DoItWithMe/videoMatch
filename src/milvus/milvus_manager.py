from pymilvus import MilvusClient
from typing import Optional, Any, Union, List, Dict
import uuid
from .exception import MilvusException, exception_handler
from utils.singleton import thread_safe_singleton
from configs.configs import MilvusConfig
from numpy import ndarray
from .schema import MediaInfo, EmbeddingsInfo

# from pymilvus import DataType

_vector_no_use: list[float] = [0.0]


@thread_safe_singleton
class MilvusClientManager:
    @exception_handler
    def __init__(
        self, milvus_embedding_cfg: MilvusConfig, milvus_media_info_cfg: MilvusConfig
    ) -> None:
        """__init__ _summary_

        Args:
            milvus_embedding_cfg (MilvusConfig): _description_
            milvus_media_info_cfg (MilvusConfig): _description_
        """
        self._milvus_embedding_cfg: MilvusConfig = milvus_embedding_cfg

        self._milvus_embedding_client: MilvusClient = MilvusClient(
            uri=f"http://{self._milvus_embedding_cfg.host}:{self._milvus_embedding_cfg.port}",
            usr=self._milvus_embedding_cfg.usr,
            password=self._milvus_embedding_cfg.password,
            db_name=self._milvus_embedding_cfg.db_name,
            token=self._milvus_embedding_cfg.token,
            timeout=self._milvus_embedding_cfg.timeout,
        )

        self._milvus_media_info_cfg: MilvusConfig = milvus_media_info_cfg
        self._milvus_media_info_client: MilvusClient = MilvusClient(
            uri=f"http://{self._milvus_media_info_cfg.host}:{self._milvus_media_info_cfg.port}",
            usr=self._milvus_media_info_cfg.usr,
            password=self._milvus_media_info_cfg.password,
            db_name=self._milvus_media_info_cfg.db_name,
            token=self._milvus_media_info_cfg.token,
            timeout=self._milvus_media_info_cfg.timeout,
        )

        self.__check_collection(
            self._milvus_embedding_client, self._milvus_embedding_cfg
        )

        self.__check_collection(
            self._milvus_media_info_client, self._milvus_media_info_cfg
        )

        # milvus can not search more than 16 * 1024 data, so we limit it to 10 * 1024
        self.__embedding_len_search_limit = 10 * 1024

    def __check_collection(
        self,
        client: MilvusClient,
        cfg: MilvusConfig,
    ) -> None:
        if not client.has_collection(collection_name=cfg.collection_name):
            raise MilvusException(f"not have collection: {cfg.collection_name}")

        if len(cfg.index_name) > 0:
            index_info = client.describe_index(
                collection_name=cfg.collection_name,
                index_name=cfg.index_name,
            )
            if not index_info:
                raise MilvusException(
                    f"collection: {cfg.collection_name} not have index: {cfg.index_name}"
                )

    def __add_media_info(self, data: Dict[Any, Any]) -> None:
        self._milvus_media_info_client.insert(
            collection_name=self._milvus_media_info_cfg.collection_name,
            data=data,
        )

    def __query_media_info(
        self, filter: str, output_fields: Optional[List[str]]
    ) -> List[dict[Any, Any]]:
        res: List[dict[Any, Any]] = self._milvus_media_info_client.query(
            collection_name=self._milvus_media_info_cfg.collection_name,
            filter=filter,
            output_fields=output_fields,
        )
        return res

    def __delete_media_info(self, ids: List[dict[Any, Any]]) -> None:
        self._milvus_media_info_client.delete(
            collection_name=self._milvus_media_info_cfg.collection_name,
            ids=[item["id"] for item in ids],
        )

    def __add_embeddings(self, data: Dict[Any, Any]):
        self._milvus_embedding_client.insert(
            collection_name=self._milvus_embedding_cfg.collection_name,
            data=data,
        )

    def __query_embedding(self, filter: str, output_fields: Optional[List[str]]):
        res: List[dict[Any, Any]] = self._milvus_embedding_client.query(
            collection_name=self._milvus_embedding_cfg.collection_name,
            filter=filter,
            output_fields=output_fields,
        )
        return res

    def __delete_embeddings(self, ids: List[dict[Any, Any]]) -> None:
        self._milvus_embedding_client.delete(
            collection_name=self._milvus_embedding_cfg.collection_name,
            ids=[item["id"] for item in ids],
        )

    @exception_handler
    def add_ref_embedding(self, ref_name: str, ref_embeddings: ndarray) -> None:
        """add_ref_embedding _summary_

        Args:
            ref_name (str): _description_
            ref_embeddings (ndarray): _description_
        """
        ref_uuid = str(uuid.uuid4())
        insert_media_data = {
            "uuid": ref_uuid,
            "filename": ref_name,
            "frame_total_len": len(ref_embeddings),
            "vector_no_use": _vector_no_use,
        }
        self.__add_media_info(insert_media_data)

        # add embedding
        for i, feat in enumerate(ref_embeddings):
            insert_embeddings = {
                "frame": i,
                "uuid": ref_uuid,
                "embedding": feat,
            }
            self.__add_embeddings(insert_embeddings)

    @exception_handler
    def del_ref_embedding(self, ref_uuid: str) -> None:
        """del_ref_embedding _summary_

        Args:
            ref_uuid (str): _description_

        Raises:
            MilvusException: _description_
        """
        if len(ref_uuid) == 0:
            raise MilvusException("empty ref_uuid")

        media_ids: List[dict[Any, Any]] = self.__query_media_info(
            filter=f"uuid in ['{ref_uuid}',]", output_fields=["id"]
        )
        if len(media_ids) > 0:
            self.__delete_media_info(ids=media_ids)

        embedding_ids = self.__query_embedding(
            filter=f"uuid in ['{ref_uuid}',]", output_fields=["id"]
        )
        if len(embedding_ids) > 0:
            self.__delete_embeddings(ids=embedding_ids)

    @exception_handler
    def get_matched_embeddings(
        self, sample_embeddings: Union[List[list], list], limit: int = 10
    ) -> list[list[EmbeddingsInfo]]:
        """get_matched_embeddings _summary_

        Args:
            sample_embeddings (Union[List[list], list]): _description_
            limit (int, optional): _description_. Defaults to 10.

        Returns:
            list[list[EmbeddingsInfo]]: _description_
        """
        results: List[List[dict[Any, Any]]] = list()
        if len(sample_embeddings) > self.__embedding_len_search_limit:
            for i in range(
                0, len(sample_embeddings), self.__embedding_len_search_limit
            ):
                tmp_results = self._milvus_embedding_client.search(
                    collection_name=self._milvus_embedding_cfg.collection_name,
                    data=sample_embeddings[i : i + self.__embedding_len_search_limit],
                    output_fields=["frame", "uuid"],
                    search_params={"metric_type": "L2"},
                    limit=limit,
                )
                results.extend(tmp_results)
        else:
            tmp_results = self._milvus_embedding_client.search(
                collection_name=self._milvus_embedding_cfg.collection_name,
                data=sample_embeddings,
                output_fields=["frame", "uuid"],
                search_params={"metric_type": "L2"},
                limit=limit,
            )
            results.extend(tmp_results)

        emnbed_info_list: list[list[EmbeddingsInfo]] = list()
        for i in range(len(results)):
            tmp_emnbed_info_list: list[EmbeddingsInfo] = list()
            for j in range(len(results[i])):
                cur_res: dict[Any, Any] = results[i][j]
                tmp_emnbed_info_list.append(
                    EmbeddingsInfo(
                        frame=cur_res["entity"]["frame"],
                        uuid=cur_res["entity"]["uuid"],
                        distance=cur_res["distance"],
                    )
                )
            emnbed_info_list.append(tmp_emnbed_info_list)
        return emnbed_info_list

    @exception_handler
    def get_media_info(self, ref_uuid: str) -> MediaInfo:
        """get_media_info _summary_

        Args:
            ref_uuid (str): _description_

        Raises:
            MilvusException: _description_
            MilvusException: _description_

        Returns:
            MediaInfo: _description_
        """
        if len(ref_uuid) == 0:
            raise MilvusException("empty ref_uuid")
        ref_info: List[dict[Any, Any]] = self.__query_media_info(
            filter=f"uuid in ['{ref_uuid}',]",
            output_fields=["filename", "frame_total_len", "uuid"],
        )

        if len(ref_info) > 1:
            raise MilvusException(
                f"There is a mapping relationship between {ref_uuid} and multiple media info "
            )
        media_info: MediaInfo = MediaInfo(**ref_info[0])
        return media_info


_milvus_manager: Optional[MilvusClientManager] = None


def init_milvus_client_manager(
    milvus_embedding_cfg: MilvusConfig, milvus_media_info_cfg: MilvusConfig
) -> None:
    """init_milvus_client_manager _summary_

    Args:
        milvus_embedding_cfg (MilvusConfig): _description_
        milvus_media_info_cfg (MilvusConfig): _description_
    """
    global _milvus_manager
    _milvus_manager = MilvusClientManager(milvus_embedding_cfg, milvus_media_info_cfg)


def get_milvus_client_manager() -> MilvusClientManager:
    """get_milvus_client_manager _summary_

    Raises:
        MilvusException: _description_

    Returns:
        MilvusClientManager: _description_
    """
    global _milvus_manager
    if _milvus_manager == None:
        raise MilvusException("milvus manager need init befor use")
    return _milvus_manager
