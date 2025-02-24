from pydantic import BaseModel, Field


class EmbeddingsInfo(BaseModel):
    frame: int = Field(..., title="frame sequence number")
    uuid: str = Field(..., title="media uuid")
    distance: float = Field(..., title="farme distance")


class MediaInfo(BaseModel):
    filename: str = Field(..., title="media file name")
    frame_total_len: int = Field(..., title="media frame length")
    uuid: str = Field(..., title="media uuid")
    id: int = Field(..., title="milvus auto id")
