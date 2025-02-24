from pydantic import BaseModel, Field


class MediaDes(BaseModel):
    name: str = Field(..., title="media name")
    uuid: str = Field(..., title="media uuid")
    frame_total_len: int = Field(..., title="media frame length")
    avg_frame_time: float = Field(default=125.0)


class QueryResult(BaseModel):
    ref_start_index: int = Field(default=-1, title="frame sequence number")
    ref_end_index: int = Field(default=-1, title="frame sequence number")
    sample_start_index: int = Field(default=-1, title="frame sequence number")
    sample_end_index: int = Field(default=-1, title="frame sequence number")
    remove_flag: bool = Field(default=False, title="remove flag")
    distance: float = Field(default=0, title="distance")
