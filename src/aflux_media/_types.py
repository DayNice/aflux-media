from fractions import Fraction

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt


class VideoStreamInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    fps: Fraction
    time_base: Fraction
    height: PositiveInt
    width: PositiveInt
    num_channels: PositiveInt
    codec: str
    pixel_format: str
    num_frames: PositiveInt


class VideoFrameInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    frame_index: int
    timestamp: Fraction
    dts: int
    pts: NonNegativeInt
    is_keyframe: bool


class VideoStatistics(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_size: PositiveInt
    min: tuple[float, float, float]
    max: tuple[float, float, float]
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
