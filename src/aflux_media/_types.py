from fractions import Fraction

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, model_validator


class AudioStreamInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_rate: PositiveInt
    time_base: Fraction
    num_channels: PositiveInt
    channel_layout: str
    codec: str
    sample_format: str
    num_samples: PositiveInt


class AudioBlock(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    from_sample_index: NonNegativeInt
    to_sample_index: NonNegativeInt
    from_timestamp: Fraction
    to_timestamp: Fraction
    num_samples: PositiveInt
    block_size: PositiveInt
    samples: npt.NDArray[np.float32]

    @model_validator(mode="after")
    def _validate_block(self) -> "AudioBlock":
        if self.to_sample_index <= self.from_sample_index:
            msg = "Audio block should have a non-empty sample range."
            raise ValueError(msg)
        if self.num_samples != self.to_sample_index - self.from_sample_index:
            msg = "Audio block sample count should agree with its sample range."
            raise ValueError(msg)
        if self.num_samples > self.block_size:
            msg = "Audio block sample count should not exceed its configured block size."
            raise ValueError(msg)
        if self.samples.ndim != 2:
            msg = "Audio block samples should have shape `(num_samples, num_channels)`."
            raise ValueError(msg)
        if self.samples.shape[0] != self.num_samples:
            msg = "Audio block array length should agree with its sample count."
            raise ValueError(msg)
        if self.samples.dtype != np.float32:
            msg = "Audio block samples should have dtype `float32`."
            raise ValueError(msg)
        if self.samples.flags.writeable:
            msg = "Audio block samples should be read-only."
            raise ValueError(msg)
        if self.to_timestamp <= self.from_timestamp:
            msg = "Audio block should have an increasing timestamp range."
            raise ValueError(msg)
        return self


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
