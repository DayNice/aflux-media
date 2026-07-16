from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import numpy as np
import numpy.typing as npt

from ._types import AudioBlock, AudioStreamInfo


class AudioReader:
    def __init__(self, audio_file: str | Path, *, block_size: int = 4096) -> None:
        if block_size <= 0:
            msg = f"Block size should be positive: {block_size}"
            raise ValueError(msg)

        self.file = Path(audio_file)
        self.block_size = block_size

        with av.open(self.file) as container:
            if len(container.streams.audio) == 0:
                msg = f"File should contain at least one audio stream: {audio_file}"
                raise ValueError(msg)
            stream = container.streams.audio[0]
            self._stream_info = self._read_stream_info(container, stream)

    @staticmethod
    def _read_stream_info(
        container: av.container.InputContainer,
        stream: av.AudioStream,
    ) -> AudioStreamInfo:
        assert stream.time_base is not None

        if stream.duration is not None and stream.duration > 0:
            num_samples = round(stream.duration * stream.time_base * stream.rate)
        else:
            num_samples = 0
            for frame in container.decode(stream):
                assert frame.pts is not None
                frame_from_sample_index = round(frame.pts * stream.time_base * stream.rate)
                frame_num_samples = frame.samples
                if frame.duration is not None and frame.duration > 0:
                    frame_duration_num_samples = round(frame.duration * stream.time_base * stream.rate)
                    frame_num_samples = min(frame_num_samples, frame_duration_num_samples)
                frame_to_sample_index = frame_from_sample_index + frame_num_samples
                num_samples = max(num_samples, frame_to_sample_index)

        return AudioStreamInfo(
            sample_rate=stream.rate,
            time_base=stream.time_base,
            num_channels=stream.layout.nb_channels,
            channel_layout=stream.layout.name,
            codec=stream.codec.canonical_name,
            sample_format=stream.format.name,
            num_samples=num_samples,
        )

    def _decode_frames(self) -> Iterator[av.AudioFrame]:
        with av.open(self.file) as container:
            stream = container.streams.audio[0]
            yield from container.decode(stream)

    def _estimate_sample_index_by_pts(self, pts: int) -> int:
        stream_info = self._stream_info
        return round(pts * stream_info.time_base * stream_info.sample_rate)

    def _normalize_frame_array(self, frame: av.AudioFrame) -> npt.NDArray[np.float32]:
        num_samples = frame.samples
        if frame.duration is not None and frame.duration > 0:
            duration_num_samples = round(frame.duration * self._stream_info.time_base * self._stream_info.sample_rate)
            num_samples = min(num_samples, duration_num_samples)

        array = frame.to_ndarray()
        if frame.format.is_planar:
            array = array.T
        else:
            array = array.reshape(frame.samples, self._stream_info.num_channels)
        array = array[:num_samples]

        return cast(npt.NDArray[np.float32], np.ascontiguousarray(array, dtype=np.float32))

    def get_stream_info(self) -> AudioStreamInfo:
        return self._stream_info

    def get_sample_index(self, timestamp: Fraction) -> int:
        if timestamp < 0:
            msg = f"Timestamp should be non-negative: {timestamp}"
            raise ValueError(msg)
        return round(timestamp * self._stream_info.sample_rate)

    def get_timestamp(self, sample_index: int) -> Fraction:
        if sample_index < 0:
            msg = f"Sample index should be non-negative: {sample_index}"
            raise ValueError(msg)
        return Fraction(sample_index, self._stream_info.sample_rate)

    def decode_blocks(
        self,
        from_sample_index: int = 0,
        to_sample_index: int | None = None,
    ) -> Iterator[AudioBlock]:
        if to_sample_index is None:
            to_sample_index = self._stream_info.num_samples

        if from_sample_index < 0:
            msg = f"From sample index should be non-negative: {from_sample_index}"
            raise ValueError(msg)
        if to_sample_index < 0:
            msg = f"To sample index should be non-negative: {to_sample_index}"
            raise ValueError(msg)

        if from_sample_index >= to_sample_index:
            return

        frame_iterator = self._decode_frames()
        frame = next(frame_iterator, None)
        frame_array = None

        for block_from_sample_index in range(from_sample_index, to_sample_index, self.block_size):
            block_to_sample_index = min(block_from_sample_index + self.block_size, to_sample_index)
            num_samples = block_to_sample_index - block_from_sample_index

            samples = np.zeros((num_samples, self._stream_info.num_channels), dtype=np.float32)
            while frame is not None:
                assert frame.pts is not None

                frame_from_sample_index = self._estimate_sample_index_by_pts(frame.pts)
                if frame_from_sample_index >= block_to_sample_index:
                    break

                frame_num_samples = frame.samples
                if frame.duration is not None and frame.duration > 0:
                    frame_duration_num_samples = round(
                        frame.duration * self._stream_info.time_base * self._stream_info.sample_rate
                    )
                    frame_num_samples = min(frame_num_samples, frame_duration_num_samples)

                frame_to_sample_index = frame_from_sample_index + frame_num_samples
                if frame_to_sample_index <= block_from_sample_index:
                    frame = next(frame_iterator, None)
                    frame_array = None
                    continue

                if frame_array is None:
                    frame_array = self._normalize_frame_array(frame)

                copy_from_sample_index = max(block_from_sample_index, frame_from_sample_index)
                copy_to_sample_index = min(block_to_sample_index, frame_to_sample_index)
                samples[
                    copy_from_sample_index - block_from_sample_index : copy_to_sample_index - block_from_sample_index
                ] = frame_array[
                    copy_from_sample_index - frame_from_sample_index : copy_to_sample_index - frame_from_sample_index
                ]

                if frame_to_sample_index > block_to_sample_index:
                    break

                frame = next(frame_iterator, None)
                frame_array = None
            samples.setflags(write=False)

            block = AudioBlock(
                from_sample_index=block_from_sample_index,
                to_sample_index=block_to_sample_index,
                from_timestamp=Fraction(block_from_sample_index, self._stream_info.sample_rate),
                to_timestamp=Fraction(block_to_sample_index, self._stream_info.sample_rate),
                num_samples=num_samples,
                block_size=self.block_size,
                samples=samples,
            )
            yield block
