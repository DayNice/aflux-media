import bisect
import functools
import operator
from collections.abc import Iterable, Iterator
from fractions import Fraction
from pathlib import Path
from types import TracebackType
from typing import Self, cast

import av
import av.container
import av.error
import av.stream
import av.video.reformatter
import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray

from ._stats_utils import get_sample_indices
from ._types import VideoFrameInfo, VideoStatistics, VideoStreamInfo


class VideoReader:
    def __init__(self, video_file: str | Path) -> None:
        self.file = Path(video_file)
        self._container = av.open(self.file)
        if len(self._container.streams.video) == 0:
            self._container.close()
            msg = f"File should contain at least one video stream: {video_file}"
            raise ValueError(msg)
        self._stream = self._container.streams.video[0]
        self._stream.thread_type = "AUTO"

        # preload attributes to prevent implicit seek while demuxing or decoding
        self._first_keyframe_pts
        self._last_keyframe_pts
        self._stream_info

    def _seek_pts(self, pts: int, *, backward: bool = True) -> bool:
        try:
            self._container.seek(pts, backward=backward, stream=self._stream)
            return True
        except av.error.PermissionError as e:
            if e.args != (1, "Operation not permitted"):
                raise
            # FFmpeg exception for trying to seek beyond last keyframe
            return False

    def _demux_packets(self) -> Iterator[av.Packet]:
        return self._container.demux(self._stream)

    def _decode_frames(self) -> Iterator[av.VideoFrame]:
        return self._container.decode(self._stream)

    @functools.cached_property
    def _first_keyframe_pts(self) -> int:
        self._seek_pts(0)
        packet = next(self._demux_packets())
        assert isinstance(packet, av.Packet)
        assert packet.is_keyframe, "Packet should belong to a keyframe."
        assert packet.pts is not None, "Keyframe should have a pts."
        return packet.pts

    @functools.cached_property
    def _last_keyframe_pts(self) -> int:
        # compute initial search boundary
        high_pts = 1
        while self._seek_pts(high_pts, backward=False):
            high_pts = high_pts * 2
        low_pts = high_pts // 2

        # binary search for last keyframe
        while (high_pts - low_pts) > 1:
            mid_pts = low_pts + (high_pts - low_pts) // 2
            if self._seek_pts(mid_pts, backward=False):
                low_pts = mid_pts
            else:
                high_pts = mid_pts
        return low_pts

    @functools.cached_property
    def _frames_per_time_base(self) -> Fraction:
        assert self._stream.time_base is not None
        assert self._stream.average_rate is not None
        return self._stream.time_base * self._stream.average_rate

    def _get_num_frames(self) -> int:
        # we assume the first frame is a keyframe
        min_pts = self._first_keyframe_pts
        max_pts = 0
        assert self._seek_pts(self._last_keyframe_pts)
        for packet in self._demux_packets():
            if packet.pts is None:
                continue
            max_pts = max(max_pts, packet.pts)
        return round((max_pts - min_pts) * self._frames_per_time_base) + 1

    @functools.cached_property
    def _stream_info(self) -> VideoStreamInfo:
        # stream attributes available in read mode
        assert self._stream.average_rate is not None
        assert self._stream.time_base is not None
        assert self._stream.pix_fmt is not None

        num_channels = len(self._stream.format.components)
        codec = self._stream.codec.canonical_name

        num_frames = self._stream.frames
        if num_frames == 0:
            num_frames = self._get_num_frames()

        video_stream_info = VideoStreamInfo(
            fps=self._stream.average_rate,
            time_base=self._stream.time_base,
            height=self._stream.height,
            width=self._stream.width,
            num_channels=num_channels,
            codec=codec,
            pixel_format=self._stream.pix_fmt,
            num_frames=num_frames,
        )
        return video_stream_info

    def _demux_frame_infos(self) -> Iterator[VideoFrameInfo]:
        for packet in self._demux_packets():
            if packet.pts is None:
                continue
            frame_index = self._estimate_frame_index_by_pts(packet.pts)
            frame_info = VideoFrameInfo(
                frame_index=frame_index,
                timestamp=packet.pts * self._stream_info.time_base,
                dts=packet.dts if packet.dts is not None else packet.pts,
                pts=packet.pts,
                is_keyframe=packet.is_keyframe,
            )
            yield frame_info

    @functools.cached_property
    def _keyframe_infos(self) -> list[VideoFrameInfo]:
        keyframe_infos: list[VideoFrameInfo] = []
        prev_keyframe_pts: int = -1

        while self._seek_pts(prev_keyframe_pts + 1, backward=False):
            keyframe_info = next(self._demux_frame_infos())
            assert isinstance(keyframe_info, VideoFrameInfo)
            assert keyframe_info.is_keyframe, "Packet should belong to a keyframe."

            keyframe_infos.append(keyframe_info)
            prev_keyframe_pts = keyframe_info.pts

        return keyframe_infos

    def _search_keyframe_info_by_pts(self, pts: int) -> VideoFrameInfo:
        if pts < 0:
            msg = "Target pts should be a non-negative integer."
            raise ValueError(msg)
        if pts <= self._keyframe_infos[0].pts:
            return self._keyframe_infos[0]

        pos = bisect.bisect_right(self._keyframe_infos, pts, key=operator.attrgetter("pts"))
        return self._keyframe_infos[pos - 1]

    def _estimate_frame_pts_by_index(self, frame_index: int) -> int:
        if frame_index < 0:
            msg = f"Frame index should be non-negative: {frame_index}"
            raise ValueError(msg)
        if frame_index >= self._stream_info.num_frames:
            msg = f"Frame index should be less than size: {frame_index}"
            raise ValueError(msg)

        # we assume the first frame is a keyframe
        first_frame_pts = self._first_keyframe_pts
        return first_frame_pts + round(frame_index / self._frames_per_time_base)

    def _estimate_frame_index_by_pts(self, pts: int) -> int:
        if pts < 0:
            msg = "Target pts should be a non-negative integer."
            raise ValueError(msg)

        # we assume the first frame is a keyframe
        first_frame_pts = self._first_keyframe_pts
        frame_index = round((pts - first_frame_pts) * self._frames_per_time_base)

        if frame_index < 0:
            msg = f"Estimated frame index was negative: {frame_index}"
            raise ValueError(msg)
        if frame_index >= self._stream_info.num_frames:
            msg = f"Estimated frame index was greater than or equal to size: {frame_index}"
            raise ValueError(msg)

        return frame_index

    def _build_keyframe_map(
        self,
        frame_indices: Iterable[int],
    ) -> dict[VideoFrameInfo, list[VideoFrameInfo]]:
        frame_indices = sorted(frame_indices)
        if len(frame_indices) != len(set(frame_indices)):
            msg = "Frame indices should be unique."
            raise ValueError(msg)

        if len(frame_indices) == 0:
            return {}

        keyframe_frame_indices_map: dict[VideoFrameInfo, list[int]] = {}
        for frame_index in frame_indices:
            target_frame_pts = self._estimate_frame_pts_by_index(frame_index)
            keyframe_info = self._search_keyframe_info_by_pts(target_frame_pts)
            if keyframe_info not in keyframe_frame_indices_map:
                keyframe_frame_indices_map[keyframe_info] = []
            keyframe_frame_indices_map[keyframe_info].append(frame_index)

        # assume at least 1 keyframe per 12 seconds
        pts_guard = 12 * self._stream_info.fps / self._frames_per_time_base

        keyframe_map: dict[VideoFrameInfo, list[VideoFrameInfo]] = {}
        for keyframe_info, keyframe_frame_indices in keyframe_frame_indices_map.items():
            keyframe_frame_infos: list[VideoFrameInfo] = []
            target_frame_index_set = set(keyframe_frame_indices)

            assert self._seek_pts(keyframe_info.pts)
            for frame_info in self._demux_frame_infos():
                if abs(frame_info.pts - keyframe_info.pts) > pts_guard:
                    raise RuntimeError("Frame pts value was outside expected demuxing range.")
                if frame_info.frame_index not in target_frame_index_set:
                    continue

                keyframe_frame_infos.append(frame_info)
                target_frame_index_set.remove(frame_info.frame_index)

                if len(target_frame_index_set) == 0:
                    break

            if len(target_frame_index_set) > 0:
                msg = f"Frames were not within expected demuxing range: {target_frame_index_set}"
                raise RuntimeError(msg)
            keyframe_frame_infos.sort(key=operator.attrgetter("frame_index"))
            keyframe_map[keyframe_info] = keyframe_frame_infos

        return keyframe_map

    def get_stream_info(self) -> VideoStreamInfo:
        return self._stream_info

    def get_frame_infos(self, frame_indices: Iterable[int] | None = None) -> list[VideoFrameInfo]:
        if frame_indices is None:
            assert self._seek_pts(0)
            frame_infos = list(self._demux_frame_infos())
            frame_infos.sort(key=operator.attrgetter("frame_index"))
            return frame_infos

        frame_indices = list(frame_indices)
        keyframe_map = self._build_keyframe_map(frame_indices)

        frame_info_map: dict[int, VideoFrameInfo] = {}
        for frame_infos in keyframe_map.values():
            for frame_info in frame_infos:
                frame_info_map[frame_info.frame_index] = frame_info

        return [frame_info_map[el] for el in frame_indices]

    def get_frame_info(self, frame_index: int) -> VideoFrameInfo:
        estimated_frame_pts = self._estimate_frame_pts_by_index(frame_index)

        pts_tolerance = Fraction(1, 2) / self._frames_per_time_base
        # assume at least 1 keyframe per 12 seconds
        pts_guard = 12 * self._stream_info.fps / self._frames_per_time_base

        assert self._seek_pts(estimated_frame_pts)
        for frame_info in self._demux_frame_infos():
            if frame_info.pts - estimated_frame_pts > pts_guard:
                raise RuntimeError("Frame pts value was outside expected demuxing range.")
            if abs(frame_info.pts - estimated_frame_pts) <= pts_tolerance:
                return frame_info

        raise RuntimeError("Frame was not within expected demuxing range")

    def get_keyframe_infos(self) -> list[VideoFrameInfo]:
        return self._keyframe_infos

    def get_prev_keyframe_info(self, frame_index: int) -> VideoFrameInfo:
        frame_pts = self._estimate_frame_pts_by_index(frame_index)
        assert self._seek_pts(frame_pts), "The first frame of a video should be a keyframe."
        keyframe_info = next(self._demux_frame_infos())

        assert keyframe_info.is_keyframe, "Packet should belong to a keyframe."
        return keyframe_info

    def get_next_keyframe_info(self, frame_index: int) -> VideoFrameInfo | None:
        frame_pts = self._estimate_frame_pts_by_index(frame_index)
        if not self._seek_pts(frame_pts, backward=False):
            return None
        keyframe_info = next(self._demux_frame_infos())

        assert isinstance(keyframe_info, VideoFrameInfo)
        assert keyframe_info.is_keyframe, "Packet should belong to a keyframe."
        return keyframe_info

    def decode_frames(self, frame_indices: Iterable[int] | None = None) -> Iterator[av.VideoFrame]:
        if frame_indices is None:
            assert self._seek_pts(0)
            yield from self._decode_frames()
            return

        frame_indices = list(frame_indices)
        if len(frame_indices) == 0:
            return

        keyframe_map = self._build_keyframe_map(frame_indices)
        keyframe_reverse_map: dict[int, VideoFrameInfo] = {}
        for keyframe_info, frame_infos in keyframe_map.items():
            for frame_info in frame_infos:
                keyframe_reverse_map[frame_info.frame_index] = keyframe_info

        found_frame_map: dict[int, av.VideoFrame] = {}
        for frame_index in frame_indices:
            if frame_index in found_frame_map:
                yield found_frame_map.pop(frame_index)
                continue

            keyframe_info = keyframe_reverse_map[frame_index]
            frame_infos = keyframe_map[keyframe_info]

            frame_pts_map = {el.pts: el.frame_index for el in frame_infos}
            assert len(frame_pts_map) == len(frame_infos), "All pts values within video should be unique."
            max_frame_pts = max(frame_pts_map.keys())

            assert self._seek_pts(keyframe_info.pts)
            for frame in self._decode_frames():
                assert frame.pts is not None
                assert frame.pts <= max_frame_pts, "Target pts should exist in video."

                if frame.pts in frame_pts_map:
                    decoded_index = frame_pts_map.pop(frame.pts)
                    found_frame_map[decoded_index] = frame
                if len(frame_pts_map) == 0:
                    break
            assert len(frame_pts_map) == 0, "Target pts should exist in video."

            yield found_frame_map.pop(frame_index)

    def decode_frame(self, frame_index: int) -> av.VideoFrame:
        frame_info = self.get_frame_info(frame_index)

        assert self._seek_pts(frame_info.pts)
        for frame in self._decode_frames():
            assert frame.pts is not None, "Frame should have a valid pts value."
            if frame.pts > frame_info.pts:
                raise RuntimeError("Frame pts value was outside expected decoding range.")
            if frame.pts == frame_info.pts:
                return frame

        raise RuntimeError("Frame not found within expected decoding range.")

    def compute_statistics(self) -> VideoStatistics:
        # The upper bound of sample size is 385.
        # For a 3840x2160 video, the upper bound of num pixels is 3,193,344,000 (< 2^32).
        # Thus, `x_sum` (< 2^40) and `x_squared_sum` (< 2^48) can fit within a 64-bit integer.
        sample_indices = get_sample_indices(self._stream_info.num_frames)
        num_pixels = len(sample_indices) * self._stream_info.height * self._stream_info.width

        x_min_list: list[list[np.int64]] = []
        x_max_list: list[list[np.int64]] = []
        x_sum_list: list[list[np.int64]] = []
        x_square_sum_list: list[list[np.int64]] = []

        for frame in self.decode_frames(sample_indices):
            arr = frame.to_ndarray(format="rgb24").astype(np.int64)
            channel_indices = range(arr.shape[-1])

            # for some reason this is much faster than arr.min(axis=(0, 1))
            x_min = [arr[..., i].min() for i in channel_indices]
            x_max = [arr[..., i].max() for i in channel_indices]
            x_sum = [arr[..., i].sum() for i in channel_indices]
            x_square_sum = [(arr[..., i] ** 2).sum() for i in channel_indices]

            x_min_list.append(x_min)
            x_max_list.append(x_max)
            x_sum_list.append(x_sum)
            x_square_sum_list.append(x_square_sum)

        x_min = np.min(x_min_list, axis=0)
        x_max = np.max(x_max_list, axis=0)
        x_sum = np.sum(x_sum_list, axis=0, dtype=np.int64)
        x_square_sum = np.sum(x_square_sum_list, axis=0, dtype=np.int64)

        x_mean = x_sum / num_pixels
        x_std = np.sqrt(x_square_sum / num_pixels - x_mean**2)

        x_min = cast(NDArray[np.float64], x_min / 255)
        x_max = cast(NDArray[np.float64], x_max / 255)
        x_mean = cast(NDArray[np.float64], x_mean / 255)
        x_std = cast(NDArray[np.float64], x_std / 255)

        statistics = VideoStatistics(
            sample_size=len(sample_indices),
            min=tuple(x_min.tolist()),
            max=tuple(x_max.tolist()),
            mean=tuple(x_mean.tolist()),
            std=tuple(x_std.tolist()),
        )
        return statistics

    @staticmethod
    def convert_frames_into_rgb_numpy(frames: Iterable[av.VideoFrame]) -> npt.NDArray[np.uint8]:
        """Convert video frames into RGB numpy.

        Args:
            frames: An iterable of video frames.

        Returns:
            An RGB array of dtype `uint8` and shape `(N, H, W, 3)`.
        """

        reformatter = av.video.reformatter.VideoReformatter()

        arr_list: list[npt.NDArray[np.uint8]] = []
        for frame in frames:
            frame = reformatter.reformat(frame, format="rgb24")
            arr = frame.to_ndarray()

            assert arr.dtype == np.uint8, "RGB array should be type `uint8`."
            assert len(arr.shape) == 3 and arr.shape[2] == 3, "RGB array should be of shape `(H, W, 3)`."
            arr = cast(npt.NDArray[np.uint8], arr)

            arr_list.append(arr)
        return np.stack(arr_list)

    def close(self) -> None:
        self._container.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


def get_video_stream_info(video_file: str | Path) -> VideoStreamInfo:
    with VideoReader(video_file) as video_reader:
        return video_reader.get_stream_info()


def get_video_frame_infos(
    video_file: str | Path,
    frame_indices: Iterable[int] | None = None,
) -> list[VideoFrameInfo]:
    with VideoReader(video_file) as video_reader:
        return video_reader.get_frame_infos(frame_indices)


def get_video_keyframe_infos(video_file: str | Path) -> list[VideoFrameInfo]:
    with VideoReader(video_file) as video_reader:
        return video_reader.get_keyframe_infos()


def decode_video_frames(
    video_file: str | Path,
    frame_indices: Iterable[int] | None = None,
) -> Iterator[av.VideoFrame]:
    with VideoReader(video_file) as video_reader:
        yield from video_reader.decode_frames(frame_indices)


def compute_video_statistics(video_file: str | Path) -> VideoStatistics:
    with VideoReader(video_file) as video_reader:
        return video_reader.compute_statistics()
