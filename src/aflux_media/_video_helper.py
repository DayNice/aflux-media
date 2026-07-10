import contextlib
import itertools
import math
import tempfile
from collections.abc import Iterable
from fractions import Fraction
from pathlib import Path
from typing import cast

import av
import av.container
import av.stream
import av.video.reformatter
import PIL.Image

from ._types import VideoStatistics
from ._video_reader import VideoReader, get_video_stream_info


def merge_video_statistics_list(video_statistics_list: Iterable[VideoStatistics]) -> VideoStatistics:
    video_statistics_list = list(video_statistics_list)
    if len(video_statistics_list) == 0:
        msg = "Provide at least one video statistics."
        raise ValueError(msg)
    first_video_statistics = video_statistics_list[0]
    if len(video_statistics_list) == 1:
        return first_video_statistics

    total_sample_size = sum(el.sample_size for el in video_statistics_list)
    total_min = []
    for i in range(len(first_video_statistics.min)):
        min_scalar = min(el.min[i] for el in video_statistics_list)
        total_min.append(min_scalar)
    total_min = cast(tuple[float, float, float], tuple(total_min))

    total_max = []
    for i in range(len(first_video_statistics.max)):
        max_scalar = max(el.max[i] for el in video_statistics_list)
        total_max.append(max_scalar)
    total_max = cast(tuple[float, float, float], tuple(total_max))

    total_mean = []
    for i in range(len(first_video_statistics.mean)):
        mean_scalar = sum(el.sample_size * el.mean[i] for el in video_statistics_list) / total_sample_size
        total_mean.append(mean_scalar)
    total_mean = cast(tuple[float, float, float], tuple(total_mean))

    total_std = []
    for i in range(len(first_video_statistics.std)):
        square_scalar = 0
        for el in video_statistics_list:
            square_scalar += (el.sample_size - 1) * (el.std[i] ** 2)
            square_scalar += el.sample_size * (el.mean[i] ** 2)
        square_scalar /= total_sample_size

        var_scalar = (square_scalar - total_mean[i] ** 2) * (total_sample_size / (total_sample_size - 1))
        std_scalar = math.sqrt(max(0, var_scalar))
        total_std.append(std_scalar)
    total_std = cast(tuple[float, float, float], tuple(total_std))

    return VideoStatistics(
        sample_size=total_sample_size,
        min=total_min,
        max=total_max,
        mean=total_mean,
        std=total_std,
    )


def infer_target_bits_per_pixel(
    num_pixels: int,
    fps: int | Fraction,
    *,
    complexity_constant: float = 16.908,
    spatial_scaling_factor: float = 0.687,
    temporal_scaling_factor: float = 0.560,
) -> float:
    """Infer target bits per pixel for a given (num_pixels, fps) combination.

    Assumes the following relation:

        bit_rate = complexity_constant
            * (num_pixels ^ spatial_scaling_factor)
            * (fps ^ temporal_scaling_factor)

        bit_rate = bits_per_pixel * num_pixels * fps

    Default parameters were set using `scipy.optimize.curve_fit` against the following values.
        - 480p 30fps: 1/15
        - 720p 30fps: 1/20
        - 1080p 30fps: 1/25
        - 1080p 60fps: 1/35
        - 4k 60fps: 1/50
    """
    if num_pixels <= 0:
        raise ValueError("Number of pixels should be a positive value.")
    if fps <= 0:
        raise ValueError("Frame rate should a positive value.")

    bits_per_pixel = complexity_constant
    bits_per_pixel *= num_pixels ** (spatial_scaling_factor - 1.0)
    bits_per_pixel *= float(fps) ** (temporal_scaling_factor - 1.0)
    return bits_per_pixel


def remux_video_into_mp4(
    input_file: str | Path,
    output_file: str | Path,
) -> None:
    with (
        av.open(input_file) as input_container,
        av.open(output_file, "w", format="mp4") as output_container,
    ):
        input_streams = [el for el in input_container.streams if isinstance(el, (av.VideoStream, av.AudioStream))]

        output_stream_map: dict[int, av.VideoStream | av.AudioStream] = {}
        for input_stream in input_streams:
            output_stream = output_container.add_stream_from_template(input_stream)
            output_stream_map[input_stream.index] = output_stream

            # prevent timestamp drift
            match input_stream:
                case av.VideoStream():
                    # 90,000Hz
                    output_stream.time_base = Fraction(1, 90000)
                case av.AudioStream():
                    output_stream.time_base = Fraction(1, input_stream.rate)
                case _:
                    assert input_stream.time_base is not None
                    output_stream.time_base = input_stream.time_base

        for packet in input_container.demux(input_streams):
            # ignore 'flush packet'
            if packet.size == 0 and packet.dts is None and packet.pts is None:
                continue

            output_stream = output_stream_map[packet.stream.index]
            packet.stream = output_stream
            output_container.mux(packet)


def encode_images_into_video(
    images: Iterable[PIL.Image.Image],
    output_file: str | Path,
    *,
    fps: int | Fraction = 30,
    max_bits_per_pixel: float | Fraction | None = None,
):
    """Encode images into av1 mp4 videos

    Note that libsvtav1 logs can only be disabled by setting the environment variable `SVT_LOG=0`
    from outside the Python process since `os.environ` has no control over the process spawned by PyAV.
    """

    output_file = Path(output_file)

    images = iter(images)
    sample_image = next(images, None)
    if sample_image is None:
        raise ValueError("Provide at least one image.")
    images = itertools.chain([sample_image], images)

    if max_bits_per_pixel is None:
        num_pixels = sample_image.width * sample_image.height
        max_bits_per_pixel = infer_target_bits_per_pixel(num_pixels, fps) * 2
    max_bits_per_sec = sample_image.width * sample_image.height * fps * max_bits_per_pixel
    output_file.parent.mkdir(parents=True, exist_ok=True)

    encoder_options = {
        "preset": "6",
        "crf": "26",
        "svtav1-params": "tune=0",
        "maxrate": f"{round(max_bits_per_sec)}",
        "bufsize": f"{round(max_bits_per_sec * 2)}",
    }

    with av.open(output_file, "w", format="mp4", options={"movflags": "faststart"}) as container:
        stream = cast(av.VideoStream, container.add_stream("libsvtav1", fps, encoder_options))
        stream.width = sample_image.width
        stream.height = sample_image.height
        stream.pix_fmt = "yuv420p10le"
        stream.gop_size = round(fps * 2)
        stream.time_base = Fraction(1, 90000)

        for image in images:
            frame = av.VideoFrame.from_image(image)
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)


def mux_copy_video_segment(
    input_file: str | Path,
    output_file: str | Path,
    from_frame_index: int,
    to_frame_index: int,
) -> None:
    """Copy a video segment by muxing packets.

    Assumes the given range corresponds to valid GOP (Group of Pictures) boundaries.
    """
    with (
        VideoReader(input_file) as input_reader,
        av.open(output_file, "w", format="mp4", options={"movflags": "faststart"}) as output_container,
    ):
        stream_info = input_reader.get_stream_info()
        from_frame_pts = input_reader.get_frame_info(from_frame_index).pts
        to_frame_pts = None
        if to_frame_index < stream_info.num_frames:
            to_frame_pts = input_reader.get_frame_info(to_frame_index).pts

        # TODO: use public attributes instead of private ones
        output_stream = output_container.add_stream_from_template(
            input_reader._stream,
            True,  # prevents 'Unknown codec' error
            rate=stream_info.fps,
            time_base=stream_info.time_base,
            width=stream_info.width,
            height=stream_info.height,
            pix_fmt=stream_info.pixel_format,
        )

        # TODO: use public methods instead of private ones
        input_reader._seek_pts(from_frame_pts)
        for packet in input_reader._demux_packets():
            # ignore 'flush packet'
            if packet.size == 0 and packet.dts is None and packet.pts is None:
                continue

            assert packet.pts is not None, "Packet should have a valid pts."
            if packet.pts < from_frame_pts:
                continue
            if to_frame_pts is not None and packet.pts >= to_frame_pts:
                break

            packet.pts -= from_frame_pts
            packet.dts = None  # re-generate dts value
            packet.stream = output_stream
            output_container.mux(packet)


def encode_copy_video_segment(
    input_file: str | Path,
    output_file: str | Path,
    from_frame_index: int,
    to_frame_index: int,
) -> None:
    """Copy a video segment by encoding frames."""
    with (
        VideoReader(input_file) as input_reader,
        av.open(output_file, "w", format="mp4", options={"movflags": "faststart"}) as output_container,
    ):
        stream_info = input_reader.get_stream_info()
        from_frame_info = input_reader.get_frame_info(from_frame_index)

        output_stream = output_container.add_stream(
            stream_info.codec,
            rate=stream_info.fps,
            time_base=stream_info.time_base,
            width=stream_info.width,
            height=stream_info.height,
            pix_fmt=stream_info.pixel_format,
        )
        output_stream = cast(av.VideoStream, output_stream)

        for frame in input_reader.decode_frames(range(from_frame_index, to_frame_index)):
            assert frame.pts is not None, "Frame should have a valid pts value."
            frame.pts -= from_frame_info.pts
            if frame.dts is not None:
                frame.dts = frame.pts
            for packet in output_stream.encode(frame):
                output_container.mux(packet)
        for packet in output_stream.encode():
            output_container.mux(packet)


def mux_concat_videos(
    input_files: Iterable[str | Path],
    output_file: str | Path,
) -> None:
    """Concatenate videos using a demuxer.

    The following attributes must be same for all input videos:
      - frame rate
      - time base
      - height
      - width
      - codec
      - pixel format
    """
    input_files = [Path(el) for el in input_files]
    if len(input_files) == 0:
        raise ValueError("Provide at least one video file.")

    stream_info = get_video_stream_info(input_files[0])
    for input_file in input_files[1:]:
        other_stream_info = get_video_stream_info(input_file)
        if (
            other_stream_info.fps != stream_info.fps
            or other_stream_info.time_base != stream_info.time_base
            or other_stream_info.height != stream_info.height
            or other_stream_info.width != stream_info.width
            or other_stream_info.codec != stream_info.codec
            or other_stream_info.pixel_format != stream_info.pixel_format
        ):
            msg = f"Found incompatible videos: {input_file} {input_files[0]}"
            raise ValueError(msg)

    with contextlib.ExitStack() as stack:
        ffconcat_file = stack.enter_context(tempfile.NamedTemporaryFile(delete_on_close=False))
        ffconcat_file = Path(ffconcat_file.name)
        ffconcat_file.write_text("".join(f"file {el.absolute().as_posix()}\n" for el in input_files))

        try:
            input_container = stack.enter_context(
                av.open(ffconcat_file, format="concat", options={"safe": "0"}),
            )
        except av.error.FileNotFoundError as e:
            # NOTE: error is misleading in that it broadly handles cases where the ffconcat content is incorrect
            # (e.g. invalid format, input files not found, incompatible files, etc)
            raise ValueError("Failed to open input files using ffconcat.") from e
        output_container = stack.enter_context(
            av.open(output_file, "w", format="mp4", options={"movflags": "faststart"}),
        )

        assert len(input_container.streams.video) > 0
        input_stream = input_container.streams.video[0]

        output_stream = output_container.add_stream_from_template(
            input_stream,
            True,
            rate=stream_info.fps,
            time_base=stream_info.time_base,
            width=stream_info.width,
            height=stream_info.height,
            pix_fmt=stream_info.pixel_format,
        )
        for packet in input_container.demux(input_stream):
            # ignore 'flush packet'
            if packet.size == 0 and packet.dts is None and packet.pts is None:
                continue
            packet.dts = None  # re-generate dts value
            packet.stream = output_stream
            output_container.mux(packet)


def encode_concat_videos(
    input_files: Iterable[str | Path],
    output_file: str | Path,
    *,
    max_bits_per_pixel: float | Fraction | None = None,
) -> None:
    """Concatenate videos using an encoder.

    The following attributes must be same for all input videos:
      - frame rate
      - height
      - width
      - num channels
    """

    input_files = [Path(el) for el in input_files]
    if len(input_files) == 0:
        raise ValueError("Should provide at least one video.")

    stream_info = get_video_stream_info(input_files[0])
    for input_file in input_files[1:]:
        other_stream_info = get_video_stream_info(input_file)
        if (
            other_stream_info.fps != stream_info.fps
            or other_stream_info.height != stream_info.height
            or other_stream_info.width != stream_info.width
            or other_stream_info.num_channels != stream_info.num_channels
        ):
            msg = f"Found incompatible videos: {input_file} {input_files[0]}"
            raise ValueError(msg)

    fps = stream_info.fps
    width = stream_info.width
    height = stream_info.height
    time_base = Fraction(1, 90000)
    pixel_format = "yuv420p10le"

    num_pixels = width * height
    if max_bits_per_pixel is None:
        max_bits_per_pixel = infer_target_bits_per_pixel(num_pixels, fps) * 2
    max_bits_per_sec = width * height * fps * max_bits_per_pixel
    pts_per_frame = 1 / (fps * time_base)

    encoder_options = {
        "preset": "6",
        "crf": "26",
        "svtav1-params": "tune=0",
        "maxrate": f"{round(max_bits_per_sec)}",
        "bufsize": f"{round(max_bits_per_sec * 2)}",
    }

    with av.open(output_file, "w", format="mp4", options={"movflags": "faststart"}) as container:
        stream = container.add_stream(
            "libsvtav1",
            fps,
            encoder_options,
            width=width,
            height=height,
            pix_fmt=pixel_format,
            gop_size=round(fps * 2),
            time_base=time_base,
        )
        stream = cast(av.VideoStream, stream)

        num_frames = 0
        reformatter = av.video.reformatter.VideoReformatter()
        for input_file in input_files:
            with VideoReader(input_file) as video_reader:
                for frame in video_reader.decode_frames():
                    frame = reformatter.reformat(frame, format=pixel_format)
                    frame.time_base = time_base
                    frame.pts = round(num_frames * pts_per_frame)
                    frame.dts = frame.pts
                    for packet in stream.encode(frame):
                        container.mux(packet)
                    num_frames += 1

        for packet in stream.encode():
            container.mux(packet)


def smart_copy_video_segment(
    input_file: str | Path,
    output_file: str | Path,
    from_frame_index: int,
    to_frame_index: int,
) -> None:
    """Copy a video segment by muxing whole GOPs and re-encoding only the partial GOPs at the ends.

    Assumes a frame depends on at most its nearest keyframe on each side, so a closed keyframe interval is safe to mux.
    """
    with VideoReader(input_file) as reader:
        start_keyframe_info = reader.get_next_keyframe_info(from_frame_index)
        end_keyframe_info = reader.get_prev_keyframe_info(to_frame_index - 1)

        # no complete keyframe interval inside the cut
        if start_keyframe_info is None or start_keyframe_info.frame_index > end_keyframe_info.frame_index:
            encode_copy_video_segment(input_file, output_file, from_frame_index, to_frame_index)
            return

        body_from_index = start_keyframe_info.frame_index
        body_to_index = end_keyframe_info.frame_index + 1  # exclusive bound of the closed interval

        # both ends already keyframe-aligned
        if from_frame_index == body_from_index and to_frame_index == body_to_index:
            mux_copy_video_segment(input_file, output_file, from_frame_index, to_frame_index)
            return

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            segment_files: list[Path] = []

            if from_frame_index < body_from_index:
                head_file = tmp_dir / "head.mp4"
                encode_copy_video_segment(input_file, head_file, from_frame_index, body_from_index)
                segment_files.append(head_file)

            body_file = tmp_dir / "body.mp4"
            mux_copy_video_segment(input_file, body_file, body_from_index, body_to_index)
            segment_files.append(body_file)

            if body_to_index < to_frame_index:
                tail_file = tmp_dir / "tail.mp4"
                encode_copy_video_segment(input_file, tail_file, body_to_index, to_frame_index)
                segment_files.append(tail_file)

            mux_concat_videos(segment_files, output_file)
