import json
from collections.abc import Iterator
from typing import Literal, cast

import av
import PIL.Image
import rich.progress
from cyclopts import App
from rich.console import Console

from aflux_media import (
    decode_video_frames,
    get_video_frame_infos,
    get_video_keyframe_infos,
    get_video_stream_info,
)

from ._parameters import Indices, InputFile, OutputDir

app = App(name="video", help="Inspect a video.")


@app.command
def stream(video: InputFile) -> None:
    """Get video stream information."""
    info = get_video_stream_info(video)
    print(info.model_dump_json())


@app.command
def frames(video: InputFile, *, keyframes_only: bool = False) -> None:
    """Get video frame informations."""
    if keyframes_only:
        frame_infos = get_video_keyframe_infos(video)
    else:
        frame_infos = get_video_frame_infos(video)
    for frame_info in frame_infos:
        print(frame_info.model_dump_json())


@app.command
def frame_images(
    video_file: InputFile,
    output_dir: OutputDir,
    indices: Indices,
    *,
    image_suffix: Literal[".png", ".jpg", ".webp", ".avif"] = ".png",
    progress_bar: bool = False,
) -> None:
    """Save frames at given indices as image files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    iterator = zip(indices, decode_video_frames(video_file, indices))
    iterator = cast(Iterator[tuple[int, av.VideoFrame]], iterator)
    iterator = rich.progress.track(
        iterator,
        total=len(indices),
        console=Console(stderr=True),
        disable=not progress_bar,
    )

    for index, frame in iterator:
        image_name = f"frame_{index:06d}{image_suffix}"
        image_file = output_dir / image_name
        pil_image = cast(PIL.Image.Image, frame.to_image())
        pil_image.save(image_file)
        print(json.dumps({"frame_index": index, "image_file": image_file.as_posix()}))
