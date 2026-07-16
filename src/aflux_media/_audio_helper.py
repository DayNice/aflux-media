from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from ._audio_reader import AudioReader


def encode_copy_audio_segment(
    input_file: str | Path,
    output_file: str | Path,
    from_sample_index: int,
    to_sample_index: int,
) -> None:
    """Copy an audio segment by encoding samples into an M4A file."""
    if from_sample_index < 0 or to_sample_index <= from_sample_index:
        msg = f"Sample range should be non-empty and non-negative: from={from_sample_index}, to={to_sample_index}"
        raise ValueError(msg)

    reader = AudioReader(input_file)
    stream_info = reader.get_stream_info()

    with av.open(
        output_file,
        "w",
        format="mp4",
        options={"movie_timescale": str(stream_info.sample_rate)},
    ) as output_container:
        output_stream = output_container.add_stream("aac", rate=stream_info.sample_rate)
        output_stream.layout = stream_info.channel_layout
        output_stream.time_base = Fraction(1, stream_info.sample_rate)

        for block in reader.decode_blocks(from_sample_index, to_sample_index):
            array = np.ascontiguousarray(block.samples.T)
            frame = av.AudioFrame.from_ndarray(array, format="fltp", layout=stream_info.channel_layout)
            frame.sample_rate = stream_info.sample_rate
            frame.time_base = output_stream.time_base
            frame.pts = block.from_sample_index - from_sample_index
            for packet in output_stream.encode(frame):
                output_container.mux(packet)

        for packet in output_stream.encode():
            output_container.mux(packet)
