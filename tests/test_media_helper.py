from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from aflux_media import AudioReader, VideoReader, mux_audio_video


@pytest.fixture(scope="session")
def audio_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    audio_file = tmp_path_factory.mktemp("media_audio") / "audio.m4a"
    sample_rate = 48000
    with av.open(audio_file, "w", format="mp4", options={"movie_timescale": str(sample_rate)}) as container:
        stream = container.add_stream("aac", rate=sample_rate)
        stream.layout = "stereo"
        stream.time_base = Fraction(1, sample_rate)
        array = np.full((2, 2048), 0.5, dtype=np.float32)
        frame = av.AudioFrame.from_ndarray(array, format="fltp", layout="stereo")
        frame.sample_rate = sample_rate
        frame.time_base = stream.time_base
        frame.pts = 0
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return audio_file


@pytest.fixture(scope="session")
def video_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    video_file = tmp_path_factory.mktemp("media_video") / "video.mp4"
    with av.open(video_file, "w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = 128
        stream.height = 128
        stream.pix_fmt = "yuv420p"

        for i in range(10):
            frame = av.VideoFrame(128, 128, "yuv420p")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return video_file


class TestMuxAudioVideo:
    def test_mux_audio_video(self, audio_file: Path, video_file: Path, tmp_path: Path) -> None:
        output_file = tmp_path / "output.mp4"
        mux_audio_video(audio_file, video_file, output_file)

        # Verify stream counts
        with av.open(output_file) as container:
            assert len(container.streams.audio) == 1
            assert len(container.streams.video) == 1

        # Verify consumer path readers
        audio_reader = AudioReader(output_file)
        assert audio_reader.get_stream_info().num_samples > 0

        video_reader = VideoReader(output_file)
        assert video_reader.get_stream_info().num_frames == 10
