from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import PIL.Image
import pytest

import aflux_media
from aflux_media import VideoReader


def _write_gradient_video(output_file: Path, num_frames: int) -> None:
    images = (
        PIL.Image.new("RGB", (128, 128), color=(v, v, v))
        for v in (round(255 * i / (num_frames - 1)) for i in range(num_frames))
    )
    aflux_media.encode_images_into_video(images, output_file, fps=10)


@pytest.fixture(scope="session")
def tmp_gradient_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    video_file = tmp_path_factory.mktemp("gradient") / "video.mp4"
    _write_gradient_video(video_file, 100)
    return video_file


@pytest.fixture(scope="session")
def tmp_keyframe_end_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    video_file = tmp_path_factory.mktemp("keyframe_end") / "video.mp4"
    # final frame lands on a keyframe at gop_size = round(FPS * 2)
    _write_gradient_video(video_file, 101)
    with VideoReader(video_file) as video_reader:
        num_frames = video_reader.get_stream_info().num_frames
        keyframe_info = video_reader.get_prev_keyframe_info(num_frames - 1)
    assert keyframe_info.frame_index == num_frames - 1
    return video_file


class TestVideoReader:
    def test_get_stream_info(self, tmp_gradient_video: Path) -> None:
        with VideoReader(tmp_gradient_video) as video_reader:
            info = video_reader.get_stream_info()

        assert info.width == 128
        assert info.height == 128
        assert info.num_frames == 100
        assert info.fps == Fraction(10, 1)

    def test_decode_frames(self, tmp_gradient_video: Path) -> None:
        with VideoReader(tmp_gradient_video) as reader:
            frames = list(reader.decode_frames([99, 0, 49]))

        assert len(frames) == 3

        arr_99 = frames[0].to_ndarray(format="rgb24")
        arr_0 = frames[1].to_ndarray(format="rgb24")
        arr_49 = frames[2].to_ndarray(format="rgb24")

        # Frame 99 is white
        assert np.mean(arr_99) > 200

        # Frame 0 is black
        assert np.mean(arr_0) < 50

        # Frame 49 is mid-gray (~126/255)
        assert np.mean(arr_49) == pytest.approx(126, abs=10)

    def test_get_frame_infos_with_indices(self, tmp_gradient_video: Path) -> None:
        with VideoReader(tmp_gradient_video) as reader:
            infos = reader.get_frame_infos([8, 2, 4])

        assert len(infos) == 3
        assert infos[0].frame_index == 8
        assert infos[1].frame_index == 2
        assert infos[2].frame_index == 4

    def test_invalid_frame_indices(self, tmp_gradient_video: Path) -> None:
        with VideoReader(tmp_gradient_video) as reader:
            with pytest.raises(ValueError, match="should be unique"):
                reader.get_frame_infos([0, 0, 1])

            with pytest.raises(ValueError, match="should be non-negative"):
                reader.get_frame_infos([-1, 0])

            with pytest.raises(ValueError, match="should be less than size"):
                reader.get_frame_infos([0, 9999])

    def test_compute_statistics(self, tmp_gradient_video: Path) -> None:
        with VideoReader(tmp_gradient_video) as reader:
            stats = reader.compute_statistics()

        assert stats.sample_size == 100

        # Gradient from black to white; all channels are equal (grayscale)
        for channel in range(3):
            assert stats.mean[channel] == pytest.approx(0.5, abs=0.1)
            assert stats.max[channel] > 0.9


class TestVideoStatisticsMerge:
    def test_merge_video_statistics_list(self, tmp_path: Path) -> None:
        video_path1 = tmp_path / "test1.mp4"
        video_path2 = tmp_path / "test2.mp4"

        red_image = PIL.Image.new("RGB", (128, 128), color=(255, 0, 0))
        blue_image = PIL.Image.new("RGB", (128, 128), color=(0, 0, 255))

        aflux_media.encode_images_into_video([red_image] * 5, video_path1, fps=Fraction(30, 1))
        aflux_media.encode_images_into_video([blue_image] * 5, video_path2, fps=Fraction(30, 1))

        with VideoReader(video_path1) as reader1:
            stats1 = reader1.compute_statistics()

        with VideoReader(video_path2) as reader2:
            stats2 = reader2.compute_statistics()

        merged_stats = aflux_media.merge_video_statistics_list([stats1, stats2])

        assert merged_stats.sample_size == stats1.sample_size + stats2.sample_size

        # Red channel (from first video)
        assert merged_stats.mean[0] == pytest.approx(0.5, abs=0.1)
        assert merged_stats.max[0] > 0.8

        # Green channel
        assert merged_stats.mean[1] == pytest.approx(0.0, abs=0.1)
        assert merged_stats.max[1] < 0.2

        # Blue channel (from second video)
        assert merged_stats.mean[2] == pytest.approx(0.5, abs=0.1)
        assert merged_stats.max[2] > 0.8


class TestSmartCopyVideoSegment:
    @staticmethod
    def _decode_means(video_file: Path) -> list[float]:
        """Mean intensity of each decoded frame, normalized to [0, 1]."""
        return [
            float(frame.to_ndarray(format="rgb24").mean()) / 255
            for frame in aflux_media.decode_video_frames(video_file)
        ]

    def test_segment(self, tmp_gradient_video: Path, tmp_path: Path) -> None:
        with VideoReader(tmp_gradient_video) as reader:
            keyframe_indices = [info.frame_index for info in reader.get_keyframe_infos()]
            num_frames = reader.get_stream_info().num_frames
        assert len(keyframe_indices) >= 4

        cases = [
            (1, 0, 2, 1),  # aligned ends -> pure mux
            (1, 3, 3, 1),  # head + body
            (1, 0, 3, 5),  # body + tail
            (1, 3, 3, 5),  # head + body + tail
            (1, 1, 1, 4),  # within one GOP -> full re-encode
        ]
        output = tmp_path / "out.mp4"
        for from_keyframe_index, from_offset, to_keyframe_index, to_offset in cases:
            from_index = keyframe_indices[from_keyframe_index] + from_offset
            to_index = keyframe_indices[to_keyframe_index] + to_offset

            aflux_media.smart_copy_video_segment(tmp_gradient_video, output, from_index, to_index)

            with VideoReader(output) as reader:
                assert reader.get_stream_info().num_frames == to_index - from_index

            means = self._decode_means(output)
            assert len(means) == to_index - from_index
            for k, mean in enumerate(means):
                expected = (from_index + k) / (num_frames - 1)
                assert mean == pytest.approx(expected, abs=0.04)

    def test_copy_to_keyframe_end(self, tmp_keyframe_end_video: Path, tmp_path: Path) -> None:
        with VideoReader(tmp_keyframe_end_video) as reader:
            keyframe_indices = [info.frame_index for info in reader.get_keyframe_infos()]
            num_frames = reader.get_stream_info().num_frames

        from_index = keyframe_indices[1]
        output = tmp_path / "out.mp4"
        aflux_media.smart_copy_video_segment(tmp_keyframe_end_video, output, from_index, num_frames)

        means = self._decode_means(output)
        assert len(means) == num_frames - from_index
        for k, mean in enumerate(means):
            expected = (from_index + k) / (num_frames - 1)
            assert mean == pytest.approx(expected, abs=0.04)

    def test_copy_to_end(self, tmp_gradient_video: Path, tmp_path: Path) -> None:
        with VideoReader(tmp_gradient_video) as reader:
            keyframe_indices = [info.frame_index for info in reader.get_keyframe_infos()]
            num_frames = reader.get_stream_info().num_frames

        from_index = keyframe_indices[1] + 2
        to_index = num_frames

        output = tmp_path / "out.mp4"
        aflux_media.smart_copy_video_segment(tmp_gradient_video, output, from_index, to_index)

        means = self._decode_means(output)
        assert len(means) == to_index - from_index
        for k, mean in enumerate(means):
            expected = (from_index + k) / (num_frames - 1)
            assert mean == pytest.approx(expected, abs=0.04)


class TestMuxConcatVideos:
    @staticmethod
    def _write_h264_gop_video(output_file: Path, num_frames: int) -> None:
        with av.open(output_file, "w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=30, options={"bf": "2"})
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            stream.gop_size = 60
            stream.time_base = Fraction(1, 15360)

            for index in range(num_frames):
                image = PIL.Image.new("RGB", (64, 64), color=(0, 0, 0))
                frame = av.VideoFrame.from_image(image)
                frame.pts = index * 512
                frame.time_base = Fraction(1, 15360)
                for packet in stream.encode(frame):
                    container.mux(packet)

            for packet in stream.encode():
                container.mux(packet)

    @pytest.fixture
    def tmp_video_with_inaccurate_container_duration(
        self,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> Path:
        input_dir = tmp_path_factory.mktemp("h264_inaccurate_duration")
        input_file = input_dir / "input.mp4"
        self._write_h264_gop_video(input_file, 121)
        input_info = aflux_media.get_video_stream_info(input_file)

        with av.open(input_file) as input_container:
            input_stream = input_container.streams.video[0]
            pts_per_frame = 1 / (input_info.fps * input_info.time_base)
            assert pts_per_frame.denominator == 1
            assert input_stream.duration != input_info.num_frames * pts_per_frame.numerator

        return input_file

    def test_preserves_pts_per_frame(
        self,
        tmp_video_with_inaccurate_container_duration: Path,
    ) -> None:
        input_file = tmp_video_with_inaccurate_container_duration
        input_stream_info = aflux_media.get_video_stream_info(input_file)

        pts_per_frame = 1 / (input_stream_info.fps * input_stream_info.time_base)
        assert pts_per_frame.denominator == 1
        pts_per_frame = pts_per_frame.numerator

        output_file = input_file.parent / "output.mp4"
        aflux_media.mux_concat_videos([input_file, input_file], output_file)

        output_stream_info = aflux_media.get_video_stream_info(output_file)
        assert output_stream_info.fps == input_stream_info.fps
        assert output_stream_info.time_base == input_stream_info.time_base
        assert output_stream_info.num_frames == input_stream_info.num_frames * 2

        frame_infos = aflux_media.get_video_frame_infos(output_file)
        frame_pts_list = [frame.pts for frame in frame_infos]
        assert frame_pts_list == [i * pts_per_frame for i in range(output_stream_info.num_frames)]
