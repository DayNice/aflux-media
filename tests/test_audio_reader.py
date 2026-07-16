from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
from pydantic import ValidationError

from aflux_media import AudioBlock, AudioReader, encode_concat_audios, encode_copy_audio_segment


@pytest.fixture(scope="session")
def aac_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    audio_file = tmp_path_factory.mktemp("aac_fixture") / "audio.m4a"
    sample_rate = 48000
    layout = "stereo"

    with av.open(
        audio_file,
        "w",
        format="mp4",
        options={
            "movie_timescale": str(sample_rate),
            "movflags": "faststart",
        },
    ) as container:
        stream = container.add_stream("aac", rate=sample_rate)
        stream.layout = layout

        for i in range(2):
            array = np.full((2, 1024), 0.5, dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(array, format="fltp", layout=layout)
            frame.sample_rate = sample_rate
            frame.time_base = Fraction(1, sample_rate)
            frame.pts = i * 1024
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    return audio_file


class TestAudioBlock:
    def test_rejects_inconsistent_fields(self) -> None:
        samples = np.zeros((2, 1), dtype=np.float32)
        samples.setflags(write=False)
        with pytest.raises(ValidationError, match="sample count should agree"):
            AudioBlock(
                from_sample_index=0,
                to_sample_index=2,
                from_timestamp=Fraction(0),
                to_timestamp=Fraction(1, 2),
                num_samples=1,
                block_size=2,
                samples=samples,
            )

    def test_rejects_writeable_samples(self) -> None:
        with pytest.raises(ValidationError, match="should be read-only"):
            AudioBlock(
                from_sample_index=0,
                to_sample_index=2,
                from_timestamp=Fraction(0),
                to_timestamp=Fraction(1, 2),
                num_samples=2,
                block_size=2,
                samples=np.zeros((2, 1), dtype=np.float32),
            )


class TestAudioReader:
    def test_stream_metadata(self, aac_file: Path) -> None:
        reader = AudioReader(aac_file)
        info = reader.get_stream_info()
        assert info.sample_rate == 48000
        assert info.num_channels == 2
        assert info.channel_layout == "stereo"
        assert info.codec == "aac"
        assert info.sample_format == "fltp"
        assert info.num_samples > 0

    def test_decode_blocks_sequential(self, aac_file: Path) -> None:
        reader = AudioReader(aac_file, block_size=1000)
        info = reader.get_stream_info()
        blocks = list(reader.decode_blocks())

        assert len(blocks) > 0

        for i, block in enumerate(blocks):
            assert block.samples.shape[1] == 2
            assert not block.samples.flags.writeable

            if i < len(blocks) - 1:
                assert block.num_samples == 1000
                assert block.samples.shape[0] == 1000
            else:
                assert block.num_samples <= 1000
                assert block.to_sample_index == info.num_samples

        for previous, current in zip(blocks, blocks[1:], strict=False):
            assert previous.to_sample_index == current.from_sample_index

    def test_decode_blocks_slicing(self, aac_file: Path) -> None:
        reader = AudioReader(aac_file, block_size=1000)
        start = 500
        end = 1500
        blocks = list(reader.decode_blocks(start, end))

        actual = np.concatenate([block.samples for block in blocks])
        assert actual.shape == (1000, 2)
        assert blocks[0].from_sample_index == 500
        assert blocks[-1].to_sample_index == 1500

    def test_decode_blocks_padding(self, aac_file: Path) -> None:
        reader = AudioReader(aac_file, block_size=1000)
        info = reader.get_stream_info()
        start = info.num_samples - 500
        end = info.num_samples + 500

        blocks = list(reader.decode_blocks(start, end))
        actual = np.concatenate([block.samples for block in blocks])

        assert actual.shape == (1000, 2)
        tail = actual[500:]
        np.testing.assert_array_equal(tail, 0)

    def test_invalid_arguments(self, aac_file: Path) -> None:
        reader = AudioReader(aac_file)
        assert list(reader.decode_blocks(20, 20)) == []
        with pytest.raises(ValueError, match="non-negative"):
            list(reader.decode_blocks(-1, 20))
        with pytest.raises(ValueError, match="non-negative"):
            reader.get_timestamp(-1)
        with pytest.raises(ValueError, match="non-negative"):
            reader.get_sample_index(Fraction(-1, 2))

    def test_timestamp_conversions(self, aac_file: Path) -> None:
        reader = AudioReader(aac_file)
        info = reader.get_stream_info()
        rate = info.sample_rate
        assert reader.get_sample_index(Fraction(1, 2 * rate)) == 0
        assert reader.get_sample_index(Fraction(3, 2 * rate)) == 2
        assert reader.get_sample_index(Fraction(1, rate)) == 1
        assert reader.get_timestamp(1) == Fraction(1, rate)


class TestEncodeCopyAudioSegment:
    def test_encode_copy_audio_segment(self, aac_file: Path, tmp_path: Path) -> None:
        output_file = tmp_path / "segment.m4a"
        from_sample_index = 500
        to_sample_index = 2500

        encode_copy_audio_segment(aac_file, output_file, from_sample_index, to_sample_index)

        with av.open(output_file) as container:
            assert len(container.streams.audio) == 1
            assert len(container.streams.video) == 0

        reader = AudioReader(output_file)
        info = reader.get_stream_info()

        assert info.num_samples == 2000

    def test_invalid_range(self, aac_file: Path, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="non-empty and non-negative"):
            encode_copy_audio_segment(aac_file, tmp_path / "out.m4a", -1, 10)
        with pytest.raises(ValueError, match="non-empty and non-negative"):
            encode_copy_audio_segment(aac_file, tmp_path / "out.m4a", 10, 10)
        with pytest.raises(ValueError, match="non-empty and non-negative"):
            encode_copy_audio_segment(aac_file, tmp_path / "out.m4a", 10, 9)


class TestEncodeConcatAudios:
    def test_valid_concat(self, aac_file: Path, tmp_path: Path) -> None:
        output_file = tmp_path / "concat.m4a"
        encode_concat_audios([aac_file, aac_file], output_file)

        reader = AudioReader(output_file)
        info = reader.get_stream_info()
        assert info.num_samples == 4096

        samples = []
        for block in reader.decode_blocks(0, info.num_samples):
            samples.append(block.samples)

        concat_samples = np.concatenate(samples, axis=0)
        assert concat_samples.shape == (4096, 2)
        # Using a generous tolerance (0.20) since AAC is lossy and MDCT introduces boundary variations
        np.testing.assert_allclose(concat_samples, 0.5, atol=0.20)

    def test_invalid_arguments(self, aac_file: Path, tmp_path: Path) -> None:
        output_file = tmp_path / "out.m4a"
        with pytest.raises(ValueError, match="Should provide at least one audio"):
            encode_concat_audios([], output_file)

        # Create a mismatched file
        mismatched_file = tmp_path / "mismatch.m4a"
        sample_rate = 44100
        with av.open(mismatched_file, "w", format="mp4") as container:
            stream = container.add_stream("aac", rate=sample_rate)
            stream.layout = "stereo"
            stream.time_base = Fraction(1, sample_rate)
            array = np.full((2, 1024), 0.5, dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(array, format="fltp", layout="stereo")
            frame.sample_rate = sample_rate
            frame.time_base = stream.time_base
            frame.pts = 0
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

        with pytest.raises(ValueError, match="Found incompatible audios"):
            encode_concat_audios([aac_file, mismatched_file], output_file)
