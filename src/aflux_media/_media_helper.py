import heapq
from collections.abc import Iterator
from pathlib import Path

import av
import av.stream


def mux_audio_video(
    audio_file: str | Path,
    video_file: str | Path,
    output_file: str | Path,
) -> None:
    """Mux an audio file and a video file into a single MP4 container."""
    with (
        av.open(audio_file) as audio_container,
        av.open(video_file) as video_container,
        av.open(output_file, "w", format="mp4", options={"movflags": "faststart"}) as output_container,
    ):
        if len(audio_container.streams.audio) == 0:
            raise ValueError("Provided audio file has no audio streams.")
        if len(video_container.streams.video) == 0:
            raise ValueError("Provided video file has no video streams.")

        input_audio_stream = audio_container.streams.audio[0]
        input_video_stream = video_container.streams.video[0]

        output_audio_stream = output_container.add_stream_from_template(input_audio_stream, True)
        output_video_stream = output_container.add_stream_from_template(input_video_stream, True)

        def demux_stream(container: av.container.InputContainer, stream: av.stream.Stream) -> Iterator[av.Packet]:
            first_pts = None
            for packet in container.demux(stream):
                if packet.size == 0 and packet.dts is None and packet.pts is None:
                    continue

                assert packet.pts is not None, "Packet should have a valid pts."

                if first_pts is None:
                    first_pts = packet.pts

                packet.pts -= first_pts
                packet.dts = None
                yield packet

        audio_packets = demux_stream(audio_container, input_audio_stream)
        video_packets = demux_stream(video_container, input_video_stream)

        def _get_physical_time(packet: av.Packet) -> float:
            assert packet.pts is not None
            assert packet.stream.time_base is not None
            return float(packet.pts * packet.stream.time_base)

        merged_packets = heapq.merge(audio_packets, video_packets, key=_get_physical_time)

        for packet in merged_packets:
            if isinstance(packet.stream, av.AudioStream):
                packet.stream = output_audio_stream
            else:
                packet.stream = output_video_stream
            output_container.mux(packet)
