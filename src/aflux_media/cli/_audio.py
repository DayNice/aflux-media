from cyclopts import App

from aflux_media import AudioReader

from ._parameters import InputFile

app = App(name="audio", help="Inspect an audio.")


@app.command
def stream(audio: InputFile) -> None:
    """Get audio stream information."""
    info = AudioReader(audio).get_stream_info()
    print(info.model_dump_json())
