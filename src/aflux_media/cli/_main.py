from cyclopts import App

from ._audio import app as audio_app
from ._video import app as video_app

app = App(name="aflux-media", help="Interact with media files.")
app.register_install_completion_command()
app.command(audio_app, name="audio")
app.command(video_app, name="video")
