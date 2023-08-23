from flask import Flask, render_template, Response
from main import VideoCamera

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return VideoCamera()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port="3500", debug=True)
