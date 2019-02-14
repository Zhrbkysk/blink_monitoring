from flask import Flask
from flask import Response
from flask import render_template
from camera import Camera

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


def streaming(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route("/streaming_camera")
def streaming_camera():
    return Response(streaming(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
