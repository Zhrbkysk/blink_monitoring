import cv2
from base_camera import BaseCamera
from datetime import datetime
from eyes_state import get_eyes_state


class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)

        if not camera.isOpened():
            raise RuntimeError('Could not start camera!')

        while True:
            # read current frame
            _, img = camera.read()
            img = img[:, ::-1, :]
            img_debug, eyes_state = get_eyes_state(img)
            if eyes_state == 0:
                face = cv2.imread("./static/images/close_close.jpg")
            if eyes_state == 1:
                face = cv2.imread("./static/images/close_open.jpg")
            if eyes_state == 2:
                face = cv2.imread("./static/images/open_close.jpg")
            if eyes_state == 3:
                face = cv2.imread("./static/images/open_open.jpg")

            # カメラ画像の高さと目の開閉状態の表す画像の高さを揃える
            scale = img.shape[0] / face.shape[0]
            face = cv2.resize(face, dsize=None, fx=scale, fy=scale)

            img = cv2.hconcat([img, face])

            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
