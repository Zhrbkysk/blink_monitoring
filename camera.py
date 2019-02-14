import cv2
from base_camera import BaseCamera
from eye_detector import EyeDetector
from blink_history import BlinkHistory


class Camera(BaseCamera):
    video_source = 0

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        detector = EyeDetector("opencv")

        if not camera.isOpened():
            raise RuntimeError('Could not start camera!')

        state_image_path = ["./static/images/close_close.jpg",
                            "./static/images/close_open.jpg",
                            "./static/images/open_close.jpg",
                            "./static/images/open_open.jpg"]
        state_image_list = [cv2.imread(p) for p in state_image_path]
        state_image_shape = state_image_list[0].shape
        # カメラ画像の高さと目の開閉状態の表す画像の高さを揃える
        _, img = camera.read()
        scale = img.shape[0] / state_image_shape[0]
        scaling = lambda img: cv2.resize(img, dsize=None, fx=scale, fy=scale)
        state_image_list = list(map(scaling, state_image_list))

        width = img.shape[1] + state_image_list[0].shape[1]
        history = BlinkHistory(width)

        tick_previous = cv2.getTickCount()

        while True:
            # read current frame
            _, img = camera.read()
            # 左右反転
            img = img[:, ::-1, :]
            img_debug, eyes_state = detector.get_eyes_state(img)

            state_image = state_image_list[eyes_state]
            img = cv2.hconcat([img_debug, state_image])

            tick = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (tick - tick_previous)
            tick_previous = tick
            cv2.putText(img, "FPS:{} ".format(int(fps)),
                        (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2, cv2.LINE_AA)

            history.add_eyes_state(eyes_state)
            graph_second = history.get_graph_image("second")
            graph_minute = history.get_graph_image("minute")
            img = cv2.vconcat([img, graph_second, graph_minute])


            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
