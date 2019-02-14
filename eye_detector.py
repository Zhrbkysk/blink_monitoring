import cv2
import dlib


class EyeDetector:
    def __init__(self, face_detection_type="opencv"):
        self.face_detection_type = face_detection_type
        if face_detection_type == "opencv":
            self.face_detector = cv2.CascadeClassifier('static/cascade/haarcascade_frontalface_alt2.xml')
        elif face_detection_type == "dlib":
            self.face_detector = dlib.get_frontal_face_detector()
        else:
            self.face_detection_type = "opencv"
            self.face_detector = cv2.CascadeClassifier('static/cascade/haarcascade_frontalface_alt2.xml')
        self.eye_detector = cv2.CascadeClassifier('static/cascade/haarcascade_eye_tree_eyeglasses.xml')
        # self.eye_detector = cv2.CascadeClassifier('static/cascade/haarcascade_eye.xml')
        self.face_previous = [160, 80, 320, 320]

    def get_unique_eyes(self, eyes):
        unique_eyes = []
        # eye_cascade.detectMultiScaleは目が検出されればnumpy、なければ空タプルを返す
        if len(eyes) >= 1:
            unique_eyes.append(eyes[0])
            eye_first_x_min = eyes[0][0]
            eye_first_x_max = eyes[0][0] + eyes[0][2]
            eye_first_y_min = eyes[0][1]
            eye_first_y_max = eyes[0][1] + eyes[0][3]
            eye_first_x_range = set(range(eye_first_x_min, eye_first_x_max))
            eye_first_y_range = set(range(eye_first_y_min, eye_first_y_max))
            for eye in eyes[1:]:
                eye_x_min = eye[0]
                eye_x_max = eye[0] + eye[2]
                eye_y_min = eye[1]
                eye_y_max = eye[1] + eye[3]
                eye_x_range = set(range(eye_x_min, eye_x_max))
                eye_y_range = set(range(eye_y_min, eye_y_max))

                is_x_overlap = bool(eye_first_x_range & eye_x_range)
                is_y_overlap = bool(eye_first_y_range & eye_y_range)

                if not is_x_overlap or not is_y_overlap:
                    unique_eyes.append(eye)
                    break

        return unique_eyes

    def low_pass_filter(self, face):
        face = [int(f_pre * 0.5 + f * 0.5) for f_pre, f in zip(self.face_previous, face)]
        return face

    def get_eyes_state(self, img):
        image = img.copy()
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self.face_detection_type == "opencv":
            faces = self.face_detector.detectMultiScale(
                image_gray, scaleFactor=1.11, minNeighbors=3, minSize=(100, 100))
        else:
            faces = self.face_detector(image_gray)

        if len(faces) >= 1:
            face = faces[0]
        else:
            face = self.face_previous

        if isinstance(face, dlib.rectangle):
            face = [face.left(), face.top(), face.width(), face.height()]

        face = self.low_pass_filter(face)

        face_left = face[0]
        face_top = face[1]
        face_right = face_left + face[2]
        face_bottom = face_top + face[3]

        cv2.rectangle(image, (face_left, face_top), (face_right, face_bottom), (255, 0, 0), 2)

        # # 処理高速化、精度向上のために顔の上半分を検出対象範囲とする
        face_upper_half = image_gray[face_top:(face_top + face_bottom) // 2, face_left:face_right]
        eyes = self.eye_detector.detectMultiScale(
            face_upper_half, scaleFactor=1.11, minNeighbors=3, minSize=(8, 8))

        unique_eyes = self.get_unique_eyes(eyes)

        for ex, ey, ew, eh in unique_eyes:
            cv2.rectangle(image, (face_left + ex, face_top + ey),
                          (face_left + ex + ew, face_top + ey + eh), (255, 255, 0), 1)

        # 画像では反転させているがright_eye, left_eyeは本来の右目左目
        is_right_eye_open = False
        is_left_eye_open = False
        face_x_center = (face_right - face_left) / 2
        for eye in eyes:
            if eye[0] < face_x_center:
                is_left_eye_open = True
            else:
                is_right_eye_open = True

        eyes_state = int(is_right_eye_open) * 1 + int(is_left_eye_open) * 2
        self.face_previous = face

        return image, eyes_state
