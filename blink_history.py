import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time


class BlinkHistory:
    def __init__(self, width):
        width_inch = width / 100
        height_inch = width_inch / 4
        self.fig, self.ax = plt.subplots(figsize=(width_inch, height_inch))
        self.ax.set_xlim([60, 0])
        self.ax.set_xlabel("second")
        self.ax.set_ylim([-0.5, 5.5])
        self.blink_per_second = [0 for _ in range(61)]
        self.eyes_state_previous = 3
        self.time_previous = time.time() - 1  # 始めのself.add_eyes_state()で描画させるため
        self.blink_count_second = 0
        self.graph_image = None

        self.add_eyes_state(3)

    # 瞬き判定　両目開き→両目とじのみ　片目から閉じたときに判定されない
    def is_blink(self, eyes_state):
        is_blink = False
        if self.eyes_state_previous == 3 and eyes_state == 0:
            is_blink = True
            print("close")
        self.eyes_state_previous = eyes_state
        return is_blink

    def add_eyes_state(self, eyes_state):
        if self.is_blink(eyes_state):
            self.blink_count_second += 1
        time_now = time.time()
        if time_now - self.time_previous >= 1:
            self.blink_per_second.pop()
            self.blink_per_second = [self.blink_count_second] + self.blink_per_second

            if self.ax.lines:
                self.ax.lines.pop()
            self.ax.plot(self.blink_per_second, color="Blue", lw=0.5)
            self.fig.canvas.draw()

            shape = self.fig.canvas.get_width_height()[::-1] + (3,)
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(shape)
            data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
            self.graph_image = data

            self.blink_count_second = 0
            self.time_previous = time_now

    def get_graph_image(self):
        return self.graph_image
