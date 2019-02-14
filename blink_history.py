import matplotlib
matplotlib.use('Agg')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
from matplotlib import pyplot as plt
import numpy as np
import cv2
import time
from dataclasses import dataclass
from dataclasses import field
from typing import List


# タイプアノテーションはあるかどうかのみで型自体は見ていない(らしい
@dataclass
class History:
    unit: str
    interval: int
    fig: int
    ax: int
    image: int = 0
    blink_per_time: List[int] = field(default_factory=list)#[0 for _ in range(61)])
    time_previous: float = 0.0  # 始めのself.add_eyes_state()で描画させるため1引いておく
    blink_count: int = 0

    def __post_init__(self):
        self.blink_per_time = [0 for _ in range(61)]
        self.ax.set_xlim([60, 0])
        self.ax.set_xlabel(self.unit)
        if self.unit == "second":
            self.ax.set_ylim([-0.5, 5.5])
        else:
            self.ax.set_ylim([-4, 44])


class BlinkHistory:
    def __init__(self, width):
        width_inch = width / 100
        height_inch = width_inch / 4
        fig1, ax1 = plt.subplots(figsize=(width_inch, height_inch))
        fig2, ax2 = plt.subplots(figsize=(width_inch, height_inch))
        self.graph = {
            "second": History("second", 1, fig1, ax1),
            "minute": History("minute", 60, fig2, ax2)
        }
        self.eyes_state_previous = 3
        self.add_eyes_state(3)

    # 瞬き判定　両目開き→両目とじのみ　片目から閉じたときに判定されない
    def is_blink(self, eyes_state):
        is_blink = False
        if self.eyes_state_previous == 3 and eyes_state == 0:
            is_blink = True
            print("eyes close")
        self.eyes_state_previous = eyes_state
        return is_blink

    def add_eyes_state(self, eyes_state):
        if self.is_blink(eyes_state):
            self.graph["second"].blink_count += 1
            self.graph["minute"].blink_count += 1
        time_now = time.time()
        for index in self.graph.keys():
            if time_now - self.graph[index].time_previous >= self.graph[index].interval:
                self.graph[index].blink_per_time.pop()
                self.graph[index].blink_per_time = [self.graph[index].blink_count] + self.graph[index].blink_per_time

                if self.graph[index].ax.lines:
                    self.graph[index].ax.lines.pop()
                self.graph[index].ax.plot(self.graph[index].blink_per_time, color="Blue", lw=0.5)
                self.graph[index].fig.canvas.draw()

                shape = self.graph[index].fig.canvas.get_width_height()[::-1] + (3,)
                graph_image = np.frombuffer(self.graph[index].fig.canvas.tostring_rgb(), dtype=np.uint8)
                graph_image = graph_image.reshape(shape)
                graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGB2BGR)
                self.graph[index].image = graph_image

                self.graph[index].blink_count = 0
                self.graph[index].time_previous = time_now

    def get_graph_image(self, unit):
        return self.graph[unit].image
