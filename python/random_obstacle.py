import os
import cv2
import PIL.Image
import numpy as np
from tensorflow_helnet.utils import rand

class RandomObstacleGenerator():
    def __init__(self):
        N_OBSTACLES = 14
        DIR = os.path.dirname(os.path.realpath(__file__)) + "/../samples/"
        self.samples = map(cv2.imread, [
            "{}/{:02d}.png".format(DIR, i) for i in range(N_OBSTACLES)
        ])

    def __call__(self, bg, MAX_OBSTACLES=5):
        n_obstacles = np.random.randint(MAX_OBSTACLES)

        if n_obstacles == 0:
            return bg, np.zeros(bg.shape[:2], dtype=np.bool)

        total_mask = False
        result = np.copy(bg)

        try:
            for i in range(n_obstacles):
                result, mask = self.gen_obstacles(result)
                total_mask |= mask
        except:
            return bg, np.zeros(bg.shape[:2], dtype=np.bool)

        return result, total_mask

    def paste(self, bg, fg, x0, y0):

        # print "bg.shape = {}, fg.shape = {}, x0 = {}, y0 = {}".format(bg.shape, fg.shape, x0, y0)

        mask = (fg[:, :, -1] > 10)[..., None]

        s0, s1 = fg.shape[:2]

        merged = fg * mask + bg[y0:y0+s0, x0:x0+s1, :] * (1.0 - mask)
        merged = cv2.GaussianBlur(merged, (3, 3), 1)

        bg[y0:y0+s0, x0:x0+s1, :] = merged

        new_mask = np.zeros(bg.shape[:2], dtype=np.bool)
        new_mask[y0:y0+mask.shape[0], x0:x0+mask.shape[1]] = mask.squeeze()

        return bg, new_mask

    def get_random_location_and_size(self, bg):
        H, W = 544, 1024

        x_min, x_max = int(0.2 * W), int(0.8 * W)
        y_min, y_max = int(H / 3), int(H - 170)

        y0 = int(rand(y_min, y_max))
        x0 = int(rand(x_min, x_max))

        size = min(1., 10880. / (3 * (H - y0) - 476.) / 170)

        '''
        print "x0 = {} ~ U({}, {}), y0 = {} ~ U({}, {}), size = {}".format(
            x0, x_min, x_max, y0, y_min, y_max, size
        )
        '''

        rH = float(bg.shape[0]) / H
        rW = float(bg.shape[1]) / W

        x0 = int(np.floor(x0 * rW))
        y0 = int(np.floor(y0 * rH))
        size = (rH * size, rW * size)

        return x0, y0, size

    def gen_obstacles(self, bg):

        x0, y0, size = self.get_random_location_and_size(bg)

        idx = np.random.randint(len(self.samples))
        obstacle = self.samples[idx]

        # Random flipping
        if rand() > 0.5:
            obstacle = obstacle[:, ::-1, ...]

        oW = int(obstacle.shape[1] * size[1])
        oH = int(obstacle.shape[0] * size[0])

        rescaled_obstacle = cv2.resize(obstacle, (oW, oH))

        result, mask = self.paste(bg, rescaled_obstacle, x0, y0)

        return result, mask
