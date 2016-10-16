#!/usr/bin/python
import cv2
import numpy as np
from random_obstacle import RandomObstacleGenerator

def main():
    bg = cv2.imread("example.jpg")
    bg = cv2.resize(bg, (512, 272))

    rog = RandomObstacleGenerator()

    for i in range(100):
        result, mask = rog(bg, 10)
        cv2.imwrite("results/{:02d}.png".format(i), result)
        cv2.imwrite("results/{:02d}_mask.png".format(i), mask.astype(np.uint8) * 255)

main()
