import math
import random
import time

import matplotlib.pyplot as plt
import aptdaemon.core
import numpy as np

from fit_model import fit, fit3

number_of_user = 2

models = ["yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt"]
m = [0 for i in range(number_of_user)]


d = np.array([0.05, 0.08, 0.12, 0.17, 0.23, 0.37, 0.55])/12

r = np.array([128, 192, 256, 320, 384, 512, 640])

# r = np.array([128 * 128, 192 * 192, 256 * 256, 320 * 320, 384 * 384, 512 * 512, 640 * 640])

# fit(d, x=r, eva=r, gt=d)

c = [
    [0.0179, 0.0181, 0.0192, 0.0197, 0.0221, 0.0286, 0.0399],
    [0.0148, 0.0143, 0.0146, 0.0148, 0.0157, 0.0192, 0.0242],
    [0.0108, 0.0109, 0.0114, 0.0118, 0.013, 0.0144, 0.0184],
    [0.0087, 0.0089, 0.0095, 0.0096, 0.0101, 0.0112, 0.0127]
]

map = [
    [0.33, 0.43, 0.48, 0.51, 0.53, 0.56, 0.56],
    [0.31, 0.4, 0.46, 0.49, 0.52, 0.54, 0.54],
    [0.28, 0.37, 0.43, 0.47, 0.49, 0.51, 0.52],
    [0.24, 0.32, 0.37, 0.4, 0.43, 0.44, 0.44]
]

if __name__ == '__main__':

    tau = 1

    s = [12, 6, 10, 8]
    p = [0.515, 0.59, 0.5, 0.45]
    v = [4.15, 4.24, 4.04, 3.95]
    m = [0, 0, 0, 0]

    nu = [0.25 for i in range(len(s))]

    r = np.array([128, 192, 256, 320, 384, 512, 640])

    co_map = [
        [0.6275, 37.9476],
        [0.6110, 38.8654],
        [0.5875, 39.8489],
        [0.5045, 34.0794]
    ]

    beta = 1

    n = [0.2, 0.4, 0.6, 0.8, 1.0]  # [0.25 0.28 0.25 0.23]
    opt = []

    """
    for i in range(len(s)):
        cost = []
        for xx in n:
            for x in r:
                cost.append(s[i] * p[i] * v[i] * (-0.009630738496795 + 0.000081516654136 * x) / xx - s[i] * beta * (
                        co_map[m[i]][0] - co_map[m[i]][1] / x))

            opt.append(math.sqrt(beta * co_map[m[i]][1] * xx / (p[i] * v[i] * 0.000081516654136)))

        plt.plot(r, cost, label=str(i))
    """

    models = ["yolov5x.pt", "yolov5l.pt", "yolov5m.pt", "yolov5s.pt"]

    d1 = np.array([0.37, 0.66, 1.04, 1.53, 2.11, 3.55, 5.44]) / 12  # 10 * RBs

    d2 = np.array([0.29, 0.39, 0.54, 0.77, 1.05, 1.72, 2.64]) / 12  # 20

    d3 = np.array([0.25, 0.33, 0.4, 0.52, 0.7, 1.14, 1.74]) / 12  # 30

    d4 = np.array([0.05, 0.08, 0.12, 0.17, 0.23, 0.37, 0.55]) / 12 # WiFi

    i = 0
    for m, model in enumerate(models):
        # m = 2
        cost = []
        for index, resolution in enumerate(r):
            cost.append(s[i] * p[i] * v[i] * d3[index] - s[i] * beta * map[m][index])  #
        plt.scatter(r, cost, label=model)

        print(cost)
    # rr = [192, 258, 320, 384 (512)]

    plt.legend()
    # plt.show()










