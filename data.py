import base64
import glob
import json
import math
from pathlib import Path
from random import sample

import cv2
import numpy as np
import matplotlib.pyplot as plt

images = []


p = Path("../datasets/images/val2017")  # os-agnostic
if p.is_dir():  # dir
    images = glob.glob(str(p / '**' / '*.jpg'), recursive=True)

r = np.array([128, 192, 256, 320, 384, 512, 640])

d_size = np.array([12.67, 25.27, 41.99, 63.55, 86.40, 140.0, 219])

d_size = [0.098984375, 0.197421875, 0.328046875, 0.496484375, 0.675, 1.09375, 1.7109375]

# d_size = [374.89, 744.51, 1249.35, 1884.81, 2543.219, 4201.79, 6539.48]


# 10 PRBs 12 FPS
d_time = [0.37, 0.66, 1.04, 1.53, 2.11, 3.55, 5.44]
d_rate = []
for i in range(len(d_size)):
    d_rate.append(d_size[i] * 12 / d_time[i])
print(np.round(np.array(d_rate) * 8 / 1024, 4))

# 20 PRBs 12 FPS
d_time = [0.29, 0.39, 0.54, 0.77, 1.05, 1.72, 2.64]
d_rate = []
for i in range(len(d_size)):
    d_rate.append(d_size[i] * 12 / d_time[i])
print(np.round(np.array(d_rate)* 8 / 1024, 4))

# 30 PRBs 12 FPS
d_time = [0.25, 0.33, 0.4, 0.52, 0.7, 1.14, 1.74]
d_rate = []
for i in range(len(d_size)):
    d_rate.append(d_size[i] * 12 / d_time[i])
print(np.round(np.array(d_rate) * 8 / 1024, 4))

# WiFi
d_time = [0.0849, 0.1636, 0.26, 0.38, 0.5165, 0.9008, 1.3595]
d_rate = []
for i in range(len(d_size)):
    d_rate.append(d_size[i] * 30 / d_time[i])
print(np.round(np.array(d_rate) * 8 / 1024, 0))

plt.plot(np.array(d_rate) * 8 / 1024)
#$plt.show()

print("################")
d_time = [0.0408, 0.0619, 0.1016, 0.1297, 0.1719, 0.2774, 0.3932]
d_rate = []
for i in range(len(d_size)):
    d_rate.append(d_size[i] * 10 / d_time[i])
print(np.round(np.average(d_rate) * 8 / 1024, 2))


prb = [10, 20, 30]
mbps = [3.7, 7.7, 11.5]

factor = []
for i in range(3):
    bandwidth = prb[i] * 180 * math.pow(10, -3)
    print(bandwidth)
    print(math.log2(1+math.pow(10, 20/10)) * bandwidth, "Mbps")
    factor.append(mbps[i]/bandwidth)

print(np.average(factor))

print(math.log2(1 + math.pow(10, 38/20)) * 5, "Mbps")


"""
r = np.array([128, 192, 256, 320, 384, 512, 640])
sizes = []
info = []
ori_file = []
#for i, item in enumerate(images):
i = 0
while True:
    iid = sample(images, 30)
    for item in iid:
        im0s = cv2.imread(item)
        img = cv2.resize(im0s, (512, 512), interpolation=cv2.INTER_AREA)
        str = base64.b64encode(cv2.imencode('.jpg', img)[1]).decode("utf-8")
        # img_bytes = cv2.imencode('.jpg', img)[1].tobytes()
        info.append(str)
        ori_file.append(item)
        # print(item, Path(item).stat().st_size/1024)
        # print(len(img_bytes)/1024)
        # print(len(b_str)/1024)
        # print("################################")

    if len(info) == 30:
        b_str = json.dumps({"data": info, "user": "db846572170927ce", "ori_file": ori_file}).encode("utf-8")
        sizes.append(round(len(b_str) / 1024, 2))
        info = []
        ori_file = []

    if len(sizes) > 0 and len(sizes) % 2 == 0:
        print(i, "data size=", np.average(sizes))

    i += 1
"""