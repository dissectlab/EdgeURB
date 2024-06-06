import json
import math

import numpy as np
import matplotlib.pyplot as plt

def get_d(d, u):
    return d[0] * (u ** 2) + d[1]


def get_c(c, m, u):
    return c[m][0] * (u ** 3) + c[m][1]


def get_i(co_map, m, u):
    return co_map[m][0] - co_map[m][1] / u


f = open(f'setting_{1}_{6}.json')
data = json.load(f)
i = 0
d = data["d"]
s = data["s"]
p = data["p"]
v = data["v"]
c = data["c"]
m = data["m"]
b = data["b"][i]
scale = data["scale"][i]
snr = data["snr"]
co_map = data["co_map"]

snr = math.log2(1 + math.pow(10, 25/20))
rho = 10.
nv = 1.
y = 0.01
x = 0.01
mu = 1.
sn = 10.
l = 1.
X = 20.
hist_val_y = []
hist_val_x = []
hist_y = []
hist_x = []

for i in range(100):
    while True:
        val_y = nv - mu * sn * get_c(c, 0, 128) / (y ** 2) + rho * (y - 1)
        hist_val_y.append(val_y)
        hist_y.append(y)
        if val_y >= -0.001:
            #print(val_y, y)
            break
        y += 0.001

    while True:
        val_x = l + rho * (x - X) - sn * (0.5 * 4.2 + mu) * get_d(d, 128) / ((x ** 2) * snr)
        hist_val_x.append(val_x)
        hist_x.append(x)
        if val_x >= -0.001:
            #print(val_x, x)
            break
        x += 0.001

    print(x)

    l = l + rho * (x - X)
    nv = nv + rho * (y - 1)

    diff = sn * (get_d(d, 128)/(x * snr) + get_c(c, 0, 128)/y) - 1
    mu = mu + diff * 0.0001

    print(sn * get_d(d, 128) * 0.5 * 4.2 / (x * snr), f"l={l},  nv={nv}, mu={mu}, diff={diff}")


