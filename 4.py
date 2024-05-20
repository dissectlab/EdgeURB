import random

import matplotlib.pyplot as plt
import numpy as np

energy=[0.1083, 0.0432, 0.0285, 0.0218, 0.0191, 0.0166, 0.0138, 0.012, 0.0111, 0.0106, 0.0107]
map=[0.519, 0.4558, 0.4182, 0.386, 0.3593, 0.345, 0.3134, 0.2833, 0.2537, 0.2537, 0.2537]

energyb = [0.0108, 0.0108, 0.0108, 0.0108, 0.0108, 0.0108, 0.0108, 0.0108, 0.0108, 0.0108, 0.0108]
energyh = [0.1574, 0.1574, 0.1574, 0.1574, 0.1574, 0.1574, 0.1574, 0.1574, 0.1574, 0.1574, 0.1574]

mapb= [0.2537, 0.2537, 0.2537, 0.2537, 0.2537, 0.2537, 0.2537, 0.2537, 0.2537, 0.2537, 0.2537]
maph= [0.5465, 0.5465, 0.5465, 0.5465, 0.5465, 0.5465, 0.5465, 0.5465, 0.5465, 0.5465, 0.5465]

# energy= 0.0184 cost_map= 0.3602
# 0.0187 0.364

e = energyh[0] - energyb[0]
m = maph[0] - mapb[0]

im_a = []
im_e = []
for i in [0, 1, 2, 4]:
    im_a.append((map[i] - mapb[i])/m)
    im_e.append((energy[i] - energyb[i]) / e)

print(im_a)
print(im_e)

import json

number = 28
edge = 4
f = open(f'setting_{edge}_{number}.json')
data = json.load(f)
b = data["b"]
snr = data["snr"]
for i in range(100):
    for k in range(edge):
        data["b"][i][k] = random.choice([75, 80, 85, 90])

with open('setting_4_28_82_5.json', 'w') as f:
   json.dump(data, f, indent=4)
