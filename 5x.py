# 640 5x latency = 0.0448
"""
vol = 4.09 / [static = 4.09, encoding=4.09, network=4.09]
energy = 3.53 / [encoding=2.19, network=1.34]
time   = 2.55 / [encoding=1.2, network=0.59, inference=0.76]
map    = 0.56
gpu    = 100
"""
# 512 5x latency = 0.0322
import numpy as np

"""
vol = 4.11 / [static = 4.11, encoding=4.11, network=4.11]
energy = 2.72 / [encoding=1.84, network=0.88]
time   = 2.01 / [encoding=1.01, network=0.38, inference=0.62]
map    = 0.56
gpu    = 100
"""

# 384 5x latency = 0.0244,
"""
vol = 4.12 / [static = 4.12, encoding=4.12, network=4.12]
energy = 2.29 / [encoding=1.71, network=0.58]
time   = 1.75 / [encoding=0.94, network=0.25, inference=0.56]
map    = 0.53
gpu    = 100
"""

# 320 5x latency = 0.0208
"""
vol = 3.94 / [static = 3.94, encoding=3.94, network=3.94]
energy = 1.96 / [encoding=1.59, network=0.37]
time   = 1.54 / [encoding=0.84, network=0.16, inference=0.54]
map    = 0.51
gpu    = 100
"""

# 256 5x latency = 0.0195
"""
vol = 4.14 / [static = 4.14, encoding = 4.14, network = 4.14]
energy = 1.99 / [encoding = 1.61, network = 0.37]
time = 1.55 / [encoding = 0.87, network = 0.16, inference = 0.52]
map = 0.48
gpu = 100
"""

# 192 5x latency = 0.0176, map = 0.4, 0.42
"""
vol = 4.12 / [static = 4.12, encoding=4.12, network=4.12]
energy = 1.63 / [encoding=1.42, network=0.22]
time   = 1.2 / [encoding=0.73, network=0.09, inference=0.38]
map    = 0.43
gpu    = 100
"""

# 128 5x latency = 0.0171
"""
vol = 3.93 / [static = 3.93, encoding=3.93, network=3.93]
energy = 1.42 / [encoding=1.32, network=0.11]
time   = 1.15 / [encoding=0.67, network=0.05, inference=0.43]
map    = 0.33
gpu    = 100
"""


"""
vol = 4.15 / [static = 4.15, encoding=4.15, network=4.15]
energy = 3.35 / [encoding=2.12, network=1.23]
time   = 1.75 / [encoding=1.16, network=0.55, inference=0.04]
map    = 0.5712
gpu    = 100
network =  [0.04, 0.07, 0.12, 0.16, 0.24, 0.36, 0.55]
inference =  [0.0179, 0.0181, 0.0192, 0.0197, 0.0221, 0.0286, 0.0399]
energy =  [0.1, 0.17, 0.26, 0.37, 0.53, 0.81, 1.23]
map =  [0.329, 0.4239, 0.4792, 0.5163, 0.5305, 0.5686, 0.5712]
"""


e = [0.1, 0.17, 0.26, 0.37, 0.53, 0.81, 1.23]
t = [0.04, 0.07, 0.12, 0.16, 0.24, 0.36, 0.55]

p = []

for i in range(len(e)):
    p.append((e[i] / t[i] )/4.15)

print(round(np.average(p), 4))
