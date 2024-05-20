import math

import numpy as np
import matplotlib.pyplot as plt

# 5
energy = [0.3368, 0.3368, 0.3450, 0.3920, 0.4580, 0.5341, 0.6320, 0.7771, 0.9996]
mAp    = [2.9215, 2.9215, 3.0013, 3.3189, 3.6244, 3.8580, 4.0701, 4.2716, 4.4431]

# 7.8
energy = [0.3250, 0.3250, 0.3254, 0.3374, 0.3781, 0.4363, 0.5204, 0.6477, 0.8623]
mAp    = [2.8734, 2.8734, 2.8779, 2.9904, 3.2608, 3.5369, 3.8070, 4.0700, 4.3125]
utility = [-1.994, -1.4537, -0.9133, -0.3832, 0.1554, 0.7606, 1.4472, 2.2455, 3.2086]

for i in range(len(energy)):
    utility.append(round(0.1 * (i+1) * mAp[i] - energy[i] * (1 - 0.1 * (i+1)) * 7.8, 4))

print(utility)


with open('log_5') as file:
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]

i = 0
IDs = []
cost = []
res = []
success = 0
fail = []
total = 0
while i < len(lines):
    if lines[i] == "################":
        total += 1
        vals = lines[i+1].split(" ")
        # print(vals)
        IDs.append(int(vals[1]))
        cost.append(-float(vals[2]))
        success += int(vals[3])
        res.append(float(vals[4]))
        if int(vals[3]) == 0:
            fail.append(int(vals[1]))
        i += 2
    i += 1

not_testId = []
for i in range(100):
    if i not in IDs:
        not_testId.append(i)

print("fail", fail)
print("total", total)
print("success", success, "cost", np.average(cost), "resolution", np.average(res))

"""
not test [3, 8, 9, 23, 33, 47, 48, 49, 56, 57, 63, 64, 65, 69, 77, 80, 81, 89, 92, 93, 94, 95]
total 78
success 74 cost 1.1405999999999996 resolution 201.12820512820514

not test [3, 8, 23, 33, 47, 48, 56, 63, 64, 69, 77, 80, 89, 92, 94]
total 7
success 7 cost 1.1999571428571427 resolution 207.42857142857142

85.44436764717102 success 14 cost 0.9234533333333332 resolution 186.8      #[33]

"""

"""
p = 0.5
v = 4
x = 1
nu = 2.4046394793724452e-05
snr = math.log2(1 + math.pow(10, 30/20))
y = 0.3
beta = 1
thta_d = 0.000004037153221
thta_c = 0.000000084989901
thta_i = 37.94761821295738

s = []
uu = []
for u in range(32, 640, 1):
    part1 = 2 * thta_d * u * ((p * v) + nu) / (x * snr)
    part2 = nu * 3 * thta_c * (u ** 2) / y
    part3 = beta * thta_i / (u ** 2)
    s.append(part1 + part2 - part3)
    uu.append(u)
    #print(s[-1], u)
    if s[-1] >= 0 :
        print(u)
        break

plt.plot(uu, s)
plt.show()

"""
