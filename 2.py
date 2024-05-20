import numpy as np
import matplotlib.pyplot as plt

energy = [0.0354, 0.028, 0.0203, 0.0163, 0.0143, 0.0124, 0.0117, 0.0105]
map    = [0.2949, 0.3311, 0.3859, 0.4226, 0.4455, 0.4719, 0.4825, 0.5003]
map2    = [0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55]

energy3 = [0.0715, 0.0599, 0.0489, 0.0407, 0.0365, 0.03,   0.0264, 0.0211]
map3    = [0.4034, 0.4342, 0.4717, 0.502,  0.5171, 0.5326, 0.5372, 0.539]

energy4 = [0.0482, 0.0388, 0.0305, 0.0255,  0.0229, 0.0201, 0.0184, 0.0161]
map4    = [0.3527, 0.3919, 0.4342, 0.4659,  0.4854, 0.5059, 0.5166, 0.5272]

map_b    = [0.25, 0.25,0.25,0.25,0.25,0.25,0.25,0.25]
energy_b = [0.0321, 0.019,  0.0102, 0.006,   0.0045, 0.0033, 0.0029, 0.0022]

energy_m = [1, 1, 0.178, 0.0846,   0.0608, 0.0387, 0.0317, 0.0232]
map_m    = [1, 1, 0.5502, 0.5502,  0.5502, 0.5502, 0.5502, 0.5502]
b_m =      [1, 1, 20.45, 37.4, 51.925, 82.85, 102.125, 142.9]

e_i_8 = []
e_i_4 = []
e_i_2 = []
a_i_8 = []
a_i_4 = []
a_i_2 = []
avg_8 = []
avg_4 = []
avg_2 = []
for i in range(2, 8):
    e_r = energy_m[i] - energy_b[i]
    a_r = 0.3
    e_i_8.append((energy[i] - energy_b[i])/e_r)
    e_i_4.append((energy4[i] - energy_b[i]) / e_r)
    e_i_2.append((energy3[i] - energy_b[i]) / e_r)

    a_i_8.append((map[i] - map_b[i]) / a_r)
    a_i_4.append((map4[i] - map_b[i]) / a_r)
    a_i_2.append((map3[i] - map_b[i]) / a_r)

    avg_8.append(a_i_8[-1]-e_i_8[-1])
    avg_4.append(a_i_4[-1] - e_i_4[-1])
    avg_2.append(a_i_2[-1] - e_i_2[-1])

print("e_i_8=", e_i_8)
print("e_i_4=", e_i_4)
print("e_i_2=", e_i_2)
print("a_i_8=", a_i_8)
print("a_i_4=", a_i_4)
print("a_i_2=", a_i_2)

print("avg_8=", avg_8, np.average(avg_8))
print("avg_4=", avg_4, np.average(avg_4))
print("avg_2=", avg_2, np.average(avg_2))

plt.plot(e_i_8)
plt.plot(e_i_4)
plt.plot(e_i_2)

plt.plot(a_i_8)
plt.plot(a_i_4)
plt.plot(a_i_2)
plt.show()