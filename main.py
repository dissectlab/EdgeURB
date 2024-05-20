import json
import random
import math
import numpy as np

from fix_x import Optimize, dist

if __name__ == '__main__':

    tau = 1
    number = 56
    edge = 8
    f = open(f'setting_{edge}_{number}.json')
    data = json.load(f)
    b = data["b"]
    snr = data["snr"]
    beta = 0.5
    for i in range(1):
        for k in range(edge):
            pass

    s = []
    for i in range(50):
        s.append(np.average([data["b"][i][k] for k in range(edge)]))
    print(np.average(s))

    dist(data, number, edge, tau, beta, start=1, end=2, alg="snr", w=8)

    co_map = data["co_map"]
    m = data["m"]
    s = data["s"]
    hm = []
    for i in range(0, 1):
        mAp = 0
        for j in range(number):
            mAp += co_map[m[i][j]][0] - co_map[m[i][j]][1] / (float(128) ** co_map[m[i][j]][2])
        hm.append(mAp/number)
    print(np.average(hm))

    print("x=", data["x"][1])
    print("y=", data["y"][1])

    """
# w = 8
6 success 26 cost 2.4658 resolution 212
energy= 0.0193 map= 0.3931
cost= [-2.438, -2.0538, -1.934, -2.4937, -2.3024, -2.2316, -2.3035, -3.0488, -2.8246, -2.1846, -2.1617, -2.5089, -2.4293, -2.3652, -2.9091, -2.4178, -2.4105, -2.5652, -2.3645, -2.6414, -3.0724, -2.9449, -2.0662, -2.6835, -3.0407, -2.0462, -2.4606, -2.6986, -2.0086, -2.3635]
84 success 30 cost 2.6428 resolution 215
energy= 0.0183 map= 0.4004
# w = 4
8 success 27 cost 3.2617 resolution 259
energy= 0.0263 map= 0.4252
cost= [-3.3817, -2.9678, -2.7433, -3.2705, -3.269, -2.857, -3.1557, -3.7115, -3.5814, -3.0727, -3.0732, -3.2652, -3.002, -3.3381, -3.8161, -3.109, -3.1136, -3.3438, -3.1723, -3.3901, -3.9646, -3.8738, -2.8228, -3.4691, -3.7743, -2.8215, -3.1243, -3.4573, -2.8826, -3.0271]
73 success 30 cost 3.3876 resolution 265
energy= 0.0258 map= 0.434
cost= [-3.4114, -3.1085, -3.0823, -3.2047, -3.4853, -2.8594, -3.6591, -3.687, -3.6, -3.2818, -3.3329, -3.3645, -3.4847, -3.5489, -4.0166, -3.1987, -3.2066, -3.3886, -3.3764, -3.5437, -3.9308, -3.9951, -2.9226, -3.4647, -3.7762, -2.8898, -3.2076, -3.4547, -3.1102, -3.0353]
# w = 12
98 success 30 cost 2.0479 resolution 189
energy= 0.0153 map= 0.3765
cost= [-1.8087, -1.4846, -1.3378, -1.8553, -2.0982, -1.884, -2.2659, -2.5232, -2.2511, -1.7959, -1.981, -2.097, -2.2559, -2.0446, -2.5931, -2.0177, -2.0329, -2.1272, -2.1467, -2.3566, -2.3424, -2.5113, -1.6291, -2.0931, -2.5387, -1.596, -2.0687, -2.0919, -1.7868, -1.8205]
6 success 27 cost 1.8398 resolution 188
energy= 0.0163 map= 0.3709
cost= [-1.7036, -1.3187, -1.2585, -1.8891, -1.5466, -1.7512, -1.5441, -2.5344, -2.2293, -1.4927, -1.4298, -1.9382, -1.9436, -1.6104, -2.2298, -1.8788, -1.8563, -1.9708, -1.7332, -2.0835, -2.38, -2.2435, -1.4676, -2.0486, -2.4752, -1.423, -1.9306, -2.1169, -1.317, -1.849]

..........................................
9 success 20 cost 2.1405 resolution 195
energy= 0.2031 map= 3.765
cost= [-2.0171, -1.7527, -1.7581, -2.1912, -2.2518, -1.6673, -2.2744, -1.208, -2.3404, -2.4512, -2.0776, -2.2199, -2.6118, -2.1047, -1.8635, -2.0011, -2.4864, -2.2197, -2.1419, -1.668, -2.2753, -2.1685, -2.3196, -1.7073, -2.6884, -2.7051, -2.6211, -2.2668, -2.098, -2.0583]

110 success 30 cost 2.3222 resolution 201
energy= 0.1968 map= 3.8965
cost= [-2.1669, -2.3614, -2.2013, -2.1907, -2.2469, -1.7689, -2.2748, -1.9818, -2.6138, -2.8245, -2.1818, -2.3767, -2.6786, -2.5071, -2.0561, -2.132, -2.6693, -2.5954, -2.2534, -1.7465, -2.5237, -2.0178, -2.4202, -2.0534, -2.7319, -2.6841, -2.692, -2.3046, -2.2961, -2.114]

.........................................
14 success 16 cost 1.8304 resolution 179
energy= 0.2198 map= 3.589
cost= [-1.1435, -1.7739, -1.7775, -1.769, -1.9006, -1.4394, -1.789, -2.1658, -1.9251, -2.2274, -2.0063, -1.8401, -1.8281, -2.0205, -2.4615, -1.5366, -1.6781, -1.73, -2.4716, -2.2495, -2.1067, -1.431, -1.7098, -1.9214, -1.7291, -1.2528, -2.095, -1.9456, -1.4619, -1.5263]

138 success 27 cost 2.0098 resolution 186
energy= 0.2181 map= 3.7545
cost= [-1.6778, -1.9298, -1.7979, -1.8463, -1.9637, -1.475, -1.8143, -2.3639, -2.0493, -2.3415, -2.3623, -2.0751, -2.2873, -2.1318, -2.4459, -1.9062, -1.8343, -1.7402, -2.5401, -2.4887, -2.2866, -1.5643, -1.9826, -2.0391, -2.0866, -1.4006, -2.0601, -2.2469, -1.856, -1.699]   
   
.........................................
16 success 15 cost 1.641 resolution 174
energy= 0.025 map= 0.3545
cost= [-1.6471, -1.7292, -1.2135, -1.3518, -1.5597, -1.8598, -1.7387, -1.6359, -1.6805, -1.7958, -1.9121, -1.8808, -1.6409, -1.7995, -1.0403, -1.9125, -1.6488, -1.7037, -1.8035, -1.6523, -1.6657, -2.1566, -1.811, -1.7071, -1.4463, -1.5846, -1.5309, -1.7165, -1.3423, -1.0613]

159 success 25 cost 1.7723 resolution 179
energy= 0.2365 map= 3.6644  
cost= [-1.6724, -1.938, -1.45, -1.8331, -1.7363, -2.0561, -1.7488, -1.6544, -1.8893, -1.8389, -2.0095, -1.9619, -1.7948, -1.923, -1.4599, -1.9854, -1.6634, -1.6916, -1.7916, -1.5597, -1.6926, -2.2545, -1.8735, -1.6528, -1.4548, -1.7813, -1.8787, -1.7843, -1.8355, -1.3032] 
    """

    # 10 success 93 cost 1.3714 resolution 228
    # energy= 0.2778 map= 4.1319
    # 15 success 74 cost 1.1857 resolution 211
    # energy= 0.3122 map= 3.9326
    # 45 success 47 cost 1.0913 resolution 202
    # energy= 0.3379 map= 3.8723
    # 74 success 22 cost 1.0218 resolution 193
    # energy= 0.3613 map= 3.85

    # 9 success 100 cost 1.283 resolution 215
    # energy= 0.2984 map= 4.0578
    # 12 success 90 cost 1.0569 resolution 197
    # energy= 0.3436 map= 3.8317
    # 18 success 50 cost 0.9562 resolution 190
    # energy= 0.3742 map= 3.7835
    # 39 success 29 cost 0.8356 resolution 181
    # energy= 0.412 map= 3.7313


    """
    cost = []
    hist_u = []
    success = 0
    total = 0
    for i in range(80, 100):
        print("#################################")
        print("s=", s[i])
        scale = data["scale"][i]
        o = [[1. / edge for j in range(edge)] for i in range(number)]
        for j in range(number):
            for k in range(edge):
                snr[i][j][k] = math.log2(1 + math.pow(10, snr[i][j][k]/20))
        u = [0 for xx in range(number)]
        opt = Optimize(o, b[i], snr[i], s[i], d, c, p[i], v[i], m[i], u, tau, beta, co_map)
        cost_avg, success_, res_avg = opt.do_equal(number_edge=edge, edge_scale=scale)
        cost.append(cost_avg)
        hist_u.append(res_avg)
        success += success_
        total += 1
        print("cost=", cost)
        print("cost", np.average(cost), "success", success, "resolution", np.average(hist_u), total)
    """

# r = [128, 192, 256, 320, 384, 512, 640]

# 4 edge + 28 users
# success 38 cost -6.353 resolution 1.115      Equal Load Server Association (ELSA)
# 233 success 82 cost -6.8195 resolution 1.354  OPT
# 322 success 81 cost -6.8347 resolution 1.365  OPT - fix x
#     success 34 cost -6.6154 resolution 1.337  MaxSNR
#     success 33 cost -6.3214 resolution 1.089      Equal Load Server Association (ELSA) - fix X


# 4 edge + 24 users
# success 67 cost -6.8837 resolution 1.666      Equal Load Server Association (ELSA)
# 233 success 99 cost -7.5569 resolution 2.120  OPT
#     success 56 cost -7.1209 resolution 1.902  MaxSNR
#     success 55 cost -6.9007 resolution 1.71      Equal Load Server Association (ELSA) - fix X


# 4 edge + 20 users
# success 91 cost -7.3959 resolution 2.258      Equal Load Server Association (ELSA)
# 233 success 100 cost -7.9224 resolution 2.649  OPT
#     success 71 cost -7.4478 resolution 2.325  MaxSNR
#     success 91 cost -7.4120 resolution 2.255      Equal Load Server Association (ELSA) - fix X


# 4 edge + 16 users
#     success 100 cost -7.9646 resolution 2.873      Equal Load Server Association (ELSA)
# 188 success 100 cost -8.3185 resolution 3.253  OPT
#     success 91 cost -7.9178 resolution 2.878  MaxSNR
#     success 99 cost -7.9115 resolution 2.914      Equal Load Server Association (ELSA) - fix X
