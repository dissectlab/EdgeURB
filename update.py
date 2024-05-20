import math
import random
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor
import time
from result.Game import request


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

"""
def worker(items):
    i, s, nu, d, c, p, v, m, u, tau, beta, co_map = items
    mi = m.copy()
    mi[i] = max(0, mi[i] - 1)
    opt = Optimize(s, nu, d, c, p, v, mi, u, tau, beta, co_map)
    _, _, _, _, cost_m, _, suc_m, _, _ = opt.do()

    if suc_m is False:
        cost_m = 0

    ui = u.copy()
    ui[i] = min(6, ui[i] + 1)
    opt = Optimize(s, nu, d, c, p, v, m, ui, tau, beta, co_map)
    _, _, _, _, cost_u, _, suc_u, _, _ = opt.do()

    if suc_u is False:
        cost_u = 0

    if cost_m < cost_u:
        return [cost_m, "model"]
    else:
        return [cost_u, "resolution"]
"""

if __name__ == '__main__':
    d = np.array([0.05, 0.08, 0.12, 0.17, 0.23, 0.37, 0.55]) / 12

    co_map = [
        [0.6275, 37.9476],
        [0.6110, 38.8654],
        [0.5875, 39.8489],
        [0.5045, 34.0794]
    ]

    c = [
        [0.0179, 0.0181, 0.0192, 0.0197, 0.0221, 0.0286, 0.0399],
        [0.0148, 0.0143, 0.0146, 0.0148, 0.0157, 0.0192, 0.0242],
        [0.0108, 0.0109, 0.0114, 0.0118, 0.013, 0.0144, 0.0184],
        [0.0087, 0.0089, 0.0095, 0.0096, 0.0101, 0.0112, 0.0127]
    ]

    """
    i = [
        [0.329, 0.4239, 0.4792, 0.5163, 0.5305, 0.5686, 0.5712],
        [0.3082, 0.4022, 0.4629, 0.4946, 0.5219, 0.5518, 0.5571],
        [0.2786, 0.3668, 0.4271, 0.4653, 0.486, 0.5113, 0.514],
        [0.2381, 0.3234, 0.3757, 0.4105, 0.4293, 0.446, 0.441]
    ]
    """

    co_map = [
        [0.33, 0.43, 0.48, 0.51, 0.53, 0.56, 0.56],
        [0.31, 0.4, 0.46, 0.49, 0.52, 0.54, 0.54],
        [0.28, 0.37, 0.43, 0.47, 0.49, 0.51, 0.52],
        [0.2226, 0.3092, 0.3603, 0.3946, 0.4127, 0.4265, 0.4259]
    ]

    tau = 1

    number = 9
    edge = 3

    s = [random.randint(3, 10) for i in range(number)]
    p = [round(random.uniform(0.3, 0.6), 2) for i in range(number)]
    v = [round(random.uniform(3.5, 4.5), 2) for i in range(number)]
    o = [random.randint(0, 2) for i in range(number)]
    scale = [random.uniform(1, 1) for i in range(edge)]

    print("s=", s)
    print("p=", p)
    print("v=", v)
    print("o=", o)
    print("scale=", scale)
    print("power=", [p[i] * v[i] for i in range(len(s))])

    m = [3 for xx in range(len(s))]
    u = [0 for xx in range(len(s))]

    r = np.array([128, 192, 256, 320, 384, 512, 640])
    beta = 1

    do = "model"

    # x = [1./len(s) for xx in range(len(s))]
    # y = [1. / len(s) for xx in range(len(s))]
    hist = []
    it = 0
    while True:
        for i in range(len(s)):
            # find best server for given server selection from others
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            pre_server = o[i]
            cost = []
            for server in range(edge):
                o[i] = server
                this_s = [s[i]]
                this_p = [p[i]]
                this_v = [v[i]]
                this_m = [m[i]]
                this_u = [u[i]]
                for j in range(len(s)):
                    if o[j] == server and j != i:
                        this_s.append(s[j])
                        this_p.append(p[j])
                        this_v.append(v[j])
                        this_m.append(m[j])
                        this_u.append(u[j])
                if len(this_s) == 0:
                    continue
                costs, diff, _, _ = request(this_s, d, c, this_p, this_v, this_m, this_u, tau, beta, co_map, r, scale[server])
                fail = False
                if diff is not None:
                    for item in diff:
                        if item > 0:
                            fail = True
                            break
                if not fail and len(costs) > 0:
                    cost.append(round(costs[0], 2))
                else:
                    cost.append(0)

            opt_server = cost.index(min(cost))
            o[i] = opt_server
            print("server cost list", cost)
            print("at time {}, user {} select server {}.".format(it, i, opt_server), o)
            cost = []
            fail = 0
            diffs = []
            for server in range(edge):
                this_s = []
                this_p = []
                this_v = []
                this_m = []
                this_u = []
                for j in range(len(s)):
                    if o[j] == server:
                        this_s.append(s[j])
                        this_p.append(p[j])
                        this_v.append(v[j])
                        this_m.append(m[j])
                        this_u.append(u[j])
                if len(this_s) == 0:
                    continue
                costs, diff, mm, uu = request(this_s, d, c, this_p, this_v, this_m, this_u, tau, beta, co_map, r, scale[server])
                if diff is not None:
                    for item in diff:
                        if item > 0:
                            fail = True
                            break
                if not fail and len(costs) > 0:
                    cost.append(round(costs[0], 2))
                    diffs += diff
                else:
                    cost.append(0)
            print("m=", mm)
            print("u=", uu)
            print("current cost=", np.round(np.array(cost), 4), f"{bcolors.OKGREEN}{round(np.average(cost), 4)}{bcolors.ENDC}")
            if diff is not None:
                print("fail={}".format(fail), np.round(np.array(diffs), 4))
            hist.append(round(np.average(cost), 4))
            print(hist)
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        plt.plot(hist)
        plt.show()
        it += 1





"""
min_diff = 1

        if do == "model":
            m[target] = max(0, m[target] - 1)
        else:
            u[target] = min(6, u[target] + 1)
"""

"""
s = [5, 7, 10, 4, 10, 4]
p = [0.41, 0.44, 0.49, 0.44, 0.53, 0.55]
v = [3.88, 4.36, 3.58, 3.89, 3.79, 3.68]
    
-2.11 model then resolution  
    u= [2, 1, 0, 3, 0, 0]
    m= [0, 0, 0, 0, 0, 0]
-2.3489392
u= [1, 1, 1, 2, 1, 1]
m= [0, 0, 1, 0, 0, 0]

-2.2462664 max-first
    u= [1, 1, 2, 1, 2, 0]
    m= [2, 0, 0, 3, 0, 3]
"""

"""
do = ""
target = -1
max_ = 1

items = []
for i in range(len(s)):
    items.append((i, s, nu, d, c, p, v, m, u, tau, beta, co_map))

with ThreadPoolExecutor(max_workers=len(items)) as executor:
    results = executor.map(worker, items)

costs = []
res = []
for item in results:
    costs.append(item[0])
    res.append(item[1])

target = costs.index(min(costs))

if res[target] == "resolution":
    u[target] = min(6, u[target] + 1)
else:
    m[target] = max(0, m[target] - 1)

continue
"""
# break
"""
for i in range(len(s)):
    mi = m.copy()
    mi[i] = max(0, mi[i] - 1)
    opt = Optimize(s, nu, d, c, p, v, mi, u, tau, beta, co_map)
    _, _, _, _, cost_m, _, suc_m, _, _ = opt.do()

    if suc_m is False:
        cost_m = 0

    if cost_m < max_:
        max_ = cost_m
        target = i
        do = "model"

    ui = u.copy()
    ui[i] = min(6, ui[i] + 1)
    opt = Optimize(s, nu, d, c, p, v, m, ui, tau, beta, co_map)
    _, _, _, _, cost_u, _, suc_u, _, _ = opt.do()

    if suc_u is False:
        cost_u = 0

    if cost_u < max_:
        max_ = cost_u
        target = i
        do = "resolution"
"""
