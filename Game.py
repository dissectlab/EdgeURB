import math
import random
from multiprocessing import Pool
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor
import time

class Optimize:
    def __init__(self, s, nu, d, c, p, v, m, u, tau, beta, co_map, scale):
        self.s = s
        self.nu = nu
        self.d = d
        self.c = c
        self.p = p
        self.v = v
        self.u = u
        self.m = m
        self.tau = tau
        self.beta = beta
        self.co_map = co_map

        self.scale = scale

        # print(self.c)

    def sum_w_x_y(self):
        sum_x = 0
        for j in range(len(self.s)):
            sum_x += self.w_x(self.s[j], self.nu[j], self.d[self.u[j]], self.p[j], self.v[j])
        sum_y = 0

        for j in range(len(self.s)):
            sum_y += self.w_y(self.s[j], self.nu[j], self.c[self.m[j]][self.u[j]])
        return sum_x, sum_y

    def w_x(self, s_n, nu_n, d_n, p_n, v_n):
        return math.sqrt((p_n * v_n + s_n * nu_n) * d_n)

    def w_y(self,s_n, nu_n, c_n):
        return math.sqrt(s_n * nu_n * c_n)

    def diff(self, x_n, y_n, s_n, d_n, c_n, tau):
        return s_n * (d_n / x_n + c_n / y_n) - tau

    def update_nu(self, nu_n, t, diff):
        if nu_n + t * diff > 0:
            return nu_n + t * diff
        else:
            return nu_n

    def energy(self, x_n, s_n, p_n, v_n, d_n):
        return s_n * (d_n * p_n * v_n / x_n)

    def total_time(self):
        total = 0
        for i in range(len(self.s)):
            total += self.s[i] * (self.d[self.u[i]] + self.c[self.m[i]][self.u[i]])
        print(total)

    def do(self):
        sum_w_x, sum_w_y = self.sum_w_x_y()

        x = [self.w_x(self.s[i], self.nu[i], self.d[self.u[i]], self.p[i], self.v[i]) / sum_w_x for i in range(len(self.s))]
        y = [self.w_y(self.s[i], self.nu[i], self.c[self.m[i]][self.u[i]]) / sum_w_y for i in range(len(self.s))]

        t = 1

        es = []

        energy_ = 0
        ee = []
        for i in range(len(self.s)):
            energy_ += self.energy(1/len(self.s), self.s[i], self.p[i], self.v[i], self.d[self.u[i]])

        # print("uniform network allocation, energy={}".format(round(energy_, 4)))
        pre_diffs = []
        for i in range(len(self.s)):
            diff_ = self.diff(x[i], y[i], self.s[i], self.d[self.u[i]], self.c[self.m[i]][self.u[i]], self.tau)
            pre_diffs.append(diff_)

        start = time.time()
        while True:
            diffs = []
            energy_ = []
            for i in range(len(self.s)):
                diff_ = self.diff(x[i], y[i], self.s[i], self.d[self.u[i]], self.c[self.m[i]][self.u[i]], self.tau)
                diffs.append(diff_)
                self.nu[i] = self.update_nu(self.nu[i], 0.01 / math.sqrt(t), diffs[i])
                energy_.append(self.energy(x[i], self.s[i], self.p[i], self.v[i], self.d[self.u[i]]))

            # if t == 1:
                # print("diffs=", np.round(np.array(diffs), 4))

            t += 1
            sum_w_x, sum_w_y = self.sum_w_x_y()
            x = [self.w_x(self.s[i], self.nu[i], self.d[self.u[i]], self.p[i], self.v[i]) / sum_w_x for i in
                 range(len(self.s))]
            y = [self.w_y(self.s[i], self.nu[i], self.c[self.m[i]][self.u[i]]) / sum_w_y for i in range(len(self.s))]
            # print("#####################################################################")

            exits = 0
            diffs = []
            network = []
            compute = []
            for i in range(len(self.s)):
                diff_ = self.diff(x[i], y[i], self.s[i], self.d[self.u[i]], self.c[self.m[i]][self.u[i]], self.tau)
                diffs.append(diff_)
                network.append(round(self.s[i] * (self.d[self.u[i]] / x[i]), 4))
                compute.append(round(self.s[i] * self.c[self.m[i]][self.u[i]] / y[i], 4))
                if np.abs(diffs[i] - pre_diffs[i]) <= 0.0001 and diff_ < 0:
                    exits += 1
            pre_diffs = diffs

            if exits == len(self.s) or t > 20000:
                # print("it={},time={}".format(t, round(time.time() - start, 4)))
                #print("diffs=", np.round(np.array(diffs), 4), np.round(np.average(diffs), 5))
                #print("x=", list(np.round(np.array(x), 2)))
                #print("y=", list(np.round(np.array(y), 2)))
                #print("network={}, compute={}".format(round(np.average(network), 4), round(np.average(compute), 4)))
                #print("u=", u)
                #print("m=", m)
                r = np.array([128, 192, 256, 320, 384, 512, 640])

                cost = []
                for i in range(len(self.s)):
                    cost.append(
                        self.s[i] * self.p[i] * self.v[i] * self.d[self.u[i]] / x[i] - self.s[i] * self.beta * self.co_map[self.m[i]][self.u[i]])
                #print("cost = ", np.round(np.array(cost), 4), np.round(np.average(cost), 4))
                is_suc = True
                for item in diffs:
                    if item > 0:
                        is_suc = False
                        break
                return t, list(x), y, cost, np.round(np.average(cost), 7), diffs, is_suc, network, compute


def worker(items):
    i, s, nu, d, c, p, v, m, u, tau, beta, co_map = items
    mi = m.copy()
    mi[i] = max(0, mi[i] - 1)
    opt = Optimize(s, nu, d, c, p, v, mi, u, tau, beta, co_map, 1)
    _, _, _, _, cost_m, _, suc_m, _, _ = opt.do()

    if suc_m is False:
        cost_m = 0

    ui = u.copy()
    ui[i] = min(6, ui[i] + 1)
    opt = Optimize(s, nu, d, c, p, v, m, ui, tau, beta, co_map, 1)
    _, _, _, _, cost_u, _, suc_u, _, _ = opt.do()

    if suc_u is False:
        cost_u = 0

    if cost_m < cost_u:
        return [cost_m, "model"]
    else:
        return [cost_u, "resolution"]


def request(s, d, c, p, v, m, u, tau, beta, co_map, r, scale):
    cost_ = []
    dd = None
    do = "model"

    # print("#############################")
    while True:

        x, y, stop, diffs, avg_cost, costs = do_opt(s, d, c, p, v, m, u, tau, beta, co_map, scale)
        if stop:
            break

        cost_ = costs
        dd = diffs

        opt_u = []
        tcost = 0
        changed = False
        for si in range(len(s)):
            cost = []
            conb = []
            for index_u, resolution in enumerate(r):
                for index_m in range(4):
                    conb.append([index_u, index_m])
                    if s[si] * (d[index_u] / x[si] + c[index_m][index_u] / y[si]) <= tau:
                        cost.append(
                            s[si] * p[si] * v[si] * d[index_u] / x[si] - s[si] * beta * co_map[index_m][index_u])
                    else:
                        cost.append(999)
            opt_u.append(conb[cost.index(min(cost))])
            tcost += min(cost)
            if opt_u[si][0] != u[si]:
                changed = True
            u[si] = opt_u[si][0]
            if opt_u[si][1] != m[si]:
                changed = True
            m[si] = opt_u[si][1]

        if not changed:
            break

    while False:
        do = ""
        target = -1
        max_ = 1

        items = []
        for i in range(len(s)):
            items.append((i, s, [1./len(s) for ii in range(len(s))], d, c, p, v, m, u, tau, beta, co_map))

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

        print("u=", u)
        print("m=", m)
        print("cost=", min(costs))
        print("########################")


    while True:

        x, y, stop, diffs, avg_cost, costs = do_opt(s, d, c, p, v, m, u, tau, beta, co_map, scale)
        if stop:
            break

        cost_ = costs
        dd = diffs

        target = -1
        max_ = 1
        if do == "model":
            for i in range(len(s)):
                if diffs[i] < max_ and m[i] != 0:
                    max_ = diffs[i]
                    target = i
            if target != -1:
                m[target] = max(0, m[target] - 1)
            else:
                do = "resolution"

        if do == "resolution":
            for i in range(len(s)):
                if diffs[i] < max_ and u[i] != 4:
                    max_ = diffs[i]
                    target = i
            if target != -1:
                u[target] = min(4, u[target] + 1)
            else:
                break

    return cost_, dd, m, u


def do_opt(s, d, c, p, v, m, u, tau, beta, co_map, scale):
    nu = [1./len(s) for ii in range(len(s))]
    opt = Optimize(s, nu, d, c, p, v, m, u, tau, beta, co_map, scale)
    t, x, y, cost, avg_cost, diffs, _, network, compute = opt.do()
    stop = False
    for i in range(len(s)):
        if diffs[i] > 0:
            stop = True
    if not stop:
        pass
        #print("time=", t, "cost=", round(avg_cost, 4), np.round(np.array(cost), 4))
        # print("network={}, compute={}".format(round(np.average(network), 4), round(np.average(compute), 4)))
        #print("diff=", np.round(np.array(diffs), 4), round(np.average(diffs), 4))
        #print("u=", u)
        #print("m=", m)
       # print("x=", list(np.round(np.array(x), 4)))
       # print("y=", list(np.round(np.array(y), 4)))
       # print("#################################")
    return x, y, stop, diffs, avg_cost, cost


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

    number = 2
    edge = 1

    s = [random.randint(3, 10) for i in range(number)]
    p = [round(random.uniform(0.3, 0.6), 2) for i in range(number)]
    v = [round(random.uniform(3.5, 4.5), 2) for i in range(number)]
    o = [random.randint(0, 2) for i in range(number)]
    scale = [random.uniform(1, 1) for i in range(edge)]

    s = [10, 20]
    p = [0.553, 0.553]
    v = [4.2643, 4.2571]

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

    request(s, d, c, p, v, m, u, tau, beta, co_map, r, 1)

########
"""
Device = c8f6d9ad8d1cdb13  
FPS = 20
current= 553.6386 vol= 4.2571
network =  [0.0605, 0.1037, 0.1801, 0.2602, 0.3745, 0.575, 0.8432]
rate = 37.15 Mbps

Device = 9fd3b96565dbebe8
FPS = 10
current= 555.2829 vol= 4.2643
network =  [0.0408, 0.0619, 0.1016, 0.1297, 0.1719, 0.2774, 0.3932]
rate = 35.56 Mbps
"""