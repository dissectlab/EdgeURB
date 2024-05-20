import math
import random
import copy
from inspect import trace
from multiprocessing import Pool
import numpy as np
from concurrent.futures.thread import ThreadPoolExecutor
import time
#from solver import solver

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


class Optimize:
    def __init__(self, o, b, snr, s,  d, c, p, v, m, u, tau, co_map):
        self.s = s
        self.tau = 1
        self.tha = [0.2 for ii in range(len(s))]
        # self.tha = None
        self.nu = [1. / len(s) for ii in range(len(s))]
        self.b = b
        self.snr = snr
        self.d = d
        self.c = c
        self.p = p
        self.v = v
        self.u = u
        self.m = m
        self.tau = tau
        self.co_map = co_map
        self.o = o
        self.lamada = None
        self.w = 8
        self.rho = 4

    def sum_w_x_y(self, k, edge_scale):
        sum_x = 0
        for j in range(len(self.s)):
            sum_x += math.sqrt(self.o[j][k] * self.s[j] * (self.w * self.p[j] * self.v[j] + self.nu[j]) * self.get_d(self.u[j]) / self.snr[j][k])
            # sum_x += self.o[j][k]

        sum_y = 0
        for j in range(len(self.s)):
            sum_y += self.w_y(self.o[j][k], self.s[j], self.nu[j], edge_scale[k] * self.get_c(self.m[j], self.u[j]))
        return sum_x, sum_y

    def w_y(self, o_n, s_n, nu_n, c_n):
        return math.sqrt(o_n * s_n * nu_n * c_n)

    def latency(self, o_n, snr_n, x_n, y_n, s_n, d_n, c_n):
        return o_n * s_n * (d_n / (x_n * snr_n) + c_n / y_n)

    def update_nu(self, nu_n, t, diff):
        if nu_n + t * diff > 0:
            return nu_n + t * diff
        else:
            return nu_n

    def compute_x_y(self, edge_scale, number_edge):
        x = []
        y = []

        # items = []
        for k in range(number_edge):
            sum_w_x, sum_w_y = self.sum_w_x_y(k, edge_scale) # , sum_w_y
            if sum_w_x != 0:

                x.append([math.sqrt(
                    self.o[j][k] * self.s[j] * (self.w * self.p[j] * self.v[j] + self.nu[j]) * self.get_d(
                        self.u[j]) / self.snr[j][k]) * self.b[k] / sum_w_x for j in range(len(self.s))])
                """
                x.append([1. * self.b[k] / len(self.s) for j in range(len(self.s))])
                """
            else:
                x.append([0 for j in range(len(self.s))])

            if sum_w_y != 0:

                y.append(
                    [math.sqrt(
                        self.o[i][k] * self.s[i] * self.nu[i] * edge_scale[k] * self.get_c(self.m[i],
                                                                                           self.u[i])) / sum_w_y for
                     i in range(len(self.s))])
                """
                y.append([1. / len(self.s) for j in range(len(self.s))])
                """
            else:
                y.append([0 for j in range(len(self.s))])
        return x, y

    def get_d(self, u):
        return self.d[0] * (u ** 2) + self.d[1]

    def get_c(self, m, u):
        return self.c[m][0] * (u ** 3) + self.c[m][1]

    def get_i(self, m, u):
        return self.co_map[m][0] - self.co_map[m][1] / (float(u)**self.co_map[m][2])

    def compute_x_y_edge(self, items):
        edge_scale, k = items
        sum_w_x, sum_w_y = self.sum_w_x_y(k, edge_scale)  # , sum_w_y
        x = [math.sqrt(
            self.o[j][k] * self.s[j] * (self.w * self.p[j] * self.v[j] + self.nu[j]) * self.get_d(self.u[j]) / self.snr[j][k]) *
                  self.b[k] / sum_w_x for j in
                  range(len(self.s))]
        # x.append([1. * self.b[k]/sum_w_x for j in range(len(self.s))])
        y = [math.sqrt(
                self.o[i][k] * self.s[i] * self.nu[i] * edge_scale[k] * self.get_c(self.m[i], self.u[i])) / sum_w_y for
             i in range(len(self.s))]
        return x, y

    def do_resource(self, selected, edge_scale, number_edge=3, do_server=False, do_res=True):
        self.nu = [1. for ii in range(len(self.s))]
        self.tha = [1. for ii in range(len(self.s))]
        x, y = self.compute_x_y(edge_scale, number_edge=number_edge)
        self.rho = 22

        for i in range(len(self.s)):
            if selected[i] is True:
                continue
            th = []
            for k in range(number_edge):
                if x[k][i] != 0 and y[k][i] != 0:
                    I = self.s[i] * self.w * self.p[i] * self.v[i] * self.get_d(self.u[i]) / (
                            x[k][i] * self.snr[i][k]) - self.s[i] * self.get_i(self.m[i], self.u[i])
                    L = self.nu[i] * (self.latency(1, self.snr[i][k], x[k][i], y[k][i], self.s[i],
                                                   self.get_d(self.u[i]),
                                                   edge_scale[k] * self.get_c(self.m[i], self.u[i])))
                    th.append((1 - self.o[i][k]) * (2 * self.rho) - I - L)
            self.tha[i] = np.average(th)

        t = 1

        t1_, t2_, t3_ = 0, 0, 0
        start_ = time.time()

        while True:
            if do_res and t % 10 == 0:
                start = time.time()
                for i in range(len(self.s)):
                    for u in range(128, 640, 8):
                        ppp = 0
                        l = 0
                        for k in range(number_edge):
                            if x[k][i] == 0 or y[k][i] == 0:
                                continue
                            part1 = 2 * self.d[0] * u * ((self.w * self.p[i] * self.v[i]) + self.nu[i]) / (
                                    x[k][i] * self.snr[i][k])
                            part2 = self.nu[i] * 3 * edge_scale[k] * self.c[self.m[i]][0] * (u ** 2) / y[k][i]
                            part3 = self.co_map[self.m[i]][2] * self.co_map[self.m[i]][1] / (u ** (self.co_map[self.m[i]][2]+1))
                            ppp += self.o[i][k] * (part1 + part2 - part3)
                            l += self.latency(self.o[i][k], self.snr[i][k], x[k][i], y[k][i], self.s[i],
                                              self.get_d(self.u[i]),
                                              edge_scale[k] * self.get_c(self.m[i], self.u[i]))
                        if self.s[i] * ppp >= 0:
                            if l - self.tau <= 0.01:
                                self.u[i] = 32 * round(u/32)
                            else:
                                self.u[i] = max(128, 32 * round(u/32) - 32)
                            break

                t3_ += time.time() - start

            if do_server:
                start = time.time()
                for k in range(number_edge):
                    for i in range(len(self.s)):
                        if selected[i] is True:
                            continue
                        I = self.s[i] * self.w * self.p[i] * self.v[i] * self.get_d(self.u[i]) / (
                                    x[k][i] * self.snr[i][k]) - self.s[i] * self.get_i(self.m[i], self.u[i])
                        L = self.nu[i] * (self.latency(1, self.snr[i][k], x[k][i], y[k][i], self.s[i],
                                                            self.get_d(self.u[i]),
                                                            edge_scale[k] * self.get_c(self.m[i], self.u[i])))
                        if i == 11 and k == 3 and t % 2000 == 0:
                            print("\t", t, f"o={self.o[i][k]}", f"nu={self.nu[i]}", f"tha={self.tha[i]}")
                            print("\t\t", [round(sum([self.o[i][k] for k in range(number_edge)]), 4) for i in range(len(self.s))])
                        self.o[i][k] = min(1., max(math.pow(10, -15), 1. - (I + L + self.tha[i])/(2 * self.rho)))
                        if i == 11 and k == 3 and t % 2000 == 0:
                            print(f"\t\t rho={self.rho}")
                            print("\t\t", f"self.o[i][k]={self.o[i][k]}")
                            cost, cost1, cost2, diffs, success = self.get_cost(x, y, number_edge, edge_scale)
                            print(f"cost={-np.average(cost)}")

                t2_ += time.time() - start

            start = time.time()
            diffs = []
            #sum_k_o = 0
            for i in range(len(self.s)):
                l = 0
                #sum_o = 0
                for k in range(number_edge):
                    if y[k][i] != 0:
                        l += self.latency(self.o[i][k], self.snr[i][k], x[k][i], y[k][i], self.s[i], self.get_d(self.u[i]),
                                          edge_scale[k] * self.get_c(self.m[i], self.u[i]))
                    #sum_o += self.o[i][k]
                diffs.append(l - self.tau)
                #sum_k_o += (sum_o - 1)**2
                self.nu[i] = self.update_nu(self.nu[i], (0.5 / math.sqrt(t)), diffs[i]) # (0.02 / math.sqrt(t))

            t1_ += time.time() - start
            if do_server:
                start = time.time()
                for i in range(len(self.s)):
                    if not selected[i]:
                        sum_o = 0
                        for k in range(number_edge):
                            sum_o += self.o[i][k]
                        self.tha[i] = max(0., self.tha[i] + (0.5 / math.sqrt(t)) * (sum_o - 1))  # (0.02 / math.sqrt(t))

                t2_ += time.time() - start

            x, y = self.compute_x_y(edge_scale, number_edge=number_edge)

            start = time.time()
            diffs = []
            exits = 0
            for i in range(len(self.s)):
                l = 0
                sum_o = 0
                for k in range(number_edge):
                    if y[k][i] != 0:
                        l += self.latency(self.o[i][k], self.snr[i][k], x[k][i], y[k][i], self.s[i], self.get_d(self.u[i]),
                                          edge_scale[k] * self.get_c(self.m[i], self.u[i]))
                    sum_o += self.o[i][k]
                diffs.append(sum_o * l - self.tau)

            for i in range(len(self.s)):
                if np.abs(sum([self.o[i][k] for k in range(number_edge)]) - 1) <= 0.002:
                    exits += 1

            t1_ += time.time() - start

            t += 1
            if (exits == len(self.s) and t > 5001) or t == 10000:  # (exits == len(self.s) and t > 200000) or
                print(t, t1_, t2_, t3_, time.time() - start_, "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
                break
        return x, y

    def get_cost(self, x, y, number_edge, edge_scale):
        cost = []
        cost1 = []
        cost2 = []
        success = 1
        diffs = []
        for i in range(len(self.s)):
            l = 0
            cost_ = 0
            cost1_ = 0
            cost2_ = 0
            for k in range(number_edge):
                if y[k][i] != 0:
                    l += self.latency(self.o[i][k], self.snr[i][k], x[k][i], y[k][i], self.s[i], self.get_d(self.u[i]),
                                      edge_scale[k] * self.get_c(self.m[i], self.u[i]))
                    cost_ += self.o[i][k] * (self.w * self.s[i] * self.get_d(self.u[i]) * self.p[i] * self.v[i] / (x[k][i] * self.snr[i][k]) - self.s[i] * self.get_i(self.m[i], self.u[i]))
                    cost1_ += self.o[i][k] * self.s[i] * self.get_d(self.u[i]) * self.p[i] * self.v[i] / (x[k][i] * self.snr[i][k])
                    cost2_ += self.o[i][k] * self.s[i] * self.get_i(self.m[i], self.u[i])
                    # print(0.9305001540783204 - 4.898609371357725/(290**0.4227363981048906))
                    #print("##################", self.m[i], self.u[i], self.get_i(self.m[i], self.u[i]))
                    #print("\t", self.co_map[self.m[i]][0], self.co_map[self.m[i]][1], self.co_map[self.m[i]][2])
                    #print("\t", self.co_map[self.m[i]][0] - self.co_map[self.m[i]][1]/(self.u[i]**self.co_map[self.m[i]][2]))

            diffs.append(l - self.tau)
            if l - self.tau > 0.01:
                success = 0
            cost.append(cost_)
            cost1.append(cost1_)
            cost2.append(cost2_)
        return cost, cost1, cost2, diffs, success

    def display(self, x, y, edge_scale, number_edge=3):
        cost, cost1, cost2, diffs, success = self.get_cost(x, y, number_edge, edge_scale)
        print("cost1", list(np.round(np.array(cost1), 4)))
        print("cost2", list(np.round(np.array(cost2), 4)))
        print("cost2", list(np.round(np.array(cost), 4)), min(cost), max(cost), np.average(cost))
        for i in range(len(self.s)):
            print("o{}=".format(i), self.o[i])
        for k in range(number_edge):
            print("\tx in edge {} = {}".format(k, list(np.round(np.array(x[k]), 4))), np.sum(x[k]))
        for k in range(number_edge):
            print("\ty in edge {} = {}".format(k, list(np.round(np.array(y[k]), 4))), np.sum(y[k]))
        # print("m = {}".format(self.m))
        print("u = {}".format(self.u), np.average(self.u))
        print("un = {}".format(list(self.nu)))
        # print("diffs ={}".format(diffs))
        # print("costs ={}".format(list(np.round(np.array(cost), 4))), round(np.average(cost), 4))
        load = [0 for k in range(number_edge)]
        seg = [0 for k in range(number_edge)]
        diff_k = [[] for k in range(number_edge)]
        u_k = [[] for k in range(number_edge)]
        for i in range(len(self.s)):
            for k in range(number_edge):
                if self.o[i][k] == 1:
                    diff_k[k].append(round(diffs[i], 4))
                    u_k[k].append(self.u[i])
        for k in range(number_edge):
            for i in range(len(self.s)):
                if self.o[i][k] == 1:
                    load[k] += 1
                    seg[k] += self.s[i]
        print("load ={}/{}".format(load, seg), round(np.average(self.m), 4))
        for k in range(number_edge):
            print(f"diff_{k}={diff_k[k]},u={u_k[k]}")
        return round(np.average(cost), 4), success, round(np.average(self.u), 4), round(np.average(cost1), 4), round(np.average(cost2), 4)

    def go_solver(self, number_edge, edge_scale, x, y):
        L = []
        I = []
        for k in range(number_edge):
            L.append([])
            I.append([])
            for i in range(len(self.s)):
                if y[k][i] != 0:
                    L[k].append(self.latency(1, self.snr[i][k], x[k][i], y[k][i], self.s[i], self.get_d(self.u[i]),
                                      edge_scale[k] * self.get_c(self.m[i], self.u[i])))
                    I[k].append(
                        self.s[i] * ((1-self.beta) * self.w * self.p[i] * self.v[i]) * self.get_d(self.u[i]) / (x[k][i] * self.snr[i][k]) - self.s[i] * self.beta *
                        self.get_i(self.m[i], self.u[i]))

        print("L=", L)
        print("I=", I)
        print("Y=", y)
        self.o = solver(list(np.round(np.array(L), 10)), list(np.round(np.array(I), 10)), list(np.round(np.array(y), 10)), len(self.s), number_edge)

    def upgrade(self, x, y, number_edge, edge_scale):
        opt_u = []
        tcost = 0
        changed = False
        for i in range(len(self.s)):
            cost = []
            conb = []
            for index_u, resolution in enumerate([128, 192, 256, 320, 384, 512, 640]):
                index_m = self.m[i]
                conb.append([index_u, index_m])
                l = 0
                cost_ = 0
                for k in range(number_edge):
                    if y[k][i] != 0:
                        l += self.latency(self.o[i][k], self.snr[i][k], x[k][i], y[k][i], self.s[i], self.d[index_u],
                                          edge_scale[k] * self.c[index_m][index_u])
                        cost_ += self.o[i][k] * (
                                self.s[i] * self.d[index_u] * self.p[i] * self.v[i] / (
                                    x[k][i] * self.snr[i][k]) - self.s[i] * self.beta *
                                self.co_map[index_m][index_u])
                if l <= self.tau:
                    cost.append(cost_)
                else:
                    cost.append(999)
            opt_u.append(conb[cost.index(min(cost))])
            tcost += min(cost)
            if opt_u[i][0] != self.u[i]:
                changed = True
            self.u[i] = opt_u[i][0]
            if opt_u[i][1] != self.m[i]:
                changed = True
            self.m[i] = opt_u[i][1]
        return changed

    def do_equal(self, number_edge, edge_scale, alg, w):
        self.w = w
        do_server = True
        if alg != "opt":
            do_server = False
            selected = [True for i in range(len(self.s))]
        else:
            selected = [False for i in range(len(self.s))]

        stop = False
        while not stop and do_server:
            x, y = self.do_resource(selected, edge_scale, number_edge=number_edge, do_server=do_server)
            self.display(x, y, edge_scale, number_edge=number_edge)
            if do_server:
                print("#######################################")
                print("selected", selected)
                changed = 0
                not_selected = []
                for i in range(len(self.s)):
                    #target_k = [self.o[i].index(max(self.o[i]))]
                    target_k = []
                    target_v = []
                    for k in range(number_edge):
                        if self.o[i][k] >= 0.5:
                            target_k.append(k)
                            target_v.append(self.o[i][k])
                    if len(target_k) > 0 and selected[i] is False:
                        k_id = target_k[target_v.index(max(target_v))]
                        selected[i] = True
                        changed += 1
                        for k in range(number_edge):
                            if k == k_id:
                                self.o[i][k] = 1
                            else:
                                self.o[i][k] = 0
                    else:
                        if not selected[i]:
                            print("\t o{}=".format(i), self.o[i])
                            not_selected.append(i)

                print("selected", selected)
                print(f"changed={changed}, not_selected Ids={not_selected}")
                if changed == 0:
                    i = random.choice(not_selected)
                    target_v = max(self.o[i])
                    target_k = []
                    for k in range(number_edge):
                        if self.o[i][k] == target_v:
                            target_k.append(k)

                    if len(target_k) > 1:
                        load = [0 for k in range(len(target_k))]
                        for k in range(len(target_k)):
                            for j in range(len(self.s)):
                                if self.o[j][k] == 1. and j != i:
                                    load[k] += self.s[j]
                        target = target_k[load.index(min(load))]
                        print("force", target_k, load, target, "i=", i)
                    else:
                        target = target_k[0]

                    for k in range(number_edge):
                        if k == target:
                            self.o[i][k] = 1
                        else:
                            self.o[i][k] = 0
                    selected[i] = True
                    not_selected.remove(i)
                    print(f"change Id={i} to selected with k={target_k}")

                for i in not_selected:
                    for k in range(number_edge):
                        self.o[i][k] = 1./number_edge

                print("#######################################")
            stop = True
            for i in range(len(self.s)):
                if selected[i] is False:
                    stop = False

        x, y = self.do_resource(selected, edge_scale, number_edge=number_edge, do_server=False)
        changed = False
        for i in range(len(self.s)):
            for k in range(number_edge):
                if self.o[i][k] == 1:
                    while self.latency(self.o[i][k], self.snr[i][k], x[k][i], y[k][i], self.s[i], self.get_d(self.u[i]),
                                       edge_scale[k] * self.get_c(self.m[i], self.u[i])) - self.tau > 0.01:
                        if self.u[i] == 128:
                            break
                        else:
                            self.u[i] = max(128, self.u[i] - 32)
                            changed = True
        if changed:
            x, y = self.do_resource(selected, edge_scale, number_edge=number_edge, do_server=False, do_res=False)
        return self.display(x, y, edge_scale, number_edge=number_edge)


def worker(items, o=None):
    i, data, number, edge, tau, alg, w = items
    d = data["d"]
    s = data["s"]
    p = data["p"]
    v = data["v"]
    c = data["c"]
    m = data["m"]
    b = data["b"]
    snr = data["snr"]
    co_map = data["co_map"]
    scale = data["scale"][i]
    if o is None:
        o = [[0 for j in range(edge)] for i in range(number)]
        if alg == "rnd":
            IDs = [j for j in range(number)]
            for k in range(edge):
                IDs_selected = random.sample(IDs, int(number / edge))
                for j in range(number):
                    if j in IDs_selected:
                        o[j][k] = 1
                        IDs.remove(j)
                    else:
                        o[j][k] = 0
        elif alg == "snr":
            for j in range(number):
                pp = []
                for k in range(edge):
                    pp.append(snr[i][j][k])
                k = pp.index(max(pp))
                o[j][k] = 1
        else:
            o = [[1. / edge for j in range(edge)] for i in range(number)]
    for j in range(number):
        for k in range(edge):
            snr[i][j][k] = math.log2(1 + math.pow(10, snr[i][j][k] / 20))
    u = [128 for xx in range(number)]
    try:
        opt = Optimize(o, b[i], snr[i], s[i], d, c, p[i], v[i], m[i], u, tau, co_map)
        cost_avg, success_, res_avg, cost_energy, cost_map = opt.do_equal(number_edge=edge, edge_scale=scale, alg=alg, w=w)
        print("################")
        print("\t", i, "cost=", cost_avg, success_, res_avg, "energy=", cost_energy, "cost_map=", cost_map)
        print("################")
    except:
        print(trace.__format__())
    return success_, res_avg, cost_avg, cost_energy, cost_map, opt.o


def dist(data, number, edge, tau, start, end, alg, w):
    items = []
    # w = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 24, 30, 32, 34, 38, 42]
    #for w in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 24, 30, 32, 34, 38, 42]:
    for i in range(start, end):
        items.append((i, data, number, edge, tau, alg, w))

    start = time.time()

    with Pool(14) as p:
        results = p.map(worker, items)

    cost = []
    hist_u = []
    cost_energy, cost_map = [], []
    success = []
    hist_o = []
    for item in results:
        success.append(item[0])
        hist_u.append(item[1])
        cost.append(item[2])
        cost_energy.append(item[3])
        cost_map.append(item[4])
        hist_o.append(item[5])

    print(round(time.time() - start), "success", sum(success), "cost", round(-np.average(cost), 4), "resolution", round(np.average(hist_u)))
    print("energy=", round(np.average(cost_energy), 4), "map=", round(np.average(cost_map), 4))
    print("cost=", cost)
    print("success=", success)
    print("o=", hist_o)
    print("energy=", cost_energy)
    print("map=", cost_map)
    return round(-np.average(cost), 4), round(np.average(cost_energy), 4), round(np.average(cost_map), 4)

"""
            for i in range(len(self.s)):
                target_v = max(self.o[i])
                target_k = []
                for k in range(number_edge):
                    if self.o[i][k] == target_v:
                        target_k.append(k)

                if len(target_k) > 1:
                    load = [0 for k in range(len(target_k))]
                    for k in range(len(target_k)):
                        for j in range(len(self.s)):
                            if self.o[j][k] == 1. and j != i:
                                load[k] += self.s[j]
                    target = target_k[load.index(min(load))]
                    print("force", target_k, load, target, "i=", i)
                else:
                    target = target_k[0]

                for k in range(number_edge):
                    if k == target:
                        self.o[i][k] = 1
                    else:
                        self.o[i][k] = 0
            """
