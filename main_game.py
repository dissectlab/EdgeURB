import json
import random
import math
import numpy as np
from inspect import trace
from result.fix_x import Optimize, dist

if __name__ == '__main__':

    tau = 1
    number = 16
    edge = 4
    f = open(f'setting_{edge}_{number}.json')
    data = json.load(f)

    i = 0
    w = 8
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
    o = [[0 for j in range(edge)] for i in range(number)]
    for j in range(number):
        pp = []
        for k in range(edge):
            pp.append(snr[i][j][k])
        k = pp.index(max(pp))
        o[j][k] = 1
    for j in range(number):
        for k in range(edge):
            snr[i][j][k] = math.log2(1 + math.pow(10, snr[i][j][k] / 20))

    u = [128 for xx in range(number)]
    while True:
        for n in range(len(s)):
            # find best server for given server selection from others
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            pre_server = o[i]
            cost = []
            for k in range(edge):
                this_s = [s[i][n]]
                this_p = [p[i][n]]
                this_v = [v[i][n]]
                this_m = [m[i][n]]
                this_u = [u[n]]
                this_snr = [[[snr[i][n][k]]]]
                this_scale = [scale[k]]
                for j in range(len(s)):
                    if o[j][k] == 1 and j != i:
                        this_s.append(s[i][j])
                        this_p.append(p[i][j])
                        this_v.append(v[i][j])
                        this_m.append(m[i][j])
                        this_u.append(u[i][j])
                if len(this_s) == 0:
                    continue
                try:
                    opt = Optimize(o, b[i], snr[i], s[i], d, c, p[i], v[i], m[i], u, tau, co_map)
                    cost_avg, success_, res_avg, cost_energy, cost_map = opt.do_equal(number_edge=1, edge_scale=scale, alg="snr", w=w)
                    print("################")
                    print("\t", i, cost_avg, success_, res_avg, "energy=", cost_energy, "cost_map=", cost_map)
                    print("################")
                except:
                    print(trace.__format__())