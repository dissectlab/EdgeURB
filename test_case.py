import math
import random
import json

if __name__ == '__main__':

    data = {
        "d": [0.000004037153221, 0.057205054378969],
        "co_map": [
            [0.9305001540783204, 4.898609371357725, 0.4227363981048906],
            [0.8746499446127548, 5.276655862661606, 0.45022082564696914],
            [0.6669540561368673, 13.118280737206824, 0.7095565400457234],
            [0.5816710952990591, 13.628604144917894, 0.728535155157919]
        ],
        "c": [
            [0.000000000106067, 0.017446414063656],
            [0.000000000064925, 0.014596814933767],
            [0.000000000055085, 0.011701821983763],
            [0.000000000041614, 0.009515105442959]
        ]
    }

    number = 56
    edge = 8

    data["s"] = []
    data["p"] = []
    data["v"] = []
    data["m"] = []
    data["b"] = []
    data["snr"] = []
    data['scale'] = []
    data['x'] = []
    data['y'] = []

    for kk in range(100):
        data["s"].append([random.randint(5, 15) for i in range(number)])
        # data["snr"].append([[random.randint(10, 30) for k in range(edge)] for i in range(number)])  # SNR (db)
        data["b"].append([random.choice([15, 20, 25, 30]) for i in range(edge)])  # bandwidth [190, 195, 200, 205] [15, 20, 25, 30]
        data["p"].append([round(random.uniform(0.3, 0.6), 2) for i in range(number)])
        data["v"].append([round(random.uniform(3.5, 4.5), 2) for i in range(number)])
        data["m"].append([random.randint(0, 3) for i in range(number)])
        data['scale'].append([round(random.uniform(0.2, 1.), 2) for i in range(edge)])

        data["x"].append([random.randint(50, 2000) for i in range(number)])
        data["y"].append([random.randint(50, 2000) for i in range(number)])

        sx = [500, 1500, 1500, 500, 750, 1000, 1250, 750]
        sy = [500, 1500, 500, 1500, 1250, 500, 1250, 750]

        snr = []
        for i in range(number):
            snr.append([])
            for k in range(edge):
                dist_x = data["x"][-1][i] - sx[k]
                dist_y = data["y"][-1][i] - sy[k]
                dist = math.sqrt(dist_x ** 2 + dist_y ** 2)
                snr[-1].append(round((1 - dist / 2000) * 20 + 10))
        data["snr"].append(snr)
        """
        data["snr"].append([[random.randint(5, 30) for k in range(edge)] for i in range(number)])  # SNR (db)
        """


    #with open('setting_8_56.json', 'w') as f:
    #    json.dump(data, f, indent=4)
    """
   
    for k in range(8):
        for i in range(len(o)):
            if o[i][k] == 1:
                xx[k].append(x[i])
                yy[k].append(y[i])

    print("x1=", xx[0])
    print("x2=", xx[1])
    print("x3=", xx[2])
    print("x4=", xx[3])
    print("x5=", xx[4])
    print("x6=", xx[5])
    print("x7=", xx[6])
    print("x8=", xx[7])

    print("y1=", yy[0])
    print("y2=", yy[1])
    print("y3=", yy[2])
    print("y4=", yy[3])
    print("y5=", yy[4])
    print("y6=", yy[5])
    print("y7=", yy[6])
    print("y8=", yy[7])

    """