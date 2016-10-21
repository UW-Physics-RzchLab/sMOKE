# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 17:48:30 2016

@author: nikolaj
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from os.path import join


def getFiles(path, data):
    datafiles = glob.glob(path + "/*averaged.txt")
    if(datafiles != []):
        for dat in datafiles:
            data.append(dat)
    files = glob.glob(path + "/*")
    for file in files:
        getFiles(file, data)


def getData(path):
    data = []
    getFiles(path, data)
    return data


def getxy(path):
    return int(path[len(path) - 18]), int(path[len(path) - 14])

data_path = r"C:\Users\rzchlab\Google Drive\NickelPMN-PT\291-2\ScMOKE\161016\1"
#data = getData(data_path)
data = glob.glob(join(data_path, "*averaged.txt"))

file = open(join(data_path, "out.txt"), "w")
headers = "\t".join(("x", "y", "left_sat(B)", "left_sat(V)", "right_sat(B)",
                     "right_sat(V)", "left_slope", "right_slope", "area\n"))
file.write(headers)

mx, my = getxy(data[len(data) - 1])
f, axarr = plt.subplots(mx + 1, my + 1)

for line in axarr:
    for curr in line:
        curr.spines["left"].set_visible(False)
        curr.spines["right"].set_visible(False)
        curr.spines["top"].set_visible(False)
        curr.spines["bottom"].set_visible(False)

        curr.xaxis.set_visible(False)
        curr.yaxis.set_visible(False)

for q in range(len(data)):
    print(data[q])
    x, y = np.loadtxt(data[q], delimiter='\t', skiprows=2, usecols=(0, 1), 
                      unpack=True)

    """clean/process data """

    smoothed = [[], []]
    smoothed[0] = gaussian_filter(x, 20)
    smoothed[1] = gaussian_filter(y, 20)

    dB = smoothed[0]
    dV = np.gradient(smoothed[1])
    dV2 = np.gradient(dV)
    dV2 = gaussian_filter(dV2, 10)
    dV3 = np.gradient(dV2)

    """find saturation points, satright and satleft"""
    maxima = [[], []]
    minima = [[], []]
    tempmax = argrelextrema(dV3, np.greater)
    tempmax = tempmax[0]
    tempmin = argrelextrema(dV3, np.less)
    tempmin = tempmin[0]
    high = max(dV3) / 2
    low = min(dV3) / 2

    upper = high - np.abs(high - low) / 4
    lower = low + np.abs(high - low) / 4

    for point in tempmax:
        if(dV3[point] > upper):
            maxima[0].append(point)
            maxima[1].append(dB[point])

    satright = maxima[0][np.argmax(maxima[1])]

    for point in tempmin:
        if(dV3[point] < lower):
            minima[0].append(point)
            minima[1].append(dB[point])

    satleft = minima[0][np.argmin(minima[1])]

    """find x0 slope"""
    ts = 10

    intercepts = [[], []]
    x0 = np.average(smoothed[1])
    # TODO: Replace this loop with something like:
    # np.argmin(np.abs(yarray-yarray.mean()))
    for i in range(1, len(smoothed[0]) - 1):
        if(smoothed[1][i - 1] > x0 and smoothed[1][i + 1] < x0):
            intercepts[0].append(i)
        if(smoothed[1][i - 1] < x0 and smoothed[1][i + 1] > x0):
            intercepts[1].append(i)
            


    lintercept = intercepts[0][-1]
    rintercept = intercepts[1][-1]
    lslope = (np.gradient(smoothed[0][lintercept - ts:lintercept + ts]) / 
              np.gradient(smoothed[1][lintercept - ts:lintercept + ts]))
    rslope = (np.gradient(smoothed[0][rintercept - ts:rintercept + ts]) /
              np.gradient(smoothed[1][rintercept - ts:rintercept + ts]) )
    lslope = np.average(lslope)
    rslope = np.average(rslope)
    """find loop area"""

    left = np.argmin(smoothed[0])
    right = np.argmax(smoothed[0])
    top_area = np.trapz(smoothed[1][right:left], smoothed[0][right:left])
    bottom_area1 = np.trapz(smoothed[1][0:right], smoothed[0][0:right])
    bottom_area2 = np.trapz(smoothed[1][left:len(
        smoothed[0]) + 1], smoothed[0][left:len(smoothed[0]) + 1])
    area = top_area - (bottom_area1 + bottom_area2)

    """write to output file"""

    x, y = getxy(data[q])
    x = str(x)
    y = str(y)

    b = smoothed[0]
    v = smoothed[1]

    curr = axarr[int(x), int(y)]

    # TODO: Better to store parameters (saturation values, areas, etc..) in
    # a 2d array and then write them out with np.savetxt
    out = x + "\t" + y + "\t" + str(b[satleft]) + "\t" + str(v[satleft]) + "\t"
    out += str(b[satright]) + "\t" + str(v[satright]) + "\t"
    out += str(lslope) + "\t" + str(rslope) + "\t" + str(area) + "\n"
    file.write(out)
    curr.set_title(x + "," + y, fontsize=10)
    curr.plot(b, v, 'k')
    curr.plot(b[satright], v[satright], 'ro', alpha=.5)
    curr.plot(b[satleft], v[satleft], 'bo', alpha=.5)
    #curr.plot(b[lintercept],v[lintercept], 'b*')
    #curr.plot(b[rintercept],v[rintercept], 'r*')

    ts = ts / 1000000

    ltanx = [b[lintercept] - ts, b[lintercept] + ts]
    ltany = [v[lintercept] - ts * lslope, v[lintercept] + ts * lslope]

    rtanx = [b[rintercept] - ts, b[rintercept] + ts]
    rtany = [v[rintercept] - ts * lslope, v[rintercept] + ts * lslope]

    curr.plot(ltanx, ltany, 'b', linewidth=2)
    curr.plot(rtanx, rtany, 'r', linewidth=2)

file.close()
