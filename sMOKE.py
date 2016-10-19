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


def getFiles(path, data):
    datafiles = glob.glob(path+"/*averaged.txt")
    if(datafiles != []):
        for dat in datafiles:
            data.append(dat)
    files = glob.glob(path+"/*")
    for file in files:
        getFiles(file, data)
        

def getData(path):
    data = []
    getFiles(path,data)
    return data
    
data_path = "/Users/nikolaj/Desktop/rzchowski/161016/1"
#data = getData(data_path)
data = glob.glob(data_path+"/*averaged.txt")

file = open("out.txt", "w") 
file.write("file\tleft_sat(b,v)\tright_sat(b,v)\tleft_slope\tright_slope\tarea\n")

size = int(np.sqrt(len(data)))+1

f, axarr = plt.subplots(size, size)

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
    rawloop = np.loadtxt(data[q], delimiter = '\t', skiprows = 2)
             
    loop=[[],[]]
    for entry in rawloop:
        loop[0].append(entry[0])
        loop[1].append(entry[1])
    
    
    """clean/process data """
    
    smoothed = [[],[]]
    smoothed[0] = gaussian_filter(loop[0], 20)
    smoothed[1] = gaussian_filter(loop[1], 20)
    
    dB = smoothed[0]
    dV = np.gradient(smoothed[1])
    dV2 = np.gradient(dV)
    dV2 = gaussian_filter(dV2, 10)
    dV3 = np.gradient(dV2)
    
    """find saturation points, satright and satleft"""
    maxima = [[],[]]
    minima = [[],[]]
    tempmax = argrelextrema(dV3, np.greater)
    tempmax = tempmax[0]
    tempmin = argrelextrema(dV3, np.less)
    tempmin = tempmin[0]
    high = max(dV3)/2
    low = min(dV3)/2
    
    upper = high - np.abs(high-low)/4
    lower = low + np.abs(high-low)/4
    
    for point in tempmax:
        if(dV3[point]>upper):
            maxima[0].append(point)
            maxima[1].append(dB[point])
    
    satright = maxima[0][np.argmax(maxima[1])]
    
    for point in tempmin:
        if(dV3[point]<lower):
            minima[0].append(point)
            minima[1].append(dB[point])
    
    satleft = minima[0][np.argmin(minima[1])]
    
    """find x0 slope"""
    ts = 10

    intercepts = [[],[]]
    x0 = np.average(smoothed[1])
    for i in range(1,len(smoothed[0])-1):
        if(smoothed[1][i-1]>x0 and smoothed[1][i+1]<x0):
            intercepts[0].append(i)
        if(smoothed[1][i-1]<x0 and smoothed[1][i+1]>x0):
            intercepts[1].append(i)
            
    lintercept = intercepts[0][len(intercepts[0])-1]
    rintercept = intercepts[1][len(intercepts[0])-1]
    lslope = np.gradient(smoothed[0][lintercept-ts:lintercept+ts])/np.gradient(smoothed[1][lintercept-ts:lintercept+ts])
    rslope = np.gradient(smoothed[0][rintercept-ts:rintercept+ts])/np.gradient(smoothed[1][rintercept-ts:rintercept+ts])
    lslope = np.average(lslope)
    rslope = np.average(rslope)
    """find loop area"""
    
    left=np.argmin(smoothed[0])
    right=np.argmax(smoothed[0])
    top_area=np.trapz(smoothed[1][right:left],smoothed[0][right:left])
    bottom_area1=np.trapz(smoothed[1][0:right],smoothed[0][0:right])
    bottom_area2=np.trapz(smoothed[1][left:len(smoothed[0])+1],smoothed[0][left:len(smoothed[0])+1])
    area=top_area-(bottom_area1+bottom_area2)
    
    """write to output file"""
    
    name = data[q].split("/")
    name = name[len(name)-1]
    name = name[0:len(name)-13]
    
    b = smoothed[0]
    v = smoothed[1]
    out = str(name)+"\t"+str(b[satleft])+","+str(v[satleft])+"\t"
    out += str(b[satright])+","+str(v[satright])+"\t"
    out += str(lslope)+ "\t" + str(rslope)+"\t" + str(area)+"\n"
    file.write(out)
    
    curr.set_title(name, fontsize = 10)
    curr = axarr[q%size, int(q/size)]
    curr.plot(b,v, 'k')
    curr.plot(b[satright],v[satright],'ro', alpha = .5)
    curr.plot(b[satleft],v[satleft],'bo', alpha = .5)
    #curr.plot(b[lintercept],v[lintercept], 'b*')
    #curr.plot(b[rintercept],v[rintercept], 'r*')
    
    ts = ts/1000000
    
    ltanx = [b[lintercept]-ts, b[lintercept]+ts]
    ltany = [v[lintercept]-ts*lslope, v[lintercept]+ts*lslope]
    
    rtanx = [b[rintercept]-ts, b[rintercept]+ts]
    rtany = [v[rintercept]-ts*lslope, v[rintercept]+ts*lslope]
    
    curr.plot(ltanx, ltany, 'b', linewidth = 2)
    curr.plot(rtanx, rtany, 'r', linewidth = 2)
    
file.close()
