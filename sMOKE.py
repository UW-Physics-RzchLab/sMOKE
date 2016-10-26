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
from matplotlib.widgets import CheckButtons


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
    
def getxy(path):
    return int(path[len(path)-18]), int(path[len(path)-14])
    
def Hc_of(x, y, ks=2):
    # Setup indices
    gt0idx = x >= 0
    lt0idx = x < 0
    ymgt0idx = np.argmin(np.abs(y[gt0idx]))
    ymlt0idx = np.argmin(np.abs(y[lt0idx]))
    # Compute Hc
    Hc_gt0 = x[gt0idx][ymgt0idx - ks:ymgt0idx + ks].mean()
    Hc_lt0 = x[lt0idx][ymlt0idx - ks:ymlt0idx + ks].mean()
    Hc_avg = (abs(Hc_gt0) + abs(Hc_lt0))/2.
    return Hc_avg
    
def Mrem_of(x, y, ks=3):
    # Setup indices
    N = len(x)+1
    inds = np.arange(N).reshape(4, N//4)
    yq03 = y[inds[[0, 3]]-1].reshape(N//2) # yq03 = y quarters 0 and 3
    xq03 = x[inds[[0, 3]]-1].reshape(N//2)
    yq12 = y[inds[[1, 2]]-1].reshape(N//2)
    xq12 = x[inds[[1, 2]]-1].reshape(N//2)
    xmq03i = np.argmin(np.abs(xq03)) # xmq03i = indsof x min quarters 0 and 3
    xmq12i = np.argmin(np.abs(xq12))
    # Average over the kernel size
    yq03avg = abs(np.mean(yq03[xmq03i-ks:xmq03i+ks]))
    yq12avg = abs(np.mean(yq12[xmq12i-ks:xmq12i+ks]))
    mrem = (yq03avg + yq12avg)/2.
    return mrem
    
data_path = "/Users/nikolaj/Desktop/rzchowski/161016/0"
#data = getData(data_path)
data = glob.glob(data_path+"/*averaged.txt")



file = open("out.txt", "w") 
file.write("x\ty\t\tleft_sat(B)\tleft_sat(V)\tright_sat(B)\tright_sat(V)\tleft_slope\tright_slope\tarea\n")

mx, my = getxy(data[len(data)-1])
f, axarr = plt.subplots(mx+1, my+1)

mrems = np.zeros((mx+1, my+1))
hcs = np.zeros((mx+1, my+1))

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
    B = smoothed[0] = gaussian_filter(loop[0], 20)
    V = smoothed[1] = gaussian_filter(loop[1], 20)
    
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
    high = np.max(dV3)/2
    low = np.min(dV3)/2
    
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
    
    
    """find mrem and hc"""
    mrem = Mrem_of(B,V)
    hc = Hc_of(B,V)
    
    """write to output file"""
    
    x,y = getxy(data[q])
    x = str(x)
    y = str(y)
    
    b = smoothed[0]
    v = smoothed[1]
    

    out = x+"\t"+y+"\t"+str(b[satleft])+"\t"+str(v[satleft])+"\t"
    out += str(b[satright])+"\t"+str(v[satright])+"\t"
    out += str(lslope)+ "\t" + str(rslope)+"\t" + str(area)+"\n"
    file.write(out)
    
    """plot data"""   
    curr = axarr[int(x), int(y)]
    curr.set_title(x+","+y, fontsize = 10)
    graph_data = curr.plot(b,v, 'g', alpha = .7)
    curr.plot(b[satright],v[satright],'ro', alpha = .5)
    curr.plot(b[satleft],v[satleft],'bo', alpha = .5)
    
    mrems[int(x)][int(y)] = mrem
    hcs[int(x)][int(y)] = hc
    
    ts = ts/1000000
    
    ltanx = [b[lintercept]-ts, b[lintercept]+ts]
    ltany = [v[lintercept]-ts*lslope, v[lintercept]+ts*lslope]
    
    rtanx = [b[rintercept]-ts, b[rintercept]+ts]
    rtany = [v[rintercept]-ts*lslope, v[rintercept]+ts*lslope]
    
    curr.plot(ltanx, ltany, 'b', linewidth = 2)
    curr.plot(rtanx, rtany, 'r', linewidth = 2)
    
"""plot hc and mrem"""

def switch(label):
    vals = []

    if(label == 'hc'):
        vals = mrems
        check.labels[0].set_text('mrem')
    if(label == 'mrem'):
        vals = hcs
        check.labels[0].set_text('hc')
    for y,line in enumerate(axarr):
        for x,curr in enumerate(line):
                curr.set_axis_bgcolor(str(vals[x][y]/np.max(vals)))

    legend_string="hc/mrem values \n"
    for y, line in enumerate(vals): 
        for x, val in enumerate(line):
            legend_string += (str(x)+","+str(y)+":"+str(val)[0:4]+"\n")
    if(label == 'show\nvalues'):
        f.text(0, 0, 'hi', visible = False)
        label = 'hide\nvalues'
    if(label == 'hide\nvalues'):
        f.text(.91, 0, legend_string)
        label = 'show\nvalues'
    plt.draw()

#switch('hc')
#switch('show\nvalues')
rax = plt.axes([0.01, 0.4, 0.095, 0.15])
check = CheckButtons(rax, ('hc', 'show\nvalues'),  (True, False))
check.on_clicked(switch)

        
        
    
file.close()
