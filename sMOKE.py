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
    return mrem, np.array((xmq03i, xmq12i))
    
def saturation_index(x, y, positive_side=True):
    # Compute some constants
    high, low = np.max(y)/2, np.min(y)/2
    mag = np.abs(high - low)
    upper, lower = high - mag/4, low + mag/4
    
    # Branch a few operations based on the positive_side flag
    comp_func = np.greater if positive_side else np.less
    extr_func = np.argmax if positive_side else np.argmin
    
    # Compute the saturation_index    
    extr_inds, = argrelextrema(y, comp_func)
    y_lextr = y[extr_inds]
    condition = y_lextr > upper if positive_side else y_lextr < lower    
    x_extr_ind = extr_func(x[extr_inds[condition]])
    return extr_inds[condition][x_extr_ind]

def intercept_indices(y):
    """ [description of algorithm and assumptions]"""            
    intercepts = [[],[]]
    x0 = np.average(y)
    for i in range(1,len(y)-1):
        if(y[i-1]>x0 and y[i+1]<x0):
            intercepts[0].append(i)
        if(y[i-1]<x0 and y[i+1]>x0):
            intercepts[1].append(i)    
    return intercepts[0][-1], intercepts[1][-1]
    
def avg_gradient(x, y, center, halfwidth):
    return np.mean(np.gradient(B[center-halfwidth:center+halfwidth]) / 
                   np.gradient(V[center-halfwidth:center+halfwidth]))
                   

def hyst_loop_area(x, y):
    left, right = np.argmin(x), np.argmax(x)
    top_area = np.trapz(y[right:left], x[right:left])
    bottom_area1 = np.trapz(y[0:right], x[0:right])
    bottom_area2 = np.trapz(y[left:], x[left:])
    return top_area - (bottom_area1 + bottom_area2)
    
    
#data_path = "/Users/nikolaj/Desktop/rzchowski/161016/0"
data_path = r"C:\Users\rzchlab\Desktop\trial1_5x5_BFO_test_sample\trial1_5x5_BFO_test_sample"

# Get the list of files with hysteresis loop data in them
datafiles = glob.glob(join(data_path, "*averaged.txt"))

file = open(join(data_path, "out.txt"), "w") 
headers = "\t".join(("x", "y", "left_sat(B)", "left_sat(V)", "right_sat(B)", 
                     "right_sat(V)", "left_slope", "right_slope", "area\n"))
file.write(headers)

mx, my = getxy(datafiles[-1])
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

for q, df in enumerate(datafiles):    
    print(df)
    x, y = np.loadtxt(df, usecols=(0, 1), unpack=True, delimiter = '\t', 
                      skiprows = 2)
      
    # smooth data
    smoothed = [[],[]]
    B = smoothed[0] = gaussian_filter(x, 20)
    V = smoothed[1] = gaussian_filter(y, 20)
    
    # compute derivatives of data
    dB = smoothed[0]
    dV = np.gradient(smoothed[1])
    dV2 = gaussian_filter(np.gradient(dV), 10)
    dV3 = np.gradient(dV2)
    
    # find saturation points, satright and satleft
    high = np.max(dV3)/2
    low = np.min(dV3)/2
    mag = np.abs(high-low)
    upper = high - mag/4
    lower = low + mag/4
    
    # Find the index where saturation is reached on each side of the loop.
    # This is done by finding a the peak in the third derivative of V with
    # the largest/smallest B value (depending on the side of the loop).     
    satright = saturation_index(dB, dV3, positive_side=True)
    satleft = saturation_index(dB, dV3, positive_side=False)
    
    # Note that these are indices not values
    lincept, rincept = intercept_indices(V)
    width = 10
    
    lslope, rslope = (avg_gradient(B, V, i, width) for i in (lincept, rincept))
    area = hyst_loop_area(B, V)

    # find mrem using JIs algorithm. Should give same value as NRs method,
    # might as well compute both and compare them. This one returns values
    # not indices
    mrem, mrem_inds = Mrem_of(B, V, ks=10)
    
    # TODO: This needs to be checked still
    hc = Hc_of(B,V)
    
    # write to output file  
    x,y = getxy(df)
    x = str(x)
    y = str(y)
    
    b = smoothed[0]
    v = smoothed[1]

    out = x+"\t"+y+"\t"+str(b[satleft])+"\t"+str(v[satleft])+"\t"
    out += str(b[satright])+"\t"+str(v[satright])+"\t"
    out += str(lslope)+ "\t" + str(rslope)+"\t" + str(area)+"\n"
    file.write(out)
    
    # plot data
    curr = axarr[int(x), int(y)]
    curr.set_title(x+","+y, fontsize = 10)
    graph_data = curr.plot(b,v, 'g', alpha = .7)
    curr.plot(b[satright],v[satright],'ro', alpha = .5)
    curr.plot(b[satleft],v[satleft],'bo', alpha = .5)
    # TODO: plot Hc, mRem(jji version)
    curr.plot(B[mrem_inds], V[mrem_inds], 'bx', alpha=0.5, ms=10)
    
    mrems[int(x)][int(y)] = mrem
    hcs[int(x)][int(y)] = hc
    
    ts = width/1e6
    
    ltanx = [b[lincept]-ts, b[lincept]+ts]
    ltany = [v[lincept]-ts*lslope, v[lincept]+ts*lslope]
    
    rtanx = [b[rincept]-ts, b[rincept]+ts]
    rtany = [v[rincept]-ts*lslope, v[rincept]+ts*lslope]
    
    curr.plot(ltanx, ltany, 'b', linewidth = 2)
    curr.plot(rtanx, rtany, 'r', linewidth = 2)

    
# plot hc and mrem
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

rax = plt.axes([0.01, 0.4, 0.095, 0.15])
check = CheckButtons(rax, ('hc', 'show\nvalues'),  (True, False))
check.on_clicked(switch)


file.close()
