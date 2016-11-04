# -*- coding: utf-8 -*-
"""sMOKE.py plots and analyzes scanning MOKE data.

Usage:
    sMOKE.py --help
    sMOKE.py plotarb --dir=<directory> [options]
    sMOKE.py plotscan --dir=<directory>

Options:
    -h --help                  Show this message.
    -d --dir=<dir>             Root directory above all data that is to be
                               plotted. If 'plotscan' this needs to be the
                               directory that was output by the scanning
                               MOKE labview program.
    -p --pattern=<pattern>     Regex used to select files to be plotted
                               [default: .*]
    -n --depth=<int>           Number of levels down in the file tree to search
                               for data files when in 'plotarb' mode.
                               [default: 1]
    -o --output=<filename>     Output filename. Output dir will be whatever
                               was passed to the '-d' switch.
                               [default: out.txt]
    -O --orientation=<NESW>    A string with some permutation of the characters
                               'NESW' (cardinal directions). The first two
                               characters specify the location of the origin
                               (0, 0) on the figure. For example, NE would
                               be the upper right. The third character is the
                               +x direction and the fourth character is the +y
                               direction. For example SENW means (0,0) is at
                               the lower right of the figure, (i, 0) is i
                               rows above (0, 0) and (0, j) is j columns to
                               the left of (0, 0).
                               [default: NWSE]
    --

Examples:
    sMOKE.py plotarb -d /path/to/data/dir -p .*flag=1.* --depth=2
             --output=outputfile.txt
"""
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.signal import argrelextrema
from os.path import join, basename
from matplotlib.widgets import CheckButtons
from docopt import docopt
from retarget import Targeter

"""
Todo
 - way to rotate coordinates
     - 'origin' flag chooses between '(N|S)(E|W)'
     - define +x and +y as (N|E|S|W)
 - fix interactivity

"""

###############################################################################
# Functions
###############################################################################


def get_grid_xy(mode, paths):
    """Get size of grid required to plot data. Read from data file name if
    in scan mode. If in 'arb' mode choose the smallest square grid that will
    fit all of the data.
    """
    if mode == 'scan':
        path = sorted(paths)[-1]
        x, y = get_xy(path)
        return (x + 1), (y + 1)
    elif mode == 'arb':
        x = np.ceil(np.sqrt(len(paths)))
        return int(x), int(x)


def get_xy(path):
    """Read from data file name if in scan mode.
    """
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
    N = len(x)+1
    # Divide into 4 quarters
    inds = np.arange(N).reshape(4, N//4) - 1
    yq03 = y[inds[[0, 3]]].reshape(N//2)  # yq03 = y quarters 0 and 3
    xq03 = x[inds[[0, 3]]].reshape(N//2)
    yq12 = y[inds[[1, 2]]].reshape(N//2)
    xq12 = x[inds[[1, 2]]].reshape(N//2)
    # get indices of B=0 in the half-arrays
    xmq03i = np.argmin(np.abs(xq03))  # xmq03i = indsof x min quarters 0 and 3
    xmq12i = np.argmin(np.abs(xq12))
    # convert to indices in the full arrays
    rem_ind_03 = inds[[0, 3]].reshape(N//2)[xmq03i]
    rem_ind_12 = inds[[1, 2]].reshape(N//2)[xmq12i]
    # Average over the kernel size
    yq03avg = abs(np.mean(yq03[xmq03i-ks:xmq03i+ks]))
    yq12avg = abs(np.mean(yq12[xmq12i-ks:xmq12i+ks]))
    mrem = (yq03avg + yq12avg)/2.
    return mrem, np.array((rem_ind_03, rem_ind_12))


def better_Hc_of(x, y):
    i = intercept_indices(y)
    return (np.abs(x[i[0]])+np.abs(x[i[1]]))/2


def better_Mrem_of(x, y):
    i = intercept_indices(y)

    intercepts = [[], []]
    y0 = (x[i[0]] + x[i[1]])/2

    for i in range(1, len(y)-1):
        if(x[i-1] > y0 and x[i+1] < y0):
            intercepts[0].append(i)
        if(x[i-1] < y0 and x[i+1] > y0):
            intercepts[1].append(i)
    i = intercepts[0][-1], intercepts[1][-1]
    return (np.abs(y[i[0]])+np.abs(y[i[1]]))/2


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
    intercepts = [[], []]
    x0 = np.average(y)
    for i in range(1, len(y) - 1):
        if(y[i - 1] > x0 and y[i + 1] < x0):
            intercepts[0].append(i)
        if(y[i - 1] < x0 and y[i + 1] > x0):
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


def get_data_files(mode, data_path, pattern=None, depth=1):
    if mode == 'scan':
        return glob.glob(join(data_path, "*averaged.txt"))
    elif mode == 'arb':
        t = Targeter()
        return t.acquire(data_path, pattern=pattern, depth=depth)


def get_axarr_coords(origin, xplus, yplus, N, i, j):
    origin, xplus, yplus = (x.upper() for x in (origin, xplus, yplus))
    verify_directions(origin, xplus, yplus)
    xh, yh, zh = np.array((1, 0, 0)), np.array((0, 1, 0)), np.array((0, 0, 1))
    unit_vs = {'N': yh, 'E': xh,
               'S': -yh, 'W': -xh}
    deg_ccw = {'NW': 0, 'NE': -90, 'SE': 180, 'SW': 90}
    need_trans = (-zh == np.cross(unit_vs[xplus], unit_vs[yplus])).all()
    if need_trans:
        i, j = j, i
    return matrix_rotate_indices(N, i, j, deg_ccw[origin])


def verify_directions(origin, xplus, yplus):
    if (origin[0] not in ('N', 'S')) or (origin[1] not in ('E', 'W')):
        raise ValueError('Parameter "origin" must be "(N|S)(E|W)" w/o parens.')
    total = origin + xplus + yplus
    for c in 'NESW':
        if total.count(c) != 1:
            emsg = 'xplus and yplus must use the two directions not in origin'
            raise ValueError(emsg)


def matrix_rotate_indices(N, i, j, deg_ccw):
    N -= 1
    d = {0: (i, j), 90: (N - j, i), -90: (j, N - i), 180: (N - i, N - j)}
    try:
        return d[deg_ccw]
    except KeyError:
        raise ValueError('Parameter deg_ccw must be one of (0, 90, -90, 180).')

###############################################################################
# Begin Execution
###############################################################################

if __name__ == '__main__':

    plt.ion()

    d = docopt(__doc__)
    print(d)

    # Parse command line args
    mode = 'scan' if d['plotscan'] else 'arb'
    data_path = d['--dir']
    pattern = d['--pattern']
    depth = int(d['--depth'])
    savefilename = d['--output']
    savefilename = 'out.txt' if savefilename is None else savefilename
    orien = d['--orientation'].upper()
    origin, xplus, yplus = orien[:2], orien[2], orien[3]

    # Get the list of files with hysteresis loop data in them
    # data_path = "/Users/nikolaj/Desktop/rzchowski/161016/0"
    # data_path = r"C:\Users\rzchlab\Desktop\trial1_5x5_BFO_test_sample\trial1_5x5_BFO_test_sample"
    datafiles = get_data_files(mode, data_path, pattern=pattern, depth=depth)

    #savefilepath = join(data_path, "out.txt")
    #file = open(savefilepath, "w")
    hyst_param_labels = ("x", "y",
                         "left_Hc", "right_Hc",
                         "top_Mr", "bot_Mr",
                         "left_sat(B)", "left_sat(V)",
                         "right_sat(B)", "right_sat(V)",
                         "left_slope", "right_slope",
                         "area")
    #headers = "".join("{:14}\t".format(s) for s in hyst_param_labels) + "\n"
    #file.write(headers)

    mx, my = get_grid_xy(mode, datafiles)
    fig, axarr = plt.subplots(mx, my)

    mrems = np.zeros((mx, my))
    hcs = np.zeros((mx, my))

    hyst_params = {k: [] for k in hyst_param_labels}

    for ax in axarr:
        for curr in ax:
            curr.spines["left"].set_visible(False)
            curr.spines["right"].set_visible(False)
            curr.spines["top"].set_visible(False)
            curr.spines["bottom"].set_visible(False)
            curr.xaxis.set_visible(False)
            curr.yaxis.set_visible(False)

    for q, df in enumerate(datafiles):
        print(df)
        B, V = np.loadtxt(df, usecols=(0, 1), unpack=True, delimiter='\t',
                          skiprows=2)

        # smooth data
        smoothed = [[], []]
        B = smoothed[0] = gaussian_filter(B, 20)
        V = smoothed[1] = gaussian_filter(V, 20)

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
        hc = Hc_of(B, V)
#        moreaccuratemrem = better_Mrem_of(B,V)
#        moreaccuratehc = better_Hc_of(B,V)

        # write to output file
        if mode == 'scan':
            x, y = get_xy(df)
        elif mode == 'arb':
            n = mx
            x, y = q // n, q % n

        hyst_params['x'].append(x)
        hyst_params['y'].append(y)
        hyst_params['left_Hc'].append(B[lincept])
        hyst_params['right_Hc'].append(B[rincept])
        hyst_params['top_Mr'].append(V[mrem_inds[0]])
        hyst_params['bot_Mr'].append(V[mrem_inds[1]])
        hyst_params['right_sat(B)'].append(B[satright])
        hyst_params['right_sat(V)'].append(V[satright])
        hyst_params['left_sat(B)'].append(B[satleft])
        hyst_params['left_sat(V)'].append(V[satleft])
        hyst_params['area'].append(area)
        hyst_params['left_slope'].append(lslope)
        hyst_params['right_slope'].append(rslope)

        b = smoothed[0]
        v = smoothed[1]

        i_ax, j_ax = get_axarr_coords(origin, xplus, yplus, mx, x, y)
        ax = axarr[i_ax, j_ax]

        # Title is coords in scan mode and filename in arb mode
        if mode == 'scan':
            title = "{}, {}".format(x, y)
        elif mode == 'arb':
            title = "{:.35}".format(basename(df))
        ax.set_title(title, fontsize=8)

        # plot data
        graph_data = ax.plot(b, v, 'g', alpha=.7)
        ax.plot(b[satright], v[satright], 'ro', alpha=.5)
        ax.plot(b[satleft], v[satleft], 'bo', alpha=.5)
        # TODO: plot Hc, mRem(jji version)
        ax.plot(B[mrem_inds], V[mrem_inds], 'ks', alpha=0.5, ms=5, lw=3)

        mrems[int(x)][int(y)] = abs(V[mrem_inds[0]] - V[mrem_inds[1]]) / 2
        hcs[int(x)][int(y)] = abs(B[lincept] - B[rincept]) / 2

        # Plot tangent lines
        ts = width/1e6
        ltanx = [b[lincept]-ts, b[lincept]+ts]
        ltany = [v[lincept]-ts*lslope, v[lincept]+ts*lslope]
        rtanx = [b[rincept]-ts, b[rincept]+ts]
        rtany = [v[rincept]-ts*lslope, v[rincept]+ts*lslope]
        ax.plot(ltanx, ltany, 'b', linewidth=2)
        ax.plot(rtanx, rtany, 'r', linewidth=2)


    # plot hc and mrem
    legend_string_hc="hc\n"
    for y, line in enumerate(hcs):
        for x, val in enumerate(line):
            legend_string_hc += (str(x)+","+str(y)+":"+str(val)[0:4]+"\n")

    legend_string_mrem="mrem values\n"
    for y, line in enumerate(mrems):
        for x, val in enumerate(line):
            legend_string_mrem += (str(x)+","+str(y)+":"+str(val)[0:4]+"\n")

    hc_text = fig.text(.91, 0.1, legend_string_hc)
    mrem_text = fig.text(.91, 0.1, legend_string_mrem, visible = False)

    def switch(label):
        vals = []
        recolor = False

        if(label == 'hc'):
            vals = mrems
            check.labels[0].set_text('mrem')

            recolor = True
            hc_text.set_visible(False)
            mrem_text.set_visible(False)

            check.labels[1].set_text('show\nvalues')


        if(label == 'mrem'):
            vals = hcs
            check.labels[0].set_text('hc')
            recolor = True
            hc_text.set_visible(False)
            mrem_text.set_visible(False)

            check.labels[1].set_text('show\nvalues')

        if recolor:
            for y, line in enumerate(axarr):
                for x, curr in enumerate(line):
                        curr.set_axis_bgcolor(str(vals[x][y]/np.max(vals)))


        if(label == 'hide\nvalues'):
            hc_text.set_visible(False)
            mrem_text.set_visible(False)
            check.labels[1].set_text('show\nvalues')

        if(label == 'show\nvalues'):

            if(check.labels[0].get_text() == 'hc'):
                hc_text.set_visible(True)
            if(check.labels[0].get_text() == 'mrem'):
                mrem_text.set_visible(True)
            check.labels[1].set_text('hide\nvalues')

        plt.draw()

    rax = plt.axes([0.01, 0.4, 0.095, 0.15])
    check = CheckButtons(rax, ('hc', 'show\nvalues'),  (True, False))
    check.on_clicked(switch)
    switch('hc')

    savefilepath = join(data_path, savefilename)
    h0 = "{:1}\t{:1}\t".format(*hyst_param_labels[:2])
    header = h0 + "".join("{:14}\t".format(s) for s in hyst_param_labels[2:])
    savedata = np.array(list(hyst_params[col] for col in hyst_param_labels))
    fmts = ["%3d", "%.1d"] + ["%+.7e"] * (len(hyst_param_labels) - 2)
    np.savetxt(savefilepath, savedata.T, fmt=fmts, delimiter="\t",
               header=header)
