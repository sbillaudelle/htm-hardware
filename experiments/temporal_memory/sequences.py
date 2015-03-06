#! /usr/bin/python2
# -*- coding: utf-8 -*-

"""Simple sequence prediction experiment with the LIF neuron based temporal
memory implementation.
Please note that you must execute the `extract.py` script first in order to
generate stimulus and connectivity files. Refer to this repository's README
for more information!
"""

import numpy as np
import addict
import pyNN.nest as pynn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--columns', type=int, default=128,
        help="number of columns")
parser.add_argument('--active-columns', type=int, default=8,
        help="number of active columns for each timestep")
parser.add_argument('--cells', type=int, default=8,
        help="number of HTM cells per column")
parser.add_argument('--steps', type=int, default=12,
        help="number of simulation steps")
parser.add_argument('--save', action='store_true',
        help="save an overview plot instead of showing it live")
parser.add_argument('--width', type=int, default=3,
        help="number of horizontal suplots")
args = parser.parse_args()

import matplotlib as mpl
if not args.save:
    mpl.use('GtkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

from htm.temporal_memory import TemporalMemory

class ConnectivityDataError(BaseException):
    pass

# set up PyNN
pynn.setup(threads=4)

# load stimulus and connectivity files
try:
    connections = np.load('connectivity.npy')
    stimulus = np.load('stimulus.npy')[:args.steps]
    labels = np.load('labels.npy')[:args.steps]
except IOError, e:
    raise ConnectivityDataError("Could not find connectivity and/or stimulus data. Please run the `extract.py` script first!")

# initialize temporal memory
params = addict.Dict()
params.config.n_columns = args.columns
params.config.n_cells = args.cells

tm = TemporalMemory(params)
tm.set_distal_connections(connections)

if args.save:
    # calculate number of subplots
    width = args.width
    height = (len(stimulus) + width - 1) / width
    fig = plt.figure(figsize=(6.6, 8.0))
    grid = gs.GridSpec((len(stimulus) + 1)/width, width)
    grid.update(hspace=0.1)
else:
    # set up live plot
    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', lambda: quit())
    plt.ion()
    ax = plt.gca()
    ax.set_xlim((-2, 65))
    ax.set_ylim((-2, 11))

# run temporal memory and plot individual time steps
predictive = None
for i, (l, s) in enumerate(zip(labels, stimulus)):
    if args.save:
        # get suplot for current timestep
        ax = plt.subplot(grid[i%height, i/height])

    # clear plot and write label
    ax.cla()
    ax.text((-2 + 129)/2, 0, l.lower(), size=9, ha='center',
            bbox=dict(fc='white', ec='grey', alpha=0.5))

    # set up axis labeling
    if (not args.save) or (i%height == height - 1):
        ax.set_xlabel("Column Index")
    else:
        ax.tick_params(
                axis='x',
                which='both',
                bottom='off',
                top='off',
                labelbottom='off'
                )
    if (not args.save) or (i/height == 0):
        ax.set_ylabel("Cell Index")
    else:
        ax.tick_params(
                axis='y',
                which='both',
                left='off',
                right='off',
                labelleft='off'
                )
    ax.set_xlim((-2, 128 + 1))
    ax.set_ylim((-0.5, 8 -0.5))

    
    # plot predictive cells
    if predictive is not None:
        x = predictive / 8
        y = predictive % 8
        ax.plot(x, y, '.', ms=12, alpha=0.4)
    else:
        ax.plot(np.array([]), np.array([]), '.')

    # simulate next timestep
    active, predictive = tm.compute(np.where(s))
    active = np.array(active[0], dtype=np.int16)
    predictive = np.array(predictive[0], dtype=np.int16)

    # plot active cells
    x = active / 8
    y = active % 8
    ax.plot(x, y, '.', ms=7)
    
    if not args.save:
        plt.pause(0.1)

# plot the eye guiding connecting lines
if args.save:
    for i in range(len(stimulus) - 1):
        j = i + 1
        bbox_a = plt.subplot(grid[i%height, i/height]).get_window_extent().transformed(fig.transFigure.inverted())
        bbox_b = plt.subplot(grid[j%height, j/height]).get_window_extent().transformed(fig.transFigure.inverted())

        y0 = bbox_a.y0
        x0 = (bbox_a.x0 + bbox_a.x1)/2

        y1 = bbox_b.y1
        x1 = (bbox_b.x0 + bbox_b.x1)/2

        ys = (y0, y0 - 0.005,   y0 - 0.005,   y1 + 0.005, y1 + 0.005, y1)
        xs = (x0,         x0,  (x0 + x1)/2,  (x0 + x1)/2,         x1, x1)
        line = mpl.lines.Line2D(xs, ys, transform=fig.transFigure, zorder=-1, lw=4, c='lightgrey')
        fig.lines.append(line)
    
    plt.savefig('sequences.pdf', bbox_inches='tight')
    plt.savefig('sequences.pgf', bbox_inches='tight')
else:
    plt.ioff()
    plt.show(block=True)

