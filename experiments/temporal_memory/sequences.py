#! /usr/bin/python2
# -*- coding: utf-8 -*-

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
parser.add_argument('--save', action='store_true')
args = parser.parse_args()

import matplotlib as mpl
if not args.save:
    mpl.use('GtkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs


from htm.temporal_memory import TemporalMemory

pynn.setup(threads=4)

params = addict.Dict()
params.config.record_traces = False
params.config.n_columns = 128
params.config.n_cells = 8

connections = np.load('connectivity.npy')
stimulus = np.load('stimulus.npy')
labels = np.load('labels.npy')

tm = TemporalMemory(params)
tm.set_distal_connections(connections)

predictive = None

stimulus = stimulus[:args.steps]
labels = labels[:args.steps]

X = 3
Y = (len(stimulus) + X - 1) / X

if args.save:
    fig = plt.figure(figsize=(6.6, 8.0))
    grid = gs.GridSpec((len(stimulus) + 1)/X, X)
    grid.update(hspace=0.1)
else:
    fig = plt.figure()
    fig.canvas.mpl_connect('close_event', lambda: quit())
    plt.ion()
    ax = plt.gca()
    ax.set_xlim((-2, 65))
    ax.set_ylim((-2, 11))

for i, (l, s) in enumerate(zip(labels, stimulus)):
    if args.save:
        ax = plt.subplot(grid[i%Y, i/Y])
    ax.cla()
    ax.text((-2 + 129)/2, 0, l.lower(), size=9, ha='center', bbox=dict(fc='white', ec='grey', alpha=0.5))

    if i%Y == Y - 1:
        ax.set_xlabel("Column Index")
    else:
        ax.tick_params(
                axis='x',
                which='both',
                bottom='off',
                top='off',
                labelbottom='off'
                )
    if i/Y == 0:
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

    if predictive is not None:
        # plot predictive cells
        x = predictive / 8
        y = predictive % 8
        ax.plot(x, y, '.', ms=12, alpha=0.4)
    else:
        ax.plot(np.array([]), np.array([]), '.')

    active, predictive = tm.compute(np.where(s))
    active = np.array(active[0], dtype=np.int16)
    predictive = np.array(predictive[0], dtype=np.int16)

    # plot active cells
    x = active / 8
    y = active % 8
    ax.plot(x, y, '.', ms=7)
    
    if not args.save:
        plt.pause(0.1)

if args.save:
    for i in range(len(stimulus) - 1):
        j = i + 1
        bbox_a = plt.subplot(grid[i%Y, i/Y]).get_window_extent().transformed(fig.transFigure.inverted())
        bbox_b = plt.subplot(grid[j%Y, j/Y]).get_window_extent().transformed(fig.transFigure.inverted())

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

