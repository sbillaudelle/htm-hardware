#! /usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import addict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pyNN.nest as pynn

from htm.temporal_memory import TemporalMemory

parser = argparse.ArgumentParser()
parser.add_argument('--cells', type=int, default=3,
        help="number of cells in the simulated column")
parser.add_argument('--steps', type=int, default=8,
        help="number of timesteps to simulate")
parser.add_argument('--classify', action='store_true',
        help="automatically classify the firepatterns")
args = parser.parse_args()

pynn.setup(threads=4)

params = addict.Dict()
params.config.record_traces = True
params.config.n_cells = args.cells
tm = TemporalMemory(params)

predictions = [[]]

for ts in range(args.steps - 1):
    n = np.round(np.minimum(np.abs(np.random.normal(.8, 1)), args.cells)).astype(np.int16)
    predictions.append(np.random.choice(args.cells, n, replace=False))

distal_stimulus = []
for i in range(2*args.cells):
    distal_stimulus.append([])

for ts, p in enumerate(predictions[1:]):
    for c in p:
        t = ts*tm.parameters.config.timestep + 5
        distal_stimulus[2*c].extend(list(np.linspace(t, t+0.3, 4)))

active, predictive = tm.compute(np.zeros((args.steps, 1)), distal=distal_stimulus)
trace_soma, trace_inhibitory, trace_distal = tm.get_traces()

#plt.figure(figsize=(6.6, 7.0))
plt.figure(figsize=(16.0, 7.0))

a = 0.05
b = 0.10
c = 0.06

for cell in range(args.cells):
    grid = gs.GridSpec(3, 1)
    grid.update(bottom=a + c*(args.cells - cell) + (args.cells - cell)*(1 - (2*a + (args.cells - 1)*c))/args.cells, top=a + c*(args.cells-cell) + (args.cells -cell + 1)*(1 - (2*a + (args.cells - 1)*c))/args.cells, hspace=b)

    timestep = tm.parameters.config.timestep
    for i, prd in enumerate(predictive):
        t = i*timestep
        # plot prediction
        if cell in prd:
            ax = plt.subplot(grid[0, 0])
            ax.axvspan(t, t + timestep, fc='lightgrey', alpha=.5)
        
    for i, act in enumerate(active):
        t = i*timestep
        # plot active cells
        if cell in act:
            ax = plt.subplot(grid[2, 0])
            ax.axvspan(t, t + timestep, fc='lightgrey', alpha=.5)
      
        if args.classify:
            tp = (cell in act) and ((cell in predictions[i]) or (len(predictions[i]) == 0))
            tn = (cell not in act) and (len(predictions[i]) != 0) and (cell not in predictions[i])
            if tp or tn:
                ax.text(t + timestep/2, -70.0, "\color{green}\ding{51}", ha='center')
            else:
                ax.text(t + timestep/2, -70.0, "\color{red}\ding{55}", ha='center')
        
        for i in range(args.cells):
            ax = plt.subplot(grid[i, 0])
            ax.axvline(t, c='lightgrey', lw=0.5)


    ax = plt.subplot(grid[0, 0])
    ax.grid(False)
    ax.tick_params(
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off'
            )
    ax.set_title("Cell {0}".format(cell + 1), size=10.0)
    ax.set_ylim((-72.0, -48.0))
    ax.set_ylabel("$V_\\text{distal}$ [\si{\milli\\volt}]", size=6.0)
    ax.plot(trace_distal[trace_distal[:,0] == 2*cell,1], trace_distal[trace_distal[:,0] == 2*cell,2])
    ax.plot(trace_distal[trace_distal[:,0] == 2*cell + 1,1], trace_distal[trace_distal[:,0] == 2*cell + 1,2])
    ax.set_xlim((0, args.steps*timestep))

    ax = plt.subplot(grid[1, 0])
    ax.grid(False)
    ax.tick_params(
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off'
            )
    ax.set_xlim((0, args.steps*timestep))
    ax.set_ylim((-72.0, -48.0))
    ax.set_ylabel("$V_\\text{inh}$ [\si{\milli\\volt}]", size=6.0)
    ax.plot(trace_inhibitory[trace_inhibitory[:,0] == cell,1], trace_inhibitory[trace_inhibitory[:,0] == cell,2])

    ax = plt.subplot(grid[2, 0])
    ax.grid(False)
    ax.set_xlim((0, args.steps*timestep))
    ax.set_ylim((-72.0, -48.0))
    if cell == args.cells - 1:
        ax.set_xlabel("$t$ [\si{\milli\second}]", size=6.0)
    else:
        ax.tick_params(\
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom='off',      # ticks along the bottom edge are off
                top='off',         # ticks along the top edge are off
                labelbottom='off')
    ax.set_ylabel("$V_\\text{soma}$ [\si{\milli\\volt}]", size=6.0)
    ax.plot(trace_soma[trace_soma[:,0] == cell,1], trace_soma[trace_soma[:,0] == cell,2])
    ax.plot(trace_soma[trace_soma[:,0] == cell,1], trace_soma[trace_soma[:,0] == cell,2])

plt.savefig('temporal_memory_traces.pdf', bbox_inches='tight')
plt.savefig('temporal_memory_traces.pgf', bbox_inches='tight')
