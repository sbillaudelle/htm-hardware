#! /usr/bin/python
# -*- coding: utf-8 -*-

import addict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pyNN.nest as pynn

from htm.temporal_memory import TemporalMemory

pynn.setup(threads=4)

a = 0.05
b = 0.10
c = 0.06

params = addict.Dict()
params.config.record_traces = True
tm = TemporalMemory(params)

distal_stimulus = {
    0: np.array([55.0, 55.1, 55.2, 55.3, 155.0, 155.1, 155.2, 155.3]),
    1: np.array([5.0, 5.1, 5.2, 5.3, 155.0, 155.1, 155.2, 155.3, 205.0, 205.1, 205.2, 205.3]),
    3: np.array([155.0, 155.1, 155.2, 155.3])
    }
active, predictive = tm.compute(np.zeros((6, 1)), distal=distal_stimulus)
trace_soma, trace_inhibitory, trace_distal = tm.get_traces()

plt.figure(figsize=(6.2, 7.0))

n = tm.parameters.config.n_cells

for cell in range(n):
    grid = gs.GridSpec(3, 1)
    grid.update(bottom=a + c*(n-cell) + (n-cell)*(1 - (2*a + (n-1)*c))/n, top=a + c*(n-cell) + (n-cell+1)*(1 - (2*a + (n-1)*c))/n, hspace=b)

    timestep = tm.parameters.config.timestep
    for i, prd in enumerate(predictive):
        t = i*timestep
        # plot prediction
        if cell in prd:
            ax = plt.subplot(grid[0, 0])
            ax.axvspan(t, t + timestep, fc='lightyellow', alpha=1)
        
    for i, act in enumerate(active):
        t = i*timestep
        # plot active cells
        if cell in act:
            ax = plt.subplot(grid[2, 0])
            ax.axvspan(t, t + timestep, fc='lightgreen', alpha=1)
        
        for i in range(3):
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
    ax.set_xlim((0, ax.get_xlim()[1]))

    ax = plt.subplot(grid[1, 0])
    ax.grid(False)
    ax.tick_params(
            axis='x',
            which='both',
            bottom='off',
            top='off',
            labelbottom='off'
            )
    ax.set_ylim((-72.0, -48.0))
    ax.set_ylabel("$V_\\text{inh}$ [\si{\milli\\volt}]", size=6.0)
    ax.plot(trace_inhibitory[trace_inhibitory[:,0] == cell,1], trace_inhibitory[trace_inhibitory[:,0] == cell,2])

    ax = plt.subplot(grid[2, 0])
    ax.grid(False)
    ax.set_ylim((-72.0, -48.0))
    if cell == n-1:
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
