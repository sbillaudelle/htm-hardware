import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import addict
import pyNN.nest as pynn

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

stimulus = stimulus[:12]
labels = labels[:12]

X = 3
Y = (len(stimulus) + X - 1) / X

SAVE = True
fig = plt.figure(figsize=(6.6, 8.0))
if SAVE:
    grid = gs.GridSpec((len(stimulus) + 1)/X, X)
    grid.update(hspace=0.1)
else:
    plt.ion()
    ax = plt.gca()
    ax.set_xlim((-2, 65))
    ax.set_ylim((-2, 11))

for i, (l, s) in enumerate(zip(labels, stimulus)):
    if SAVE:
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

    print "stm", np.where(s)[0]
    
    if predictive is not None:
        # plot predictive cells
        x = predictive / 8
        y = predictive % 8
        ax.plot(x, y, '.', ms=12, alpha=0.4)
        print "prd", x
    else:
        ax.plot(np.array([]), np.array([]), '.')

    active, predictive = tm.compute(np.where(s))
    active = np.array(active[0], dtype=np.int16)
    predictive = np.array(predictive[0], dtype=np.int16)

    # plot active cells
    x = active / 8
    y = active % 8
    print "act", np.unique(x)
    ax.plot(x, y, '.', ms=7)
    
    if not SAVE:
        plt.pause(0.1)

if not SAVE:
    plt.pause(5)

if SAVE:
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

plt.savefig('live.pdf', bbox_inches='tight')
plt.savefig('live.pgf', bbox_inches='tight')
