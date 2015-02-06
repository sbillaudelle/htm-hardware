import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import argparse

import pyNN.nest as pynn

parser = argparse.ArgumentParser()
parser.add_argument('--columns', type=int, default=1000)
parser.add_argument('--inputs', type=int, default=10000)

parser.add_argument('--plot-histogram', action='store_true')
parser.add_argument('--plot-traces', action='store_true')

args = parser.parse_args()

pynn.setup(min_delay=0.01, timestep=0.01, threads=4)

# input populations
connections = (np.random.uniform(0, 1, args.columns*args.inputs) > 0.95).reshape(args.columns, args.inputs).astype(np.int64)
data = np.zeros(args.inputs)
data[np.random.choice(np.arange(args.inputs), int(round(args.inputs*0.02)), replace=False)] = 1

activity = np.dot(connections, data)
stimulus = pynn.Population(args.columns, pynn.SpikeSourceArray, {'spike_times': []})
for i, s in enumerate(stimulus):
    s.spike_times = np.sort(np.random.normal(1.0, 0.01, activity[i]))

# create compartments
params_columns = {
        'v_rest': -65.0,
        'e_rev_E': 0.0,
        'e_rev_I': -65.0,
        'v_reset': -65.0,
        'v_thresh': -50.0,
        'tau_m': 15.0,
        'tau_refrac': 5.0,
        'tau_syn_E': 6.0,
        'tau_syn_I': 10.0
        }
columns = pynn.Population(args.columns, pynn.IF_cond_exp, params_columns, structure=pynn.space.Line())

params_kill_switch = {
        'v_rest': -65.0,
        'e_rev_E': 0.0,
        'e_rev_I': -80.0,
        'v_reset': -70.0,
        'v_thresh': -60.0,
        'tau_m': 0.3,
        'tau_refrac': 5.0,
        'tau_syn_E': 1.0,
        'tau_syn_I': 5.0
        }
kill_switch = pynn.Population(1, pynn.IF_cond_exp, params_kill_switch)

# connect populations
pynn.Projection(stimulus, columns, pynn.OneToOneConnector(weights=0.007 + np.random.normal(0, 0.0001, args.columns)))
pynn.Projection(columns, kill_switch, pynn.AllToAllConnector(weights=0.015))
pynn.Projection(kill_switch, columns, pynn.AllToAllConnector(weights=0.3), target='inhibitory')
pynn.Projection(stimulus, columns, pynn.FixedProbabilityConnector(0.05, weights=0.00003), target='inhibitory')

columns.record()
if args.plot_traces:
    columns.record_v()
    kill_switch.record_v()

pynn.run(50.)

spikes = columns.getSpikes()
active = np.unique(spikes[:,0]).astype(np.int32)

print active.size
open('asd.csv', 'a').write("{0:d}\n".format(active.size))

# plot histogram
if args.plot_histogram:
    counts, bins, patches = plt.hist(activity, 26, (-0.5, 25.5), lw=0, rwidth=0.9)
    plt.hist(activity[active], bins=bins, lw=0, rwidth=0.9)

    plt.xlim((-0.5, 25.5))
    plt.savefig('activity.pdf')

if args.plot_traces:
    traces = columns.get_v()

    plt.figure(figsize=(6.2, 9.0))

    # plot kill switch
    grid = gs.GridSpec(1, 1)
    grid.update(top=0.95, bottom=0.85, hspace=0.05)

    trace = kill_switch.get_v()
    ax = plt.subplot(grid[0, 0])
    ax.grid(False)
    ax.set_title("Inhibitory Pool")
    ax.plot(trace[:,1], trace[:,2])

    choice = np.argsort(activity)[-active.size-3:-active.size+3]

    # plot columns
    grid = gs.GridSpec(choice.size, 1)
    grid.update(top=0.80, bottom=0.05, hspace=0.05)

    for i, col in enumerate(choice):
        ax = plt.subplot(grid[i, 0])
        ax.grid(False)
        ax.set_ylim((-66, -49))
        ax.set_yticks(np.linspace(-65, -50, 4))

        if i == 0:
            ax.set_title("Columns")
        
        if i == (choice.size - 1):
            ax.set_xlabel("$t$ [\si{\milli\second}]")
        else:
            ax.tick_params(\
                    axis='x',
                    which='both',
                    bottom='off',
                    top='off',
                    labelbottom='off')

        if spikes[spikes[:,0] == col,:].size > 0:
            ax.set_axis_bgcolor('lightgreen')

        mask = (traces[:,0] == col)
        ax.plot(traces[mask,1], traces[mask,2])
       
        ax.text(ax.get_xlim()[1] - 0.6, ax.get_ylim()[1] - 1, "{0}".format(activity[col]), va='top', ha='right')

    plt.savefig('spatial_pooler.pdf', bbox_inches='tight')
