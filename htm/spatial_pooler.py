#! /usr/bin/python2
# -*- coding: utf-8 -*-

import numpy as np
import addict

import pyNN.nest as pynn

class SpatialPooler(object):
    """Timing based implementation of a spatial pooler"""

    # default parameters for the spatial pooler
    defaults = addict.Dict()

    defaults.config.timestep = 50.
    defaults.config.input_size = 10000
    defaults.config.record_traces = False

    defaults.populations.columns.size = 1000
    defaults.populations.columns.neurons.e_rev_I = -65.0
    defaults.populations.columns.neurons.v_thresh = -50.0
    defaults.populations.columns.neurons.tau_m = 15.0
    defaults.populations.columns.neurons.tau_refrac = 5.0
    defaults.populations.columns.neurons.tau_syn_E = 6.0
    defaults.populations.columns.neurons.tau_syn_I = 10.0

    defaults.populations.inhibition.neurons.e_rev_I = -80.0
    defaults.populations.inhibition.neurons.v_reset = -70.0
    defaults.populations.inhibition.neurons.v_thresh = -60.0
    defaults.populations.inhibition.neurons.tau_m = 0.3
    defaults.populations.inhibition.neurons.tau_refrac = 5.0
    defaults.populations.inhibition.neurons.tau_syn_E = 1.0
    defaults.populations.inhibition.neurons.tau_syn_I = 5.0

    defaults.projections.stimulus.weight = 0.002
    defaults.projections.stimulus.jitter = 0.00001
    defaults.projections.accumulation.weight = 0.015
    defaults.projections.inhibition.weight = 0.3
    defaults.projections.forward_inhibition.probability = 0.05
    defaults.projections.forward_inhibition.weight = 0.00001

    def __init__(self, params={}):
        # merge parameters into defaults
        self.parameters = self.defaults.copy()
        self.parameters.update(params)

        # setup network
        self.create()
        self.connect()

    def create(self):
        """Create the neuron populations"""

        n_columns = self.parameters.populations.columns.size
        self.stimulus = pynn.Population(n_columns, pynn.SpikeSourceArray)

        # create compartments
        self.columns = pynn.Population(n_columns, pynn.IF_cond_exp, self.parameters.populations.columns.neurons)
        self.kill_switch = pynn.Population(1, pynn.IF_cond_exp, self.parameters.populations.inhibition.neurons)
       
        self.stimulus.record()
        self.columns.record()
        if self.parameters.config.record_traces:
            self.columns.record_v()
            self.kill_switch.record_v()

    def connect(self):
        """Setup connections between populations"""

        # connect populations
        params = self.parameters.projections

        stimulus_connector = pynn.OneToOneConnector(weights=params.stimulus.weight + np.random.normal(0, params.stimulus.jitter, len(self.columns)))
        pynn.Projection(self.stimulus, self.columns, stimulus_connector)

        accumulation_connector = pynn.AllToAllConnector(weights=params.accumulation.weight)
        pynn.Projection(self.columns, self.kill_switch, accumulation_connector)

        inhibition_connector = pynn.AllToAllConnector(weights=params.inhibition.weight)
        pynn.Projection(self.kill_switch, self.columns, inhibition_connector, target='inhibitory')

        forward_inhibition_connector = pynn.FixedProbabilityConnector(params.forward_inhibition.probability, weights=params.forward_inhibition.weight)
        pynn.Projection(self.stimulus, self.columns, forward_inhibition_connector, target='inhibitory')
        
        n_columns = self.parameters.populations.columns.size
        n_inputs = self.parameters.config.input_size
    
        self.connections = (np.random.uniform(0, 1, n_columns*n_inputs) > 0.80).reshape(len(self.columns), n_inputs).astype(np.int64)

    def compute(self, data):
        """Perform the actual computation"""

        timestep = self.parameters.config.timestep

        # generate spike train from given input vector data
        train = np.ndarray((0, 2))
        for i, d in enumerate(data):
            activity = np.dot(self.connections, d)
            for j in range(len(self.stimulus)):
                spikes = np.sort(np.random.normal(1.0 + i*timestep, 0.01, activity[j]))
                train = np.vstack([train, np.vstack([np.ones(spikes.size)*j, spikes]).T])

        # set stimulus
        for j, s in enumerate(self.stimulus):
            s.spike_times = train[train[:,0] == j,1]

        # run simulation
        pynn.run(timestep*len(data))

        # extract spikes and calculate activity
        spikes = self.columns.getSpikes()
        activity = []
        for i in range(len(data)):
            mask = (spikes[:,1] > i*timestep) & (spikes[:,1] < (i + 1)*timestep)
            active = np.unique(spikes[mask,0]).astype(np.int32)
            activity.append(active)
        return activity

    def get_spikes(self):
        """Extract spike times from sources as well as columns"""

        return (self.stimulus.getSpikes(), self.columns.getSpikes())

    def get_traces(self):
        """Extract traces from columns.
        Requires setting config.record_traces
        """

        assert self.parameters.config.record_traces
        return self.columns.get_v()

"""
print active.size
open('asd.csv', 'a').write("{0:d}\n".format(active.size))

# plot histogram
if args.plot_histogram:
    plt.figure(figsize=(8.0, 4.5))
    counts, bins, patches = plt.hist(activity, 61, (9.5, 70.5), lw=0, rwidth=0.9)
    plt.hist(activity[active], bins=bins, lw=0, rwidth=0.9)
    
    plt.xlabel("Presynaptic Events")
    plt.ylabel("\#")
    plt.xlim((9.5, 70.5))
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
"""
