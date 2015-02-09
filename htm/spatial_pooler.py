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
        self.columns = pynn.Population(n_columns, pynn.IF_cond_exp,
                self.parameters.populations.columns.neurons
                )
        self.kill_switch = pynn.Population(1, pynn.IF_cond_exp,
                self.parameters.populations.inhibition.neurons
                )
       
        self.stimulus.record()
        self.columns.record()
        if self.parameters.config.record_traces:
            self.columns.record_v()
            self.kill_switch.record_v()

    def connect(self):
        """Setup connections between populations"""

        params = self.parameters.projections
       
        # generate weights with normal distributed jitter and set up stimulus 
        w = params.stimulus.weight + np.random.normal(0, params.stimulus.jitter, len(self.columns))
        stimulus_connector = pynn.OneToOneConnector(weights=w)
        pynn.Projection(self.stimulus, self.columns, stimulus_connector)

        # projection to accumulate/count the number of active columns
        accumulation_connector = pynn.AllToAllConnector(weights=params.accumulation.weight)
        pynn.Projection(self.columns, self.kill_switch, accumulation_connector)

        # projection to inhibit all columns
        inhibition_connector = pynn.AllToAllConnector(weights=params.inhibition.weight)
        pynn.Projection(self.kill_switch, self.columns, inhibition_connector, target='inhibitory')

        # forward inhibition
        forward_inhibition_connector = pynn.FixedProbabilityConnector(params.forward_inhibition.probability, weights=params.forward_inhibition.weight)
        pynn.Projection(self.stimulus, self.columns, forward_inhibition_connector, target='inhibitory')
       
        # calculate connectivity matrix
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

