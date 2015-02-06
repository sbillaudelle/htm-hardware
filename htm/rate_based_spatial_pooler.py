#! /usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pyNN.nest as pynn

import sdr

class SpatialPooler(object):
    def __init__(self):
        self.create()
        self.connect()

    def create(self):
        # set up columns
        self.columns = pynn.Population(1000, pynn.IF_cond_exp, {
                'v_rest': -65.0,
                'e_rev_E': 0.0,
                'e_rev_I': -80.0,
                'v_reset': -65.0,
                'v_thresh': -50.0,
                'tau_m': 30.0,
                'tau_refrac': 2.0,
                'tau_syn_E': 10.0,
                'tau_syn_I': 5.0
                },
                structure=pynn.space.Line())

        self.inhibitory_pool = pynn.Population(10, pynn.IF_cond_exp)

        # create input cells
        self.sources = pynn.Population(2000, pynn.SpikeSourceArray, {'spike_times': np.array([])})

        self.sources.record()
        self.columns.record()

    def connect(self):
        # connect populations
        self.injection = pynn.Projection(self.sources, self.columns,
                pynn.FixedProbabilityConnector(0.02, weights=0.0033*2),
                target='excitatory',
                rng=pynn.random.NumpyRNG(seed=5337)
                )
        self.recurrent_excitation = pynn.Projection(self.columns, self.columns,
                pynn.OneToOneConnector(weights=0.01*2),
                target='excitatory'
                )
        self.lateral_inhibition = pynn.Projection(self.columns, self.columns,
                pynn.DistanceDependentProbabilityConnector('d < 10', weights=0.004*2),
                target='inhibitory',
                rng=pynn.random.NumpyRNG(seed=5337)
                )
        self.global_inhibition = pynn.Projection(self.inhibitory_pool, self.columns,
                pynn.FixedProbabilityConnector(0.8, weights=0.0012*2),
                target='inhibitory',
                rng=pynn.random.NumpyRNG(seed=5337)
                )
        self.forward_inhibition = pynn.Projection(self.sources, self.inhibitory_pool,
                pynn.FixedProbabilityConnector(0.8, weights=0.0035),
                target='excitatory',
                rng=pynn.random.NumpyRNG(seed=5337)
                )

    def feed(self, data):
        train = np.ndarray((0, 2))

        for i, d in enumerate(data):
            spikes = sdr.sdr_to_spikes(d, 100., 160.)
            spikes[:,1] += 5 + i * 200.
            train = np.vstack([train, spikes])
        
        for j in range(len(self.sources)):
            self.sources[j].spike_times = train[train[:,0] == j,1]

    def get_spikes(self):
        source_spikes = self.sources.getSpikes()
        column_spikes = self.columns.getSpikes()

        return (source_spikes, column_spikes)

