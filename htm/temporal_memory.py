#! /usr/bin/python
# -*- coding: utf-8 -*-

import addict
import numpy as np
import pyNN.nest as pynn

from htm.neuron_model import iaf_4_cond_exp

class TemporalMemory(object):
    defaults = addict.Dict()

    defaults.config.timestep = 50.
    defaults.config.n_columns = 1
    defaults.config.n_cells = 3
    defaults.config.n_segments = 2
    defaults.config.record_traces = False

    defaults.populations.distal.neurons.v_rest = -65.0
    defaults.populations.distal.neurons.e_rev_E = 0.0
    defaults.populations.distal.neurons.e_rev_I = -80.0
    defaults.populations.distal.neurons.v_reset = -70.0
    defaults.populations.distal.neurons.v_thresh = -50.0
    defaults.populations.distal.neurons.tau_m = 30.0
    defaults.populations.distal.neurons.tau_refrac = 1
    defaults.populations.distal.neurons.tau_syn_E = 4.0
    defaults.populations.distal.neurons.tau_syn_I = 4.0
    
    defaults.populations.inhibitory.neurons.v_rest = -65.0
    defaults.populations.inhibitory.neurons.e_rev_E = 0.0
    defaults.populations.inhibitory.neurons.e_rev_I = -55.0
    defaults.populations.inhibitory.neurons.v_reset = -65.0
    defaults.populations.inhibitory.neurons.v_thresh = -55.0
    defaults.populations.inhibitory.neurons.tau_m = 30.0
    defaults.populations.inhibitory.neurons.tau_refrac = 1.0
    defaults.populations.inhibitory.neurons.tau_syn_E = 5.0
    defaults.populations.inhibitory.neurons.tau_syn_I = 15.0
                
    defaults.populations.soma.neurons.E_L = -65.0
    defaults.populations.soma.neurons.E_syn_1 = 0.0
    defaults.populations.soma.neurons.E_syn_2 = -90.0
    defaults.populations.soma.neurons.E_syn_3 = -50.0
    defaults.populations.soma.neurons.V_reset = -70.0
    defaults.populations.soma.neurons.V_th = -50.0
    defaults.populations.soma.neurons.g_L = 16.667*4
    defaults.populations.soma.neurons.C_m = 250.0*4
    defaults.populations.soma.neurons.t_ref = 5.0
    defaults.populations.soma.neurons.tau_syn_1 = 9.0
    defaults.populations.soma.neurons.tau_syn_2 = 6.0
    defaults.populations.soma.neurons.tau_syn_3 = 20.0
    
    defaults.projections.stimulus.jitter = 0.0002

    def __init__(self, params={}):
        # merge parameters into defaults
        self.parameters = self.defaults.copy()
        self.parameters.update(params)
        
        self.create()
        self.connect()

    def create(self):
        """Create all cell populations"""

        n_columns = self.parameters.config.n_columns
        n_cells = self.parameters.config.n_cells
        n_segments = self.parameters.config.n_segments

        # input populations
        self.proximal_input = pynn.Population(n_columns, pynn.SpikeSourceArray)
        self.distal_input = pynn.Population(n_segments*n_columns*n_cells, pynn.SpikeSourceArray)

        # create compartments
        self.distal = pynn.Population(n_segments*n_columns*n_cells, pynn.IF_cond_exp, self.parameters.populations.distal.neurons, structure=pynn.space.Line())
        self.inhibitory = pynn.Population(n_columns*n_cells, pynn.IF_cond_exp, self.parameters.populations.inhibitory.neurons, structure=pynn.space.Line())
        self.soma = pynn.Population(n_columns*n_cells, iaf_4_cond_exp, self.parameters.populations.soma.neurons, structure=pynn.space.Line())
        self.soma.initialize('V_m', -65.)
        
        self.soma.record()
        self.distal.record()
        self.inhibitory.record()

        if self.parameters.config.record_traces:
            self.soma._record('V_m')
            self.distal.record_v()
            self.inhibitory.record_v()

    def connect(self):
        n_cells = self.parameters.config.n_cells
        n_segments = self.parameters.config.n_segments
        
        # connect populations
        pynn.Projection(self.distal_input, self.distal, pynn.OneToOneConnector(weights=0.025))

        for i in range(self.parameters.config.n_columns):
            # get "compartments" for all cells in this column
            inhibitions = self.inhibitory[i*n_cells:(i + 1)*n_cells]
            somas = self.soma[i*n_cells:(i + 1)*n_cells]
            proximal_input = pynn.PopulationView(self.proximal_input, [i])

            # set up connections with columnar symmetry
            pynn.Projection(inhibitions, somas,
                    pynn.DistanceDependentProbabilityConnector('d>=1', weights=0.2), target='SYN_2') # 4
            pynn.Projection(proximal_input, somas,
                    pynn.AllToAllConnector(weights=0.08), target='SYN_1') # 1
            pynn.Projection(proximal_input, inhibitions,
                    pynn.AllToAllConnector(weights=0.042)) # 2
            
            for j in range(self.parameters.config.n_cells):
                # get "compartments" for this specific cell
                segments = self.distal[i*n_cells*n_segments + j*n_segments:i*n_cells*n_segments + (j + 1)*n_segments]
                inhibition = pynn.PopulationView(self.inhibitory, [i*n_cells + j])
                soma = pynn.PopulationView(self.soma, [i*n_cells + j])

                # set up connections with cellular symmetry
                pynn.Projection(segments, inhibition,
                        pynn.AllToAllConnector(weights=0.1), target='inhibitory') # 3
                pynn.Projection(segments, soma,
                        pynn.AllToAllConnector(weights=0.15), target='SYN_3') # 3

    def set_distal_connections(self, connections):
        for src, tgt, sgm in connections:
            soma = pynn.PopulationView(self.soma, [int(src)])
            segment = pynn.PopulationView(self.distal, [2*int(tgt) + int(sgm)])
            pynn.Projection(soma, segment, pynn.OneToOneConnector(weights=0.014))

    def compute(self, proximal, distal=None):
        if distal is not None:
            for i, times in enumerate(distal):
                self.distal_input[i].spike_times = times

        active = []
        predictive = []

        if not (isinstance(proximal[0], list) or isinstance(proximal[0], np.ndarray)):
            proximal = [proximal]

        timestep = self.parameters.config.timestep

        for p in proximal:
            t = pynn.get_current_time()
            for c in p:
                self.proximal_input[int(c)].spike_times = np.array([t + 0.01])
            pynn.run(self.parameters.config.timestep)

            spikes_soma = self.soma.getSpikes()
            mask = (spikes_soma[:,1] >= t) & (spikes_soma[:,1] < t + timestep)
            active.append(np.unique(spikes_soma[mask,0]))

            spikes_distal = self.distal.getSpikes()
            mask = (spikes_distal[:,1] >= t) & (spikes_distal[:,1] < t + timestep)
            predictive.append(np.unique(spikes_distal[mask,0].astype(np.int16)/2))

        return (active, predictive)

    def get_traces(self):
        assert self.parameters.config.record_traces
        
        trace_soma = self.soma.recorders['V_m'].get()
        trace_inhibitory = self.inhibitory.get_v()
        trace_distal = self.distal.get_v()
        
        return (trace_soma, trace_inhibitory, trace_distal)
