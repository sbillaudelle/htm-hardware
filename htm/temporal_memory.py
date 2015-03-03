import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import addict

import pyNN.nest as pynn

from htm.neuron_model import iaf_4_cond_exp

TIMESTEP = 50.
STEPS = 6

pynn.setup()

class TemporalMemory(object):
    defaults = addict.Dict()

    defaults.config.timestep = 50.
    defaults.config.n_columns = 1
    defaults.config.n_cells = 3
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
    defaults.populations.soma.neurons.tau_syn_1 = 8.0
    defaults.populations.soma.neurons.tau_syn_2 = 6.0
    defaults.populations.soma.neurons.tau_syn_3 = 25.0
    
    defaults.projections.stimulus.jitter = 0.0002

    def __init__(self, params={}):
        # merge parameters into defaults
        self.parameters = self.defaults.copy()
        self.parameters.update(params)
        
        self.create()
        self.connect()

    def create(self):
        n = self.parameters.config.n_cells
        # input populations
        self.proximal_input = pynn.Population(1, pynn.SpikeSourceArray, {'spike_times': np.arange(STEPS)*TIMESTEP + 0.01})
        self.distal_input = pynn.Population(2*n, pynn.SpikeSourceArray)
        self.distal_input[0].spike_times = np.array([55.0, 55.1, 55.2, 55.3, 155.0, 155.1, 155.2, 155.3])
        self.distal_input[1].spike_times = np.array([5.0, 5.1, 5.2, 5.3, 155.0, 155.1, 155.2, 155.3, 205.0, 205.1, 205.2, 205.3])
        self.distal_input[3].spike_times = np.array([155.0, 155.1, 155.2, 155.3])

        # create compartments
        self.distal = pynn.Population(2*n, pynn.IF_cond_exp, self.parameters.populations.distal.neurons, structure=pynn.space.Line())
        self.inhibitory = pynn.Population(n, pynn.IF_cond_exp, self.parameters.populations.inhibitory.neurons, structure=pynn.space.Line())
        self.soma = pynn.Population(n, iaf_4_cond_exp, self.parameters.populations.soma.neurons, structure=pynn.space.Line())
        self.soma.initialize('V_m', -65.)
        
        self.soma.record()
        self.distal.record()
        self.inhibitory.record()

        if self.parameters.config.record_traces:
            self.soma._record('V_m')
            self.distal.record_v()
            self.inhibitory.record_v()

    def connect(self):
        # connect populations
        pynn.Projection(self.proximal_input, self.soma, pynn.AllToAllConnector(weights=0.08), target='SYN_1') # 1
        pynn.Projection(self.proximal_input, self.inhibitory, pynn.AllToAllConnector(weights=0.04)) # 2
        pynn.Projection(self.distal_input, self.distal, pynn.OneToOneConnector(weights=0.025))
        pynn.Projection(self.inhibitory, self.soma, pynn.DistanceDependentProbabilityConnector('d>=1', weights=0.2), target='SYN_2') # 4

        for i in range(self.parameters.config.n_cells):
            pynn.Projection(self.distal[i*2:(i + 1)*2], pynn.PopulationView(self.inhibitory, [i]), pynn.AllToAllConnector(weights=0.1), target='inhibitory') # 3
            pynn.Projection(self.distal[i*2:(i + 1)*2], pynn.PopulationView(self.soma, [i]), pynn.AllToAllConnector(weights=0.1), target='SYN_3') # 3


    def compute(self):
        pynn.run(STEPS*TIMESTEP)

        spikes_soma = self.soma.getSpikes()
        spikes_inhibitory = self.inhibitory.getSpikes()
        spikes_distal = self.distal.getSpikes()

        return (spikes_soma, spikes_inhibitory, spikes_distal)

    def get_traces(self):
        assert self.parameters.config.record_traces
        
        trace_soma = self.soma.recorders['V_m'].get()
        trace_inhibitory = self.inhibitory.get_v()
        trace_distal = self.distal.get_v()
        
        return (trace_soma, trace_inhibitory, trace_distal)

if __name__ == '__main__':
    a = 0.05
    b = 0.10
    c = 0.06

    params = addict.Dict()
    params.config.record_traces = True
    tm = TemporalMemory(params)

    spikes_soma, spikes_inhibitory, spikes_distal = tm.compute()
    trace_soma, trace_inhibitory, trace_distal = tm.get_traces()

    plt.figure(figsize=(6.2, 7.0))

    n = tm.parameters.config.n_cells

    for cell in range(n):
        grid = gs.GridSpec(3, 1)
        grid.update(bottom=a + c*(n-cell) + (n-cell)*(1 - (2*a + (n-1)*c))/n, top=a + c*(n-cell) + (n-cell+1)*(1 - (2*a + (n-1)*c))/n, hspace=b)

        for t in np.arange(TIMESTEP, (STEPS+1)*TIMESTEP, TIMESTEP):
            # plot prediction
            mask = (spikes_distal[:,0] == 2*cell) & (spikes_distal[:,1] > t - TIMESTEP) & (spikes_distal[:,1] < t)
            if spikes_distal[mask].size > 0:
                ax = plt.subplot(grid[0, 0])
                ax.axvspan(t - TIMESTEP, t, fc='lightyellow', alpha=1)
            
            mask = (spikes_distal[:,0] == 2*cell + 1) & (spikes_distal[:,1] > t - TIMESTEP) & (spikes_distal[:,1] < t)
            if spikes_distal[mask].size > 0:
                ax = plt.subplot(grid[0, 0])
                ax.axvspan(t - TIMESTEP, t, fc='lightyellow', alpha=1)
            
            # plot active cells
            mask = (spikes_soma[:,0] == cell) & (spikes_soma[:,1] > t - TIMESTEP) & (spikes_soma[:,1] < t)
            if spikes_soma[mask].size > 0:
                ax = plt.subplot(grid[2, 0])
                ax.axvspan(t - TIMESTEP, t, fc='lightgreen', alpha=1)
            
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

    plt.savefig('temporal_memory.pdf', bbox_inches='tight')
