import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import pyNN.nest as pynn

N = 3

TIMESTEP = 50.
STEPS = 6

pynn.setup()

# input populations
proximal_input = pynn.Population(1, pynn.SpikeSourceArray, {'spike_times': np.arange(STEPS)*TIMESTEP + 0.01})
distal_input = pynn.Population(N, pynn.SpikeSourceArray)
distal_input[0].spike_times = np.array([55.0, 55.1, 55.2, 55.3, 155.0, 155.1, 155.2, 155.3])
distal_input[1].spike_times = np.array([5.0, 5.1, 5.2, 5.3, 155.0, 155.1, 155.2, 155.3, 205.0, 205.1, 205.2, 205.3])

# create compartments
params_distal = {
        'v_rest': -65.0,
        'e_rev_E': 0.0,
        'e_rev_I': -80.0,
        'v_reset': -70.0,
        'v_thresh': -50.0,
        'tau_m': 30.0,
        'tau_refrac': 1,
        'tau_syn_E': 4.0,
        'tau_syn_I': 4.0
        }
distal = pynn.Population(N, pynn.IF_cond_exp, params_distal, structure=pynn.space.Line())

params_inhibitory = {
        'v_rest': -65.0,
        'e_rev_E': 0.0,
        'e_rev_I': -50.0,
        'v_reset': -65.0,
        'v_thresh': -55.0,
        'tau_m': 30.0,
        'tau_refrac': 1.0,
        'tau_syn_E': 5.0,
        'tau_syn_I': 15.0
        }
inhibitory = pynn.Population(N, pynn.IF_cond_exp, params_inhibitory, structure=pynn.space.Line())

params_soma = {
        'v_rest': -65.0,
        'e_rev_E': 0.0,
        'e_rev_I': -90.0,
        'v_reset': -70.0,
        'v_thresh': -50.0,
        'tau_m': 15.0,
        'tau_refrac': 5.0,
        'tau_syn_E': 8.0,
        'tau_syn_I': 6.0
        }
soma = pynn.Population(N, pynn.IF_cond_exp, params_soma, structure=pynn.space.Line())

# connect populations
pynn.Projection(proximal_input, soma, pynn.AllToAllConnector(weights=0.08)) # 1
pynn.Projection(proximal_input, inhibitory, pynn.AllToAllConnector(weights=0.03)) # 2
pynn.Projection(distal_input, distal, pynn.OneToOneConnector(weights=0.025))
pynn.Projection(distal, inhibitory, pynn.OneToOneConnector(weights=0.1), target='inhibitory') # 3
pynn.Projection(distal, soma, pynn.OneToOneConnector(weights=0.05), target='excitatory') # 3
pynn.Projection(inhibitory, soma, pynn.DistanceDependentProbabilityConnector('d>=1', weights=0.2), target='inhibitory') # 4

soma.record()
distal.record()
inhibitory.record()

soma.record_v()
distal.record_v()
inhibitory.record_v()

pynn.run(STEPS*TIMESTEP)

spikes_soma = soma.getSpikes()
spikes_inhibitory = inhibitory.getSpikes()
spikes_distal = distal.getSpikes()

trace_soma = soma.get_v()
trace_inhibitory = inhibitory.get_v()
trace_distal = distal.get_v()

a = 0.05
b = 0.10
c = 0.06

plt.figure(figsize=(6.2, 7.0))

for cell in range(N):
    grid = gs.GridSpec(3, 1)
    grid.update(bottom=a + c*(N-cell) + (N-cell)*(1 - (2*a + (N-1)*c))/N, top=a + c*(N-cell) + (N-cell+1)*(1 - (2*a + (N-1)*c))/N, hspace=b)

    for t in np.arange(TIMESTEP, (STEPS+1)*TIMESTEP, TIMESTEP):
        # plot prediction
        mask = (spikes_distal[:,0] == cell) & (spikes_distal[:,1] > t - TIMESTEP) & (spikes_distal[:,1] < t)
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
    ax.plot(trace_distal[trace_distal[:,0] == cell,1], trace_distal[trace_distal[:,0] == cell,2])

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
    if cell == N-1:
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
