import numpy as np
import matplotlib.pyplot as plt

import pyNN.nest as pynn

from htm import sdr
from htm.spatial_pooler import SpatialPooler
import utils.pynn

# generate input spike trains
N_SOURCES = 2000

orig = np.zeros(N_SOURCES)
orig[np.random.choice(np.arange(N_SOURCES), 100, replace=False)] = 1

data = []
for sparsity in np.linspace(0.00, 0.15, 21):
    for i in range(5):
        d = np.zeros(N_SOURCES)
        d[np.random.choice(np.arange(N_SOURCES), N_SOURCES*sparsity, replace=False)] = 1
        data.append(d)

duration = len(data) * 200.
data = np.array(data)

# setup and run simultion
pynn.setup(threads=4)

pooler = SpatialPooler()
pooler.feed(data)

utils.pynn.run(duration)
(source_spikes, column_spikes) = pooler.get_spikes()
pynn.end()

# calculate sparsity
sparsity = np.ndarray((0, 2))
for i in range(len(data)):
    t_0 = i*200.
    t_1 = (i+1)*200.
    mask = (column_spikes[:,1] > t_0) & (column_spikes[:,1] < t_1)

    r = sdr.spikes_to_sdr(column_spikes[mask,:], 1000) 
    
    sparsity = np.vstack([sparsity, np.array([
            np.sum(data[i]),
            np.sum(r)
            ])])

# average sparsity
averaged_sparsity = np.ndarray((0, 3))
for o in np.unique(sparsity[:,0]):
    mask = (sparsity[:,0] == o)
    averaged_sparsity = np.vstack([
            averaged_sparsity,
            np.array([o, np.mean(sparsity[mask,1]), np.std(sparsity[mask,1])])
            ])

# plot sparsity
plt.figure(figsize=(6.2, 4.0))
plt.errorbar(averaged_sparsity[:,0], averaged_sparsity[:,1], yerr=averaged_sparsity[:,2], fmt='.')

plt.xlabel("number of active columns")
plt.ylabel("number of active active input cells")

plt.savefig(__file__.replace('py', 'pgf'), bbox_inches='tight', transparent=True)
plt.savefig(__file__.replace('py', 'pdf'), bbox_inches='tight')
