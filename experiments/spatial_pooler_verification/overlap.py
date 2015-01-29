import numpy as np
import matplotlib.pyplot as plt

import pyNN.nest as pynn

from htm import patterns
from htm import sdr
from htm.spatial_pooler import SpatialPooler

# generate input spike trains
N_SOURCES = 2000

orig = np.zeros(N_SOURCES)
orig[np.random.choice(np.arange(N_SOURCES), 100, replace=False)] = 1

data = []
data.append(orig)

for overlap in np.linspace(0.0, 1.0, 21):
    for i in range(10):
        data.append(patterns.generate_pattern(orig.copy(), overlap))

duration = len(data) * 200.
data = np.array(data)

# setup and run simultion
pynn.setup(threads=4)

pooler = SpatialPooler()
pooler.feed(data)

pynn.run(duration)
(source_spikes, column_spikes) = pooler.get_spikes()
pynn.end()

# calculate overlaps
overlaps = np.ndarray((0, 2))
orig_representation = None
for i in range(len(data)):
    t_0 = i*200.
    t_1 = (i+1)*200.
    mask = (column_spikes[:,1] > t_0) & (column_spikes[:,1] < t_1)

    r = sdr.spikes_to_sdr(column_spikes[mask,:], 1000) 
    
    if i == 0:
        orig_representation = r
    else:
        overlaps = np.vstack([overlaps, np.array([
                sdr.overlap(orig, data[i]),
                sdr.overlap(orig_representation, r)
                ])])

# average overlaps
averaged_overlaps = np.ndarray((0, 3))
for o in np.unique(overlaps[:,0]):
    mask = (overlaps[:,0] == o)
    averaged_overlaps = np.vstack([
            averaged_overlaps,
            np.array([o, np.mean(overlaps[mask,1]), np.std(overlaps[mask,1])])
            ])

# plot overlaps
plt.errorbar(averaged_overlaps[:,0], averaged_overlaps[:,1], yerr=averaged_overlaps[:,2], fmt='.')

plt.xlim((0, 1))
plt.ylim((0, 1))

plt.xlabel("Input SDR Overlap")
plt.ylabel("Output SDR Overlap")

plt.savefig(__file__.replace('py', 'pdf'))
