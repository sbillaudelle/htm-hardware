#! /usr/bin/python2
# -*- coding: utf-8 -*-

import sys
import numpy as np
import matplotlib.pyplot as plt
import addict
import argparse
import pyNN.nest as pynn

from htm import patterns
from htm import sdr
from htm.spatial_pooler import SpatialPooler

parser = argparse.ArgumentParser()
parser.add_argument('--stimulus-weight', type=float, default=0.002)
parser.add_argument('--forward-inhibition-weight', type=float, default=0.00008)
parser.add_argument('--accumulation-weight', type=float, default=0.017)

args = parser.parse_args()

# generate input spike trains
N_SOURCES = 10000

orig = np.zeros(N_SOURCES)
orig[np.random.choice(np.arange(N_SOURCES), 400, replace=False)] = 1

data = []
data.append(orig)

for overlap in np.linspace(0.0, 1.0, 21):
    for i in range(10):
        data.append(patterns.generate_pattern(orig.copy(), overlap))

data = np.array(data)

# setup and run simulation
pynn.setup(min_delay=0.01, timestep=0.01, threads=4)

params = addict.Dict()
params.config.timestep = 50.0
#params.projections.stimulus.weight = args.stimulus_weight
#params.projections.forward_inhibition.weight = args.forward_inhibition_weight
#params.projections.accumulation.weight = args.accumulation_weight
#params.populations.columns.neurons.tau_m = 20.0
#params.populations.columns.neurons.tau_syn_E = 8.0
pooler = SpatialPooler(params)

active = []
for i, a in enumerate(pooler.compute(data, learn=False)):
    active.append(a)
    sys.stdout.write("\rComputing… [{0}/{1}]".format(i + 1, len(data)))
    sys.stdout.flush()

sys.stdout.write("\n")
sys.stdout.flush()

pynn.end()

# calculate overlaps
overlaps = np.ndarray((0, 2))
orig_representation = None
for i in range(len(data)):
    r = np.zeros(1000)
    r[active[i]] = 1
    if i == 0:
        orig_representation = r
    else:
        overlaps = np.vstack([overlaps, np.array([
                sdr.overlap(orig, data[i]),
                sdr.overlap(orig_representation, r)
                ])])

r1 = np.zeros(1000)
r1[active[1]] = 1

r2 = np.zeros(1000)
r2[active[2]] = 1

c = 0
for i in active[1]:
    for j in active[2]:
        if i == j:
            c += 1
print c

print sdr.overlap(data[1], data[2])
print sdr.overlap(r1, r2)

# average overlaps
averaged_overlaps = np.ndarray((0, 3))
for o in np.unique(overlaps[:,0]):
    mask = (overlaps[:,0] == o)
    averaged_overlaps = np.vstack([
            averaged_overlaps,
            np.array([o, np.mean(overlaps[mask,1]), np.std(overlaps[mask,1])])
            ])

import time
np.save('overlap_{0}.npy'.format(int(time.time())), averaged_overlaps)

# plot overlaps
plt.figure(figsize=(6.2, 4.0))

plt.errorbar(averaged_overlaps[:,0], averaged_overlaps[:,1], yerr=averaged_overlaps[:,2], fmt='.')

plt.xlim((0, 1))
plt.ylim((0, 1))

plt.xlabel("input SDR overlap")
plt.ylabel("output SDR overlap")

plt.savefig(__file__.replace('py', 'pgf'), bbox_inches='tight', transparent=True)
plt.savefig(__file__.replace('py', 'pdf'), bbox_inches='tight')
