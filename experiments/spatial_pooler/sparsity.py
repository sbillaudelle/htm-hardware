#! /usr/bin/python2
# -*- coding: utf-8 -*-

import sys
import addict
import numpy as np
import matplotlib.pyplot as plt
import argparse

import pyNN.nest as pynn

from htm.spatial_pooler import SpatialPooler

parser = argparse.ArgumentParser()
parser.add_argument('--stimulus-weight', type=float, default=0.002)
parser.add_argument('--forward-inhibition-weight', type=float, default=0.00008)
parser.add_argument('--accumulation-weight', type=float, default=0.017)

args = parser.parse_args()

# generate input spike trains
N_SOURCES = 10000

orig = np.zeros(N_SOURCES)
orig[np.random.choice(np.arange(N_SOURCES), 200, replace=False)] = 1

data = []
for sparsity in np.linspace(0.00, 0.04, 41):
    for i in range(5):
        d = np.zeros(N_SOURCES)
        d[np.random.choice(np.arange(N_SOURCES), N_SOURCES*sparsity, replace=False)] = 1
        data.append(d)

# setup and run simultion
pynn.setup(min_delay=0.01, timestep=0.01, threads=4)

params = addict.Dict()
params.config.timestep = 50.0
params.projections.stimulus.weight = args.stimulus_weight
params.projections.forward_inhibition.weight = args.forward_inhibition_weight
params.projections.accumulation.weight = args.accumulation_weight
params.populations.columns.neurons.tau_m = 20.0
params.populations.columns.neurons.tau_syn_E = 8.0
pooler = SpatialPooler(params)

active = []
for i, a in enumerate(pooler.compute(data)):
    active.append(a)
    sys.stdout.write("\rComputing… [{0}/{1}]".format(i + 1, len(data)))
    sys.stdout.flush()
sys.stdout.write("\n")
sys.stdout.flush()

pynn.end()

# calculate sparsity
sparsity = np.ndarray((0, 2))
for i in range(len(data)):
    r = np.zeros(1000)
    r[active[i]] = 1
    
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
plt.errorbar(averaged_sparsity[:,0]/float(N_SOURCES)*100, averaged_sparsity[:,1]/float(1000)*100, yerr=averaged_sparsity[:,2]/float(1000)*100, fmt='.')

plt.xlabel("input sparsity [\si{\%}]")
plt.ylabel("output sparsity [\si{\%}]")

plt.xlim((0, 4.0))
plt.ylim((0, 20))

import time
np.save('sparsity_{0}.npy'.format(int(time.time())), averaged_sparsity)

#plt.savefig(__file__.replace('py', 'pgf'), bbox_inches='tight', transparent=True)
plt.savefig(__file__.replace('py', 'pdf'), bbox_inches='tight')
#plt.savefig('sparsity_{0}_{1}_{2}.pdf'.format(args.stimulus_weight, args.forward_inhibition_weight, args.accumulation_weight), bbox_inches='tight')
