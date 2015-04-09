#! /usr/bin/python2
# -*- coding: utf-8 -*-

"""Script for extracting the connectivity of the temporal pooler implementation
in NuPIC. A sample network will be presented with multiple sequences which will
be learned by the system. After the learning process, all realized connections
between presynaptic cells and postsynaptic dendritic segments will be dumped.
"""

import string
import numpy as np

from nupic.research.fast_temporal_memory import FastTemporalMemory as TemporalMemory
from nupic.bindings.algorithms import ConnectionsCell

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--columns', type=int, default=128,
        help="number of columns")
parser.add_argument('--active-columns', type=int, default=8,
        help="number of active columns for each timestep")
parser.add_argument('--cells', type=int, default=8,
        help="number of HTM cells per column")
parser.add_argument('--alphabet-size', type=int, default=256,
        help="size of the alphabet to be used")
parser.add_argument('--sequences', type=int, default=3,
        help="number of sequences to be fed into the network")
parser.add_argument('--sequence-size', type=int, default=3,
        help="length of the sequences")
parser.add_argument('--steps', type=int, default=64,
        help="number of simulation steps")
parser.add_argument('--live', action='store_true',
        help="show activity of the network in a live plot")
args = parser.parse_args()

if args.live:
    import matplotlib as mpl
    mpl.use('GtkAgg')
    import matplotlib.pyplot as plt

    plt.ion()

    plt.figure()
    ax = plt.gca()
    ax.set_xlim((-2, 65))
    ax.set_ylim((-2, 9))

# set up temporal memory
tm = TemporalMemory(
        columnDimensions=(args.columns,),
        cellsPerColumn=args.cells,
        minThreshold=args.cells - 1 - 8,
        activationThreshold=args.cells - 0 - 8,
        maxNewSynapseCount=args.cells - 8
        )

# generate random alphabet
alphabet = []
for i in range(args.alphabet_size):
    alphabet.append(np.zeros(args.alphabet_size, dtype=np.int16))
    alphabet[-1][np.random.choice(args.columns, args.active_columns, replace=False)] = 1

# generate sequences
sequences = []
for i in range(args.sequences):
    sequences.append([])
    for j in np.random.choice(len(alphabet), args.sequence_size, replace=False):
        sequences[-1].append(alphabet.pop(j))

# generate stimulus and labels for each timestep
stimulus = []
labels = []
for i in range(args.steps): # FIXME: step count is not correct
    if i > 2:
        c = np.random.choice(len(sequences))
    else:
        c = i
    sequence = sequences[c]
    for k, j in enumerate(sequence):
        stimulus.append(j)
        labels.append(string.ascii_uppercase[c] + str(k + 1))

    stimulus.append(alphabet[np.random.choice(len(alphabet))])
    labels.append(r"\textit{random}")

# iterate over stimulus and feed it into temporal memory
for l, c in zip(labels, stimulus):
    predictive = np.array([i.idx for i in tm.predictiveCells])
    tm.compute(set(np.where(c == 1)[0]))
    active = np.array([i.idx for i in tm.activeCells])

    if args.live:
        # generate a live updating plot
        ax.cla()
        ax.set_title(l)
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Cell Index")

        # plot predictive cells
        x = predictive / args.cells
        y = predictive % args.cells
        ax.plot(x, y, '.', ms=20)
        
        ax.set_xlim((-2, args.columns + 1))
        ax.set_ylim((-0.5, args.cells -0.5))
        
        # plot active cells
        x = active / args.cells
        y = active % args.cells
        ax.plot(x, y, '.', ms=16)

        plt.pause(0.5)

# extract connectivity from temporal pooler
connectivity = []
for i in range(args.columns):
    for j in range(args.cells):
        cell = ConnectionsCell(i*args.cells + j)
        segments = tm.connections.segmentsForCell(cell)
        for segment in segments:
            synapses = tm.connections.synapsesForSegment(segment)
            for synapse in synapses:
                data = tm.connections.dataForSynapse(synapse)
                if (data.permanence >= 0.5) and (data.destroyed == False):
                    src = data.presynapticCell.idx
                    tgt = cell.idx
                    sgm = str(segment).split('-')[-1]
                    connectivity.append([src, tgt, sgm])

# dump everything to disk
np.save('stimulus.npy', np.array(stimulus))
np.save('labels.npy', np.array(labels))
np.save('connectivity.npy', np.array(connectivity))
