#! /usr/bin/python2
# -*- coding: utf-8 -*-

# I have sinned. Please, have mercy on me, Guido! Forgive me for not complying to PEP 8!

import string
import numpy as np
import matplotlib as mpl
mpl.use('GtkAgg')
import matplotlib.pyplot as plt

N_COLUMNS = 128
N_ACTIVE_COLUMNS = 8
N_CELLS = 8

ALPHABET_SIZE = 128
SEQUENCE_SIZE = 3
STEPS = 64

PLOT = False

plt.ion()

plt.figure()
ax = plt.gca()
ax.set_xlim((-2, 65))
ax.set_ylim((-2, 9))

from nupic.research.fast_temporal_memory import FastTemporalMemory as TemporalMemory
from nupic.bindings.algorithms import ConnectionsCell

tm = TemporalMemory(
        columnDimensions=(N_COLUMNS,),
        cellsPerColumn=N_CELLS,
        minThreshold=N_CELLS - 1,
        activationThreshold=N_CELLS - 0,
        maxNewSynapseCount=N_CELLS
        )

alphabet = []
for i in range(ALPHABET_SIZE):
    alphabet.append(np.zeros(ALPHABET_SIZE, dtype=np.int16))
    alphabet[-1][np.random.choice(N_COLUMNS, N_ACTIVE_COLUMNS)] = 1

sequences = []
for i in range(SEQUENCE_SIZE):
    sequences.append([])
    for j in np.random.choice(len(alphabet), SEQUENCE_SIZE, replace=False):
        sequences[-1].append(alphabet.pop(j))

patterns = []
labels = []
for i in range(STEPS):
    c = np.random.choice(len(sequences))
    sequence = sequences[c]
    for k, j in enumerate(sequence):
        patterns.append(j)
        labels.append(string.ascii_uppercase[c] + str(k + 1))

    patterns.append(alphabet[np.random.choice(len(alphabet))])
    labels.append(r"\textit{random}")

for l, c in zip(labels, patterns):
    predictive = np.array([i.idx for i in tm.predictiveCells])
    tm.compute(set(np.where(c == 1)[0]))
    active = np.array([i.idx for i in tm.activeCells])

    if PLOT:
        ax.cla()
        ax.set_title(l)
        ax.set_xlabel("Column Index")
        ax.set_ylabel("Cell Index")

        # plot predictive cells
        x = predictive / N_CELLS
        y = predictive % N_CELLS
        print "stp", np.where(c)[0]
        print "prd", x

        ax.set_xlim((-2, N_COLUMNS + 1))
        ax.set_ylim((-0.5, N_CELLS -0.5))
        ax.plot(x, y, '.', ms=20)
        
        # plot active cells
        x = active / N_CELLS
        y = active % N_CELLS
        print "act", np.unique(x)

        ax.plot(x, y, '.', ms=16)

        plt.pause(0.01)

np.save('stimulus.npy', np.array(patterns))
np.save('labels.npy', np.array(labels))

connectivity = []
for i in range(N_COLUMNS):
    for j in range(N_CELLS):
        cell = ConnectionsCell(i*N_CELLS + j)
        segments = tm.connections.segmentsForCell(cell)
        for segment in segments:
            synapses = tm.connections.synapsesForSegment(segment)
            for synapse in synapses:
                data = tm.connections.dataForSynapse(synapse)
                if (data.permanence >= 0.5) and (data.destroyed == False):
                    src = data.presynapticCell.idx
                    tgt = cell.idx
                    sgm = str(segment).split('-')[-1]
                    #print "{0:4d} → {1:4d}.{2}".format(src, tgt, sgm)
                    connectivity.append([src, tgt, sgm])

np.save('connectivity.npy', np.array(connectivity))
