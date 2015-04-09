import os
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(3.6, 2.2))

reference = np.loadtxt('report.csv')

#mask = np.random.choice(np.arange(reference.shape[0]), 3000, replace=False)
#reference = reference[mask,:]

plt.plot(reference[:,0], reference[:,1], '.', alpha=1, color='lightgrey', label="reference data")

i = 1
for p in os.listdir('.'):
    if not p.startswith('overlap') or not p.endswith('.npy'):
        continue
    
    activity = np.load(p)
    plt.errorbar(activity[:,0], activity[:,1], yerr=activity[:,2], fmt='.-', label="simulation {0}".format(i), zorder=10)

    i += 1

plt.xlabel("Input Overlap")
plt.ylabel("Output Overlap")

plt.xlim((0.0, 1.0))
plt.ylim((0.0, 1.0))

plt.legend(loc='upper left')

plt.savefig('overlap.pgf', bbox_inches='tight')
plt.savefig('overlap.pdf', bbox_inches='tight')
