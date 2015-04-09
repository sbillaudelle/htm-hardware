import os
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(3.6, 2.2))

for p in os.listdir('.'):
    if not p.startswith('sparsity') or not p.endswith('.npy'):
        continue
    
    activity = np.load(p)
    plt.errorbar(activity[:,0]/100., activity[:,1]/10., yerr=activity[:,2]/10., fmt='.-')

plt.xlabel("Input Sparsity [\si{\%}]")
plt.ylabel("Output Sparsity [\si{\%}]")

plt.xlim((0, 6.0))
plt.ylim((0, 6))

plt.savefig('sparsity.pgf', bbox_inches='tight')
plt.savefig('sparsity.pdf', bbox_inches='tight')
