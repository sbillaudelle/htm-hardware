import numpy as np
import matplotlib.pyplot as plt

activity = np.load('activity.npy')
active = np.load('active.npy')

plt.figure(figsize=(3.6, 2.2))
counts, bins, patches = plt.hist(activity, 61, (9.5, 70.5), lw=0, rwidth=0.9, label="All Columns")
plt.hist(activity[active], bins=bins, lw=0, rwidth=0.9, label="Active Columns")

np.save('activity.npy', activity)
np.save('active.npy', active)

plt.xlabel("Input Events")
plt.ylabel("\#")
plt.xlim((19.5, 60.5))

plt.legend(loc='upper right')

plt.savefig('activity.pgf', bbox_inches='tight')
