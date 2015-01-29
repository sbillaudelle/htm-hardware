import numpy as np

def sdr_to_spikes(data, rate, duration):
    train = np.ndarray((0, 2))
    for j in range(data.size):
        if data[j]:
            spikes = np.arange(0, duration, 1000. / rate)
            spikes += np.random.randn(spikes.size)
            train = np.vstack([train, np.vstack([np.ones(spikes.size)*j, spikes]).T])
    
    #for i in range(data.size):
    #    if data[i]:
    #        s = generate_poisson_spikes(30., 1000.)
    #        train = np.vstack([train, np.vstack([np.ones(s.size)*i, s]).T])
    
    return train


def spikes_to_sdr(spikes, size):
    sdr = np.zeros(size)
    for j in np.unique(spikes[:,0]):
        mask = spikes[:,0] == j
        if spikes[mask,1].size > 5:
            sdr[j] = 1
    return sdr


def overlap(a, b):
    s = np.sum(a.astype(np.bool) & b.astype(np.bool))
    a = max(np.sum(a), np.sum(b))
    return s/float(a)


if __name__ == '__main__':
    spikes = np.load('columns.npy')
    mask = spikes[:,1] < 200.
    print spikes_to_sdr(spikes[mask,:], 1000)
