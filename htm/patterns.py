import numpy as np

def generate_pattern(data, overlap):
    pattern = data.copy()

    n = int(round(np.sum(data) * (1 - overlap)))

    indices = np.arange(data.size)

    mask = (data == 1)
    pattern[np.random.choice(indices[mask], n, replace=False)] = 0
    pattern[np.random.choice(indices[np.invert(mask)], n, replace=False)] = 1

    return pattern

if __name__ == '__main__':
    from sdr import overlap

    data = np.zeros(1000)
    data[np.random.choice(np.arange(1000), 100, replace=False)] = 1

    print overlap(data, generate_pattern(data, 0.3))
