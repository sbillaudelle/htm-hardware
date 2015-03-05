# Continuous-Time, Dynamic Implementation of Numenta's Hirarchical Temporal Memory Networks

## Experiments

### Spatial Pooler

### Temporal Memory

#### Sequences

The sequence experiment for the temporal memory implements a small LIF based temporal memory network with 128 columns and 8 cells each. However, no actual learning rules are implemented. Therefore, the connectivity of the model must be set externally.

At first, a NuPIC temporal memory instance is trained with a predifined stimulus. It's connectivity can be dumped using the `extract.py` script:

```bash
$ python2 extract.py
```

You will find the files `stimulus.npy`, `labels.npy`, and `connectivity.npy` in your directory. You might want to get familiar with the script's command line arguments by simply appending `--help` to the command string.

The actual simulation is initiated by running

```bash
$ python2 sequences.py
```

Again, check the command line help for options.
