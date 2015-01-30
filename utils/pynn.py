import sys
import pyNN.nest as pynn

def run(duration):
    t = 0
    while t < duration:
        pynn.run(duration / 100.)
        t = pynn.get_current_time()

        sys.stdout.write("\rRunning simulation... [{0:d}%]".format(int(round(100*t/duration))))
        sys.stdout.flush()

    sys.stdout.write("\rRunning simulation... [{0:d}%]\n".format(int(round(100*t/duration))))
