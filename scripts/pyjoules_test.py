import pandas as pd
import numpy as np
import argparse
import datetime
import time
from pyJoules.energy_meter import measure_energy
from pyJoules.handler.csv_handler import CSVHandler

from pyJoules.energy_meter import measure_energy

@measure_energy
def foo(aa):
    a = np.fft.fft(aa)
    return a

def main():
    n_list = [10, 100, 1000]
    for n in n_list:
        print(n)
        aa = np.mgrid[0:n:1,0:n:1][0]
        foo(aa)

if __name__=='__main__':
   main()
