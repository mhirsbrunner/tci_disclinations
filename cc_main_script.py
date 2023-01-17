import src.disclination as disc
import numpy as np
from sys import argv
from time import time
from datetime import datetime


def main(nz: int, nx: int, mass: float, disc_type: str, half_sign: int, spin: int):
    phs_mass = np.min(np.abs((mass - 3, mass - 1, mass + 1, mass + 3)))

    print("Calculating disclination_rho for mass = ", mass)
    fname = 'nz_{}_nx_{}_mass_{}_type_{}_half_{}_spin_{}'.format(nz, nx, mass, disc_type, half_sign, spin)

    # date = datetime.now().strftime("%Y%m%d-%H%M%S")
    # fname = date + '-' + fname

    disc.calculate_disclination_rho(nz, nx, mass, phs_mass, disc_type=disc_type, half_sign=half_sign, spin=spin,
                                    use_gpu=False, fname=fname)


if __name__ == '__main__':

    print('Running main')
    tic = time()
    main(int(argv[1]), int(argv[2]), float(argv[3]), str(argv[4]), int(argv[5]), int(argv[6]))
    print(f'Done running main. Elapsed time: {time() - tic} s')
