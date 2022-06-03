import src.site_disclination as disc
import numpy as np
from sys import argv
from time import time
from datetime import datetime


def main(nx: int, nz: int, mass: float, half_model: bool, other_half: bool, spin: int):
    phs_mass = np.min(np.abs((mass - 3, mass - 1, mass + 1, mass + 3)))

    print("Calculating disclination_rho for mass = ", mass)
    fname = 'nx_{}_nz_{}_mass_{}_half_{}_other_{}_spin_{}'.format(nx, nz, mass, half_model, other_half, spin)

    # date = datetime.now().strftime("%Y%m%d-%H%M%S")
    # fname = date + '-' + fname

    disc.calculate_disclination_rho(nz, nx, mass, phs_mass, half_model, other_half, spin, use_gpu=True, fname=fname)


if __name__ == '__main__':
    # nx = int(argv[1])
    # nz = int(argv[2])
    # mass = float(argv[3])
    # half_model = bool(argv[4])
    # other_half = bool(argv[5])
    # spin = int(arvg[6])

    print('Running main')
    tic = time()
    main(int(argv[1]), int(argv[2]), float(argv[3]), bool(eval(argv[4])), bool(eval(argv[5])), int(eval(argv[6])))
    print(f'Done running main. Elapsed time: {time() - tic} s')
