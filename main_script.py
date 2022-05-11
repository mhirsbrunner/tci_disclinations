import src.disclination_ed as disc
import numpy as np
from sys import argv


def main(nx: int, nz: int, mass: float, half_model=True, other_half=False):
    phs_mass = np.min(np.abs((mass - 3, mass - 1, mass + 1, mass + 3)))

    print("Calculating disclination_rho for mass = ", mass)

    fname = 'model_half_{}_other_{}_mass_{}'.format(half_model, other_half, mass)

    disc.calculate_disclination_rho(nz, nx, mass, phs_mass, half_model, other_half, fname)


if __name__ == '__main__':
    # nx = int(argv[1])
    # nz = int(argv[2])
    # mass = float(argv[3])
    # half_model = bool(argv[4])
    # other_half = bool(argv[5])
    print('Running main')
    main(int(argv[1]), int(argv[2]), float(argv[3]), bool(argv[4]), bool(argv[5]))
    print('Done running main')
