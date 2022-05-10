import joblib
import numpy as np
from numpy import sin, cos, pi

import matplotlib.pyplot as plt
from src.utils import add_colorbar

from pathlib import Path
import pickle as pkl

from tqdm import tqdm

import src.utils as utils

from joblib import Parallel, delayed

# File structure
project_src = Path(__file__).parent
project_root = project_src.parent
styles_dir = project_src / 'matplotlib_styles'
data_dir = project_root / 'data'
figure_dir = project_root / 'figures'

# Define Pauli and Gamma matrices for convenience
sigma_0 = np.array([[1, 0], [0, 1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

gamma_0 = np.kron(sigma_z, sigma_x)
gamma_x = np.kron(sigma_x, sigma_0)
gamma_y = np.kron(sigma_y, sigma_0)
gamma_z = np.kron(sigma_z, sigma_z)
gamma_5 = np.kron(sigma_z, sigma_y)


def calculate_surface_dos(dk: float, mass: float, phs_mass: float, energy_axis, eta=1e-6, fname='surf_dos'):
    ks, k_nodes = utils.high_symmetry_lines(dk)

    def parallel_temp(k):
        kx, ky = k

        sin_term = sin(kx) * gamma_x + sin(ky) * gamma_y
        mass_term = (mass + cos(kx) + cos(ky)) * gamma_0

        h00 = np.kron(sin_term, sigma_0) + np.kron(mass_term, sigma_z)
        h01 = -1j / 2 * np.kron(gamma_z, sigma_0) + 1 / 2 * np.kron(gamma_0, sigma_z)

        h_surf = phs_mass * np.kron(gamma_5, sigma_0)

        temp_surf_dos = np.zeros(len(energy_axis))

        for jj, energy in enumerate(energy_axis):
            gs = utils.surface_green_function(energy + 1j * eta, h00, h01, surf_pert=h_surf)

            a = utils.spectral_function(g=gs)
            temp_surf_dos[jj] = np.sum(np.diag(a)) / (2 * pi)

        return temp_surf_dos

    surf_dos = Parallel(n_jobs=joblib.cpu_count())(delayed(parallel_temp)(k) for k in ks)
    surf_dos = np.transpose(surf_dos)

    results = surf_dos
    params = (mass, phs_mass, energy_axis, eta, ks, k_nodes)
    data = (results, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)


def plot_surf_dos(data_fname='surf_dos', save=True, fig_fname='surf_gf_dos', vmax=None):
    with open(data_dir / (data_fname + '.pickle'), 'rb') as handle:
        results, params = pkl.load(handle)

    dos = results

    mass, phs_mass, energy_axis, eta, ks, k_nodes = params

    labels = (r'$Y$', r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$')

    plt.style.use(styles_dir / 'bands.mplstyle')

    fig, ax = plt.subplots(figsize=(6, 4))

    if vmax is None:
        im = ax.imshow(dos, origin='lower', cmap="magma", aspect='auto', vmin=0)
    else:
        im = ax.imshow(dos, origin='lower', cmap="magma", aspect='auto', vmin=0, vmax=vmax)
    add_colorbar(im, aspect=15, pad_fraction=1.0)

    ax.set_xticks(k_nodes)
    ax.set_xticklabels(labels)

    for k in k_nodes[1:-1]:
        ax.vlines(k, ymin=0, ymax=len(energy_axis) - 1, color='white', linewidth=1, linestyles='--')

    ax.set_ylabel('E (ev)')

    ax.set_yticks((-1, len(energy_axis)))
    ax.set_yticklabels((energy_axis[0], energy_axis[-1]))

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def plot_occupied_dos(data_fname='surf_dos', save=True, fig_fname='surf_gf_occ'):
    with open(data_dir / (data_fname + '.pickle'), 'rb') as handle:
        results, params = pkl.load(handle)

    dos = results

    mass, phs_mass, energy_axis, eta, ks, k_nodes = params
    de = energy_axis[1] - energy_axis[0]

    data = np.sum(dos, axis=0) * de

    labels = (r'$Y$', r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$')

    plt.style.use(styles_dir / 'bands.mplstyle')

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.set_xticks(k_nodes)
    ax.set_xticklabels(labels)

    ax.plot(data)

    ax.set_ylabel(r'$\rho(k)$')
    # ax.set_ylim([0, 4.1])

    ax.set_title(r'$|4-<\rho>| = {:.5f}$'.format(np.abs(4 - np.mean(data))))

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def main():
    mass = 2
    phs_mass = 0.5

    dk = 0.05

    e_min = -6
    e_max = 0
    e_pts = 500
    energy_axis = np.linspace(e_min, e_max, e_pts)
    de = energy_axis[1] - energy_axis[0]

    eta = 0.5 * de

    calculate_surface_dos(dk, mass, phs_mass, energy_axis, eta)
    # plot_surf_dos(vmax=10)
    plot_occupied_dos()


if __name__ == '__main__':
    main()
