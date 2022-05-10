import numpy as np
import numpy.linalg as nlg
from numpy import pi

from scipy import linalg as slg

import matplotlib.pyplot as plt
from src.utils import add_colorbar

import itertools

from pathlib import Path
import pickle as pkl

from tqdm import tqdm

import src.utils as utils

from time import time

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


# Disclination Functions
def disclination_surface_indices(nx: int):
    """
    Returns a list of ones and zeros where ones indicate that the index corresponds to a surface site
    :param nx:
    :return:
    """
    # bottom surface
    surf_sites = np.ones(nx)

    # sides below disclination
    for ii in range(nx // 2 - 1):
        temp = np.concatenate((np.ones(1), np.zeros(nx - 2), np.ones(1)))
        surf_sites = np.concatenate((surf_sites, temp))

    # sides above disclination
    for ii in range(nx // 2 - 1):
        temp = np.concatenate((np.ones(1), np.zeros(nx // 2 - 1)))
        surf_sites = np.concatenate((surf_sites, temp))

    # top edge
    temp = np.ones(nx // 2)
    surf_sites = np.concatenate((surf_sites, temp))

    return surf_sites


def disclination_hamiltonian_blocks(nx: int, mass: float, phs_mass: float, half_model=False):
    if nx % 2 == 1:
        nx += 1

    ny = nx

    if half_model:
        gamma_xy = -1j * np.dot(gamma_x, gamma_y)
        u_4 = slg.expm(1j * pi / 4 * (gamma_xy + np.identity(4, dtype=complex)))

        h_onsite = mass * gamma_0
        h_phs_mass = phs_mass * gamma_5

        h_x = 1j / 2 * gamma_x + 1 / 2 * gamma_0
        h_y = 1j / 2 * gamma_y + 1 / 2 * gamma_0
        h_z = 1j / 2 * gamma_z + 1 / 2 * gamma_0

        norb = 4
    else:
        gamma_xy = -1j * np.dot(gamma_x, gamma_y)
        u_4 = slg.expm(1j * pi / 4 * (np.kron(gamma_xy, sigma_0) + np.kron(np.identity(4, dtype=complex), sigma_z)))

        h_onsite = mass * np.kron(gamma_0, sigma_z)
        h_phs_mass = phs_mass * np.kron(gamma_5, sigma_0)

        h_x = 1j / 2 * np.kron(gamma_x, sigma_0) + 1 / 2 * np.kron(gamma_0, sigma_z)
        h_y = 1j / 2 * np.kron(gamma_y, sigma_0) + 1 / 2 * np.kron(gamma_0, sigma_z)
        h_z = 1j / 2 * np.kron(gamma_z, sigma_0) + 1 / 2 * np.kron(gamma_0, sigma_z)

        norb = 8

    h_disc = np.dot(nlg.inv(u_4), h_y)  # hopping term across disclination

    n_tot = (3 * nx * ny) // 4

    h00 = np.zeros((n_tot * norb, n_tot * norb), dtype=complex)

    # Onsite Hamiltonian
    h00 += np.kron(np.identity(n_tot, dtype=complex), h_onsite)

    # PHS Breaking on all surfaces

    surf_sites = disclination_surface_indices(nx)
    h00 += np.kron(np.diag(surf_sites), h_phs_mass)

    # X-Hopping
    temp_x = np.zeros(0)
    for ii in range(ny // 2):
        temp_x = np.concatenate((temp_x, np.ones(nx - 1, dtype=complex), (0,)))
    for ii in range(ny // 2 - 1):
        temp_x = np.concatenate((temp_x, np.ones(nx // 2 - 1, dtype=complex), (0,)))
    temp_x = np.concatenate((temp_x, np.ones(nx // 2 - 1, dtype=complex)))

    hop_x = np.diag(temp_x, k=1)
    h00 += np.kron(hop_x, h_x) + np.kron(hop_x, h_x).conj().T

    # Y-Hopping
    temp_y_1 = np.concatenate((np.ones(nx * (ny // 2 - 1) + nx // 2, dtype=complex),
                               np.zeros(nx // 2 * (ny // 2 - 1), dtype=complex)))
    temp_y_2 = np.concatenate((np.zeros(nx * ny // 2, dtype=complex),
                               np.ones(nx // 2 * (ny // 2 - 1), dtype=complex)))

    hop_y = np.diag(temp_y_1, k=nx) + np.diag(temp_y_2, k=nx // 2)
    h00 += np.kron(hop_y, h_y) + np.kron(hop_y, h_y).conj().T

    # Disclination Hopping
    for ii in range(nx // 2):
        ind_1 = norb * (nx * (ny // 2 - 1) + nx // 2 + ii)
        ind_2 = norb * (nx * (ny // 2) + nx // 2 * (1 + ii) - 1)

        h00[ind_1:ind_1 + norb, ind_2:ind_2 + norb] += h_disc
        h00[ind_2:ind_2 + norb, ind_1:ind_1 + norb] += h_disc.conj().T

    h01 = np.kron(np.identity(n_tot, dtype=complex), h_z)

    return h00, h01


def calculate_disclination_ldos(nx: int, mass: float, phs_mass: float, energy_axis, eta: float,
                                fname='disclination_ldos', half_model=False):
    h00, h01 = disclination_hamiltonian_blocks(nx, mass, phs_mass, half_model)

    if half_model:
        norb = 4
    else:
        norb = 8

    n_tot = h00.shape[0] // norb

    # add surface phs-breaking mass, being careful not to add it twice on the hinges of the side boundaries
    surf_sites = disclination_surface_indices(nx)
    bulk_sites = np.ones(n_tot, dtype=complex) - surf_sites
    if half_model:
        surf_pert = phs_mass * np.kron(np.diag(bulk_sites), gamma_5)
    else:
        surf_pert = phs_mass * np.kron(np.diag(bulk_sites), np.kron(gamma_5, sigma_0))

    ldos = np.zeros((len(energy_axis), n_tot))

    for ii, energy in enumerate(tqdm(energy_axis)):
        gs = utils.surface_green_function(energy + 1j * eta, h00, h01, surf_pert)
        a = utils.spectral_function(g=gs)
        temp_rho = np.diag(a)
        ldos[ii] = np.add.reduceat(temp_rho, np.arange(0, len(temp_rho), norb)) / (2 * pi)

    results = ldos
    params = (nx, mass, phs_mass, energy_axis, eta)
    data = (results, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)


def plot_disclination_rho(data_fname='disclination_ldos', save=True, fig_fname='disclination_rho', half_model=False):
    # TODO: Use networkx package to get coordinates of points for a disclination
    with open(data_dir / (data_fname + '.pickle'), 'rb') as handle:
        results, params = pkl.load(handle)

    nx, mass, phs_mass, energy_axis, eta = params
    de = energy_axis[1] - energy_axis[0]

    ldos = results

    # Calculate charge density
    rho = np.zeros(ldos.shape[1])

    for ii, energy in enumerate(energy_axis):
        if energy <= 0:
            rho += ldos[ii] * de

    # Subtract background charge and calculate the total charge (mod 8)
    if half_model:
        data = rho - 2
    else:
        data = rho - 4
    normalized_data = data / np.max(np.abs(data))
    vmax = np.max(np.abs(data))

    total_charge = np.sum(data)

    if half_model:
        modded_total_charge = (((total_charge * 8) % 1) / 8) * 16
    else:
        modded_total_charge = (((total_charge * 4) % 1) / 4) * 8

    # surf_sites = disclination_surface_indices(nx)
    # bulk_sites = np.ones_like(surf_sites) - surf_sites
    # bulk_charge = np.sum(np.multiply(rho, bulk_sites))
    # bulk_charge_mod_eight = (bulk_charge % 1) * 8

    # Diagnostic line plot
    plt.plot(data)
    plt.tight_layout()
    plt.show()

    # Generate list of lattice sites
    x = []
    y = []
    for site in itertools.product(range(nx), range(nx)):
        if site[0] < nx // 2 or site[1] < nx // 2:
            x.append(site[1])
            y.append(site[0])

    # Plot charge density
    fig, ax = plt.subplots(figsize=(6, 4))

    marker_scale = 250
    im = ax.scatter(x, y, s=marker_scale * np.abs(normalized_data), c=data, cmap='bwr', marker='o', alpha=0.7,
                    vmin=-vmax, vmax=vmax)
    ax.scatter(x, y, s=2, c='black')
    ax.set_aspect('equal')

    cbar = add_colorbar(im, aspect=15, pad_fraction=1.0)
    cbar.set_label(r'$\rho$')

    ax.margins(x=0.2)

    # ax.set_title(r'$\rho_0 = {:.2f}$, $Q = {:.2f}$, $Q_8 = {:.2f}$ (e/8)'.format(np.mean(rho), total_charge,
    # total_charge_over_eight))
    if half_model:
        ax.set_title(r'$Q = {:.4f}$, $Q_4 = {:.2f}$ (e/4)'.format(total_charge, modded_total_charge))
    else:
        ax.set_title(r'$Q = {:.4f}$, $Q_8 = {:.2f}$ (e/8)'.format(total_charge, modded_total_charge))

    plt.axis('off')
    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def plot_disclination_dos(data_fname='disclination_ldos', save=True, fig_fname='disclination_dos'):
    with open(data_dir / (data_fname + '.pickle'), 'rb') as handle:
        results, params = pkl.load(handle)

    nx, mass, phs_mass, energy_axis, eta = params
    e_pts = len(energy_axis)

    ldos = results
    dos = np.sum(ldos, axis=-1)

    plt.style.use(styles_dir / 'line_plot.mplstyle')

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(dos, energy_axis, 'k-')
    ax.fill_betweenx(energy_axis, np.zeros(e_pts), dos)

    ax.set_xlim(left=0, right=np.max(dos) * 1.1)
    ax.margins(y=0)

    ax.set_xticks((0, np.max(dos)))
    ax.set_xlabel('DOS')

    ax.set_ylabel('E (eV)')

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def main():
    half_model = True

    mass = 2
    phs_mass = 1

    nx = 40

    e_min = -7
    e_max = 0
    e_pts = 5
    energy_axis = np.linspace(e_min, e_max, e_pts)
    de = energy_axis[1] - energy_axis[0]

    eta = 0.5 * de

    calculate_disclination_ldos(nx, mass, phs_mass, energy_axis, eta, half_model=half_model)
    plot_disclination_dos()
    plot_disclination_rho(half_model=half_model)

    # Testing cupy
    # h00, h01 = disclination_hamiltonian_blocks(nx, mass, phs_mass, half_model=half_model)
    #
    # print(f'Matrix dimesions: {h00.shape}')
    #
    # n_tot = (3 * nx * nx) // 4
    # surf_sites = disclination_surface_indices(nx)
    # bulk_sites = np.ones(n_tot, dtype=complex) - surf_sites
    # if half_model:
    #     surf_pert = phs_mass * np.kron(np.diag(bulk_sites), gamma_5)
    # else:
    #     surf_pert = phs_mass * np.kron(np.diag(bulk_sites), np.kron(gamma_5, sigma_0))
    #
    # tic = time()
    # gs = utils.surface_green_function(-2 + 1j * eta, h00, h01, surf_pert)
    # print(f'numpy time = {time()-tic}')
    #
    # tic = time()
    # cp_gs = utils.cp_surface_green_function(-2 + 1j * eta, h00, h01, surf_pert)
    # print(f'cupy time = {time()-tic}')

if __name__ == '__main__':
    main()

    # results, params = utils.load_results(data_dir, 'disclination_ldos')
    # ldos = results
    # nx, mass, phs_mass, energy_axis, eta = params
