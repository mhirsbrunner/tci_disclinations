import numpy as np
import numpy.linalg as nlg
from numpy import pi

import cupy as cp
import cupy.linalg as clg

from scipy import linalg as slg

import matplotlib.pyplot as plt
from src.utils import add_colorbar

import itertools

from pathlib import Path
import pickle as pkl

import networkx as netx

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


def disclination_graph(nx: int):
    graph = netx.Graph()

    # for ii in range(nx):
    #     for jj in range(nx):
    #         if jj < nx // 2 or ii < nx // 2:
    #             graph.add_node((ii, jj))

    for ii in range(nx):
        for jj in range(nx):
            # x-hoppings
            if jj < nx // 2:
                if ii < nx - 1:
                    graph.add_edge((ii, jj), (ii + 1, jj))
            else:
                if ii < nx // 2 - 1:
                    graph.add_edge((ii, jj), (ii + 1, jj))
            # y-hoppings
            if ii < nx // 2:
                if jj < nx - 1:
                    graph.add_edge((ii, jj), (ii, jj + 1))
            else:
                if jj < nx // 2 - 1:
                    graph.add_edge((ii, jj), (ii, jj + 1))

    for ii in range(nx // 2):
        graph.add_edge((nx // 2 + ii, nx // 2 - 1), (nx // 2 - 1, nx // 2 + ii))

    pos = netx.kamada_kawai_layout(graph)

    return graph, pos


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


def disclination_hamiltonian(nz: int, nx: int, mass: float, phs_mass: float, half_model=False):
    h00, h01 = disclination_hamiltonian_blocks(nx, mass, phs_mass, half_model)

    h = np.kron(np.identity(nz), np.array(h00))
    h += np.kron(np.diag(np.ones(nz - 1), k=1), h01)
    h += np.kron(np.diag(np.ones(nz - 1), k=-1), h01.conj().T)

    return h


def calculate_disclination_rho(nz: int, nx: int, mass: float, phs_mass: float, half_model=False,
                               fname='ed_disclination_ldos'):
    if half_model:
        norb = 4
    else:
        norb = 8

    h = cp.asarray(disclination_hamiltonian(nz, nx, mass, phs_mass, half_model))

    evals, evecs = clg.eigh(h)
    evals = evals.get()
    evecs = evecs.get()

    rho = np.zeros((nz, (3 * nx * nx) // 4))

    for ii, energy in enumerate(evals):
        if energy <= 0:
            wf = evecs[:, ii]
            temp_rho = np.reshape(np.multiply(np.conj(wf), wf), (nz, -1, norb))
            rho += np.sum(temp_rho, axis=-1).real

    results = rho
    params = (nz, nx, mass, phs_mass, half_model)
    data = (results, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)

    return rho


def plot_disclination_rho(half='bottom', data_fname='ed_disclination_ldos', save=True, fig_fname='ed_disclination_rho'):
    with open(data_dir / (data_fname + '.pickle'), 'rb') as handle:
        results, params = pkl.load(handle)

    nz, nx, mass, phs_mass, half_model = params

    if half.lower() == 'bottom':
        rho = np.sum(results[:nz // 2], axis=0)
    elif half.lower() == 'top':
        rho = np.sum(results[nz // 2:], axis=0)
    else:
        raise ValueError('Input "half" must specify "bottom" or "top" half of the system over which to sum the '
                         'density of states')

    # Subtract background charge and calculate the total charge (mod 8)
    if half_model:
        data = rho - 2 * nz // 2
    else:
        data = rho - 4 * nz // 2
    normalized_data = data / np.max(np.abs(data))
    vmax = np.max(np.abs(data))

    total_charge = np.sum(data)

    if half_model:
        modded_total_charge = (((total_charge * 8) % 1) / 8) * 16
    else:
        modded_total_charge = (((total_charge * 4) % 1) / 4) * 8

    # Generate list of lattice sites
    x = []
    y = []
    # for site in itertools.product(range(nx), range(nx)):
    #     if site[0] < nx // 2 or site[1] < nx // 2:
    #         x.append(site[1])
    #         y.append(site[0])
    graph, pos = disclination_graph(nx)

    for ii in range(nx):
        for jj in range(nx):
            if ii < nx // 2 or jj < nx // 2:
                site = pos[ii, jj]
                x.append(site[0])
                y.append(site[1])

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

    if half_model:
        ax.set_title(r'$Q = {:.4f}$, '.format(total_charge) + r'$Q_{16}$'
                     + r'$ = {:.2f}$ (e/16)'.format(modded_total_charge))
    else:
        ax.set_title(r'$Q = {:.4f}, $'.format(total_charge) + r'$Q_{8}$'
                     + r'$ = {:.2f}$ (e/8)'.format(modded_total_charge))

    plt.axis('off')
    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def plot_charge_per_layer(data_fname='ed_disclination_ldos', save=True, fig_fname='ed_disclination_rho_z'):
    with open(data_dir / (data_fname + '.pickle'), 'rb') as handle:
        rho, params = pkl.load(handle)

    nz, nx, mass, phs_mass, half_model = params

    if half_model:
        data = rho - 2
    else:
        data = rho - 4

    data = np.sum(data, axis=1)

    plt.style.use(styles_dir / 'line_plot.mplstyle')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(data)) + 1, np.zeros_like(data), 'k--')
    ax.plot(np.arange(len(data)) + 1, data, 'r-')

    ax.set_xticks((1, len(data) // 2, len(data)))

    ax.set_ylabel(r'$\rho(z)$ (e)')
    ax.set_xlabel(r'$z$')

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def main():
    half_model = True

    mass = 2
    phs_mass = 1

    nz = 16
    nx = 16

    # calculate_disclination_rho(nz, nx, mass, phs_mass, half_model)

    plot_disclination_rho('bottom')
    plot_charge_per_layer()


def abid_main():
    half_model = True

    # mass = [-4, -3.5, -2.5, -2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0, 2.5, 3.5, 4.0]
    mass = [2, ]

    phs_mass = []
    for m in mass:
        phs_mass.append(np.min(np.abs((m - 3, m - 1, m + 1, m + 3))))

    nz = 10
    nx = 10

    for ii in range(len(mass)):
        calculate_disclination_rho(nz, nx, mass[ii], phs_mass[ii], half_model, fname='half_model_abid_run_' + f'{ii}')

    plot_disclination_rho('bottom')
    plot_charge_per_layer()


if __name__ == '__main__':
    abid_main()

    # results, params = utils.load_results(data_dir, 'disclination_ldos')
    # ldos = results
    # nx, mass, phs_mass, energy_axis, eta = params
