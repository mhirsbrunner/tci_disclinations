import matplotlib.pyplot as plt
import networkx as netx
import src.utils as utils

import numpy as np
from numpy import sin, cos, pi

from pathlib import Path
import pickle as pkl
from os import listdir
from os.path import isfile, join

# File structure
project_src = Path(__file__).parent
project_root = project_src.parent
styles_dir = project_root / 'matplotlib_styles'
data_dir = project_root / 'data'
figure_dir = project_root / 'figures'


def plot_band_structure(dk, mass, phs_mass, nnn=False, save=True, fig_fname='bands'):
    sigma_0 = np.array([[1, 0], [0, 1]], dtype=complex)
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

    gamma_0 = np.kron(sigma_z, sigma_x)
    gamma_x = np.kron(sigma_x, sigma_0)
    gamma_y = np.kron(sigma_y, sigma_0)
    gamma_z = np.kron(sigma_z, sigma_z)
    gamma_5 = np.kron(sigma_z, sigma_y)

    def ham(kx, ky, kz):
        h = np.zeros((8, 8), dtype=complex)

        h += sin(kx) * np.kron(gamma_x, sigma_0)
        h += sin(ky) * np.kron(gamma_y, sigma_0)
        h += sin(kz) * np.kron(gamma_z, sigma_0)

        h += (mass + cos(kx) + cos(ky) + cos(kz)) * np.kron(gamma_0, sigma_z)

        h += phs_mass * np.kron(gamma_5, sigma_0)

        if nnn:
            h += sin(kx) * sin(kz) * np.kron(gamma_0, sigma_x)
            h += sin(ky) * sin(kz) * np.kron(gamma_0, sigma_y)

        return h

    # Build momentum slice
    gamma = (0, 0, 0)
    x = (pi, 0, 0)
    y = (0, pi, 0)
    m = (pi, pi, 0)
    r = (pi, pi, pi)

    hsps = (y, gamma, x, m, gamma, r)

    k_nodes = [0]

    k0 = hsps[0]
    k1 = hsps[1]

    dist = np.sqrt(np.sum((np.array(k1)-np.array(k0)) ** 2))
    nk = int(dist // dk)
    kx = np.linspace(k0[0], k1[0], nk)
    ky = np.linspace(k0[1], k1[1], nk)
    kz = np.linspace(k0[2], k1[2], nk)

    k_nodes.append(len(kx) - 1)

    for ii, k in enumerate(hsps[2:]):
        k0 = k1
        k1 = k

        dist = np.sqrt(np.sum((np.array(k1)-np.array(k0)) ** 2))
        nk = int(dist // dk)
        kx = np.concatenate((kx, np.linspace(k0[0], k1[0], nk + 1)[1:]))
        ky = np.concatenate((ky, np.linspace(k0[1], k1[1], nk + 1)[1:]))
        kz = np.concatenate((kz, np.linspace(k0[2], k1[2], nk + 1)[1:]))

        k_nodes.append(len(kx) - 1)

    ks = np.stack((kx, ky, kz), axis=1)

    evals = np.zeros((len(ks), 8))

    for ii, k in enumerate(ks):
        kx, ky, kz = k
        evals[ii] = np.linalg.eigvalsh(ham(kx, ky, kz))

    labels = (r'$Y$', r'$\Gamma$', r'$X$', r'$M$', r'$\Gamma$', r'$R$')

    plt.style.use(styles_dir / 'bands.mplstyle')

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(np.zeros(evals.shape[0]), 'k--')
    ax.plot(evals, 'b-')

    ax.set_xticks(k_nodes)
    ax.set_xticklabels(labels)

    for k in k_nodes[1:-1]:
        ax.axvline(k, 0, 1, color='black', linewidth=2, linestyle='-')

    ax.set_ylabel('Energy')

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()

    return


# TODO: Add code for site-centered disclinations
def disclination_graph(nx: int):
    graph = netx.Graph()

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


def plot_disclination_rho(z_half='bottom', data_fname='ed_disclination_ldos', save=True, fig_fname='ed_disclination_rho'):
    results, params = utils.load_results(data_fname)
    nz, nx, mass, phs_mass, disc_type, half_sign, spin = params

    if z_half.lower() == 'bottom':
        rho = np.sum(results[:nz // 2], axis=0)
    elif z_half.lower() == 'top':
        rho = np.sum(results[nz // 2:], axis=0)
    else:
        raise ValueError('Input "half" must specify "bottom" or "top" half of the system over which to sum the '
                         'density of states')

    # Subtract background charge and calculate the total charge (mod 8)
    if half_sign is not None and half_sign != 0:
        data = rho - 2 * nz // 2
        print(f'Modded total charge: {(rho.sum() % (1 / 8)) * 8}')
    else:
        data = rho - 4 * nz // 2
        print(f'Modded total charge: {(rho.sum() % (1 / 4)) * 4}')

    normalized_data = data / np.max(np.abs(data))
    # alpha_data = [np.min((x, 0.9)) for x in np.abs(normalized_data)]
    alpha_data = np.abs(normalized_data)

    # Generate list of lattice sites and positions
    x = []
    y = []
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
    im = ax.scatter(x, y, s=marker_scale * np.abs(normalized_data), c='red', marker='o',
                    alpha=alpha_data, vmin=0)
    # im = ax.scatter(x, y, s=marker_scale * np.abs(normalized_data), c=-normalized_data, cmap='magma', marker='o',
    #                 vmin=-1, vmax=1)
    ax.scatter(x, y, s=2, c='black')
    ax.set_aspect('equal')

    # cbar = utils.add_colorbar(im, aspect=15, pad_fraction=1.0)
    # cbar.ax.set_title(r'$|\rho|$', size=14)
    # cbar.ax.tick_params(labelsize=14)

    ax.margins(x=0.2)

    plt.axis('off')
    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def plot_charge_per_layer(data_fname='ed_disclination_ldos', save=True, fig_fname='ed_disclination_rho_z', ylim=None):
    rho, params = utils.load_results(data_fname)
    nz, nx, mass, phs_mass, half_model, other_half, nnn = params

    if half_model:
        norb = 2
    else:
        norb = 4

    data = np.sum(rho - norb, axis=1)

    plt.style.use(styles_dir / 'line_plot.mplstyle')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(data)) + 1, np.zeros_like(data), 'bo-')
    ax.plot(np.arange(len(data)) + 1, data, 'ro-', fillstyle='none', markersize=8, markeredgewidth=2)

    ax.set_xticks((1, len(data) // 2, len(data)))

    ax.set_ylabel(r'$Q(z)$')
    ax.set_xlabel(r'$z$')

    if ylim is not None:
        ax.set_ylim((-ylim, ylim))

    plt.margins(x=0.05)
    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def plot_q_vs_mass(data_folder_name: str, half_model: bool, other_half: bool, nnn: bool, mod=True, save=True,
                   fig_fname="q_vs_mass", ylim=None):
    if half_model:
        norb = 2
    else:
        norb = 4

    masses = []
    qs = []

    filenames = [f for f in listdir(data_dir / data_folder_name) if isfile(join(data_dir / data_folder_name, f))]

    for fname in filenames:
        with open(data_dir / data_folder_name / fname, 'rb') as handle:
            rho, params = pkl.load(handle)

        nz, nx, mass, phs_mass, temp_half_model, temp_other_half, temp_nnn = params

        if not(half_model == temp_half_model and other_half == temp_other_half and nnn == temp_nnn):
            continue

        masses.append(mass)

        total_charge = np.sum(rho[:nz // 2] - norb)

        if mod:
            if half_model:
                modded_total_charge = (total_charge % (1 / 8)) * 8
            else:
                modded_total_charge = (total_charge % (1 / 4)) * 4

            # This line is a cheat to get the plots to look the way Julian wants
            if modded_total_charge > 0.5:
                modded_total_charge = 1 - modded_total_charge

            qs.append(modded_total_charge)
        else:
            qs.append(total_charge)

    qs = [x for _, x in sorted(zip(masses, qs))]
    masses.sort()

    plt.style.use(styles_dir / 'line_plot.mplstyle')
    fig, ax = plt.subplots(figsize=(6, 4))

    # ax.plot(masses, qs, 'o--', mec='red', mfc='red', color='k')
    ax.plot(masses, qs, 'r.')

    for m in [-3, -1, 1, 3]:
        plt.axvline(x=m, color='k', linewidth=2, linestyle='--')

    ax.set_xlabel(r'$M$')
    ax.set_xticks((-3, -1, 0, 1, 3))

    ax.set_ylabel(r'$Q_0$')
    ax.grid(axis='y')

    if half_model:
        if mod:
            ax.set_yticks((1.0, 0.5, 0.0))
            ax.set_yticklabels(('1/8', '1/16', '0'))
        else:
            ax.set_yticks((0.1875, 0.125, 0.0625, 0.0, -0.0625, -0.125))
            ax.set_yticklabels(('3/16', '1/8', '1/16', '0', '-1/6'))
    else:
        if mod:
            ax.set_yticks((1.0, 0.5, 0.0))
            ax.set_yticklabels(('1/4', '1/8', '0'))
        else:
            ax.set_yticks((0.375, 0.25, 0.125, 0.0, -0.125, -0.25))
            ax.set_yticklabels(('3/8', '1/4', '1/8', '0', '-1/8', '-1/4'))

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def plot_q_vs_mass_summed_halves(data_folder_name: str, mod=True, save=True, fig_fname="q_vs_mass_summed_halves"):
    norb = 2

    masses_p = []
    qs_p = []

    masses_m = []
    qs_m = []

    filenames = [f for f in listdir(data_dir / data_folder_name) if isfile(join(data_dir / data_folder_name, f))]

    for fname in filenames:
        rho, params = utils.load_results(data_folder_name / fname)

        nz, nx, mass, phs_mass, half_model, other_half = params

        total_charge = np.sum(rho[:nz // 2] - norb)

        if mod:
            if half_model:
                modded_total_charge = (total_charge % (1 / 8)) * 8
            else:
                modded_total_charge = (total_charge % (1 / 4)) * 4

            temp_charge = modded_total_charge
        else:
            temp_charge = total_charge

        if other_half:
            masses_m.append(mass)
            qs_m.append(temp_charge)
        else:
            masses_p.append(mass)
            qs_p.append(temp_charge)

    qs_p = [x for _, x in sorted(zip(masses_p, qs_p))]
    qs_m = [x for _, x in sorted(zip(masses_m, qs_m))]
    masses = masses_m.sort()

    qs = [a + b for a, b in zip(qs_p, qs_m)]

    plt.style.use(styles_dir / 'line_plot.mplstyle')
    fig, ax = plt.subplots(figsize=(6, 4))

    # ax.plot(masses, qs, 'o--', mec='red', mfc='red', color='k')
    ax.plot(masses, qs, 'r.')

    ax.set_xlabel(r'$M$')
    ax.set_xticks((-3, -1, 0, 1, 3))
    for m in [-3, -1, 1, 3]:
        plt.axvline(x=m, color='k', linewidth=2, linestyle='--')

    ax.set_ylabel(r'$Q(M)$')
    if mod:
        ax.set_yticks((0.25, 0.125, 0.0))
        ax.set_yticklabels(('1/4', '1/8', '0'))
    else:
        ax.set_yticks((0.375, 0.25, 0.125, 0.0, -0.125, -0.25))
        ax.set_yticklabels(('3/8', '1/4', '1/8', '0', '-1/8', '-1/4'))

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def plot_open_z_bands(data_fname='open_z_bands', fig_fname='open_z_bands', save=True):
    results, params = utils.load_results(data_fname)

    evals = results
    nz, mass, phs_mass, ks, k_nodes, nnn = params

    labels = (r'$Y$', r'$\Gamma$', r'$X$', r'$M$')

    plt.style.use(styles_dir / 'bands.mplstyle')

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(np.zeros(len(ks)), 'k--')
    ax.plot(evals, 'b-')

    ax.set_xticks(k_nodes)
    ax.set_xticklabels(labels)

    for k in k_nodes[1:-1]:
        ax.axvline(k, 0, 1, color='black', linewidth=2, linestyle='-')

    ax.set_ylabel('Energy)')

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def plot_open_z_dos(data_fname='open_z_dos', fig_fname='open_z_dos', save=True, vmax=None):
    results, params = utils.load_results(data_fname)

    dos = results
    nz, mass, phs_mass, energy_axis, eta, ks, k_nodes, nnn = params

    labels = (r'$Y$', r'$\Gamma$', r'$X$', r'$M$')

    plt.style.use(styles_dir / 'bands.mplstyle')

    fig, ax = plt.subplots(figsize=(6, 4))

    if vmax is None:
        im = ax.imshow(dos, origin='lower', cmap="magma", aspect='auto', vmin=0)
    else:
        im = ax.imshow(dos, origin='lower', cmap="magma", aspect='auto', vmin=0, vmax=vmax)

    ax.set_xticks(k_nodes)
    ax.set_xticklabels(labels)

    for k in k_nodes[1:-1]:
        ax.vlines(k, ymin=0, ymax=len(energy_axis) - 1, color='white', linewidth=2, linestyles='--')

    ax.set_ylabel('Energy')

    ax.set_yticks((-1, len(energy_axis)))
    ax.set_yticklabels((energy_axis[0], energy_axis[-1]))

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()
