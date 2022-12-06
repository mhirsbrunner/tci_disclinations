import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.ticker as plticker

import networkx as netx
import src.utils as utils
import src.disclination as disc

import numpy as np
from numpy import sin, cos, pi

from pathlib import Path
import pickle as pkl
from os import listdir
from os.path import isfile, join

import warnings

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

    dist = np.sqrt(np.sum((np.array(k1) - np.array(k0)) ** 2))
    nk = int(dist // dk)
    kx = np.linspace(k0[0], k1[0], nk)
    ky = np.linspace(k0[1], k1[1], nk)
    kz = np.linspace(k0[2], k1[2], nk)

    k_nodes.append(len(kx) - 1)

    for ii, k in enumerate(hsps[2:]):
        k0 = k1
        k1 = k

        dist = np.sqrt(np.sum((np.array(k1) - np.array(k0)) ** 2))
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


def disclination_graph(nx: int, disc_type='plaq'):
    graph = netx.Graph()

    if disc_type.lower() == 'plaq':
        if nx % 2 == 1:
            raise ValueError('Plaquette-centered disclinations must have nx even.')
        site_mod = 0
    elif disc_type.lower() == 'site':
        if nx % 2 == 0:
            raise ValueError('Site-centered disclinations must have nx odd.')
        site_mod = 1
    else:
        raise ValueError('Parameter "disc_type" must be either "plaq" or "site".')

    for ii in range(nx):
        for jj in range(nx):
            # x-hoppings
            if jj < nx // 2 + site_mod:
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
                if jj < nx // 2 - 1 + site_mod:
                    graph.add_edge((ii, jj), (ii, jj + 1))

    for ii in range(nx // 2):
        graph.add_edge((nx // 2 + ii + site_mod, nx // 2 - 1 + site_mod), (nx // 2 - 1, nx // 2 + ii + site_mod))

    pos = netx.kamada_kawai_layout(graph)

    return graph, pos


def plot_disclination_rho(layer_ind=None, z_half='bottom', data_fname='ed_disclination_ldos', save=True,
                          fig_fname='ed_disclination_rho', close_disc=True):
    results, params = utils.load_results(data_fname)
    nz, nx, mass, phs_mass, disc_type, half_sign, spin = params

    if layer_ind is not None:
        rho = results[layer_ind]
        layers = 1
    elif z_half.lower() == 'bottom':
        rho = np.sum(results[:nz // 2], axis=0)
        layers = nz // 2
    elif z_half.lower() == 'top':
        rho = np.sum(results[nz // 2:], axis=0)
        layers = nz // 2 + 1
    else:
        raise ValueError('Input "half" must specify "bottom" or "top" half of the system over which to sum the '
                         'density of states')

    print(rho)

    # Subtract background charge and calculate the total charge (mod 8)
    if half_sign is not None and half_sign != 0:
        data = rho - 2 * layers
        print(f'Total charge: {data.sum()}')
        print(f'Modded total charge: {(np.abs(rho.sum()) % (1 / 8)) * 8}')
    else:
        data = rho - 4 * layers
        print(f'Total charge: {data.sum()}')
        print(f'Modded total charge: {(np.abs(rho.sum()) % (1 / 4)) * 4}')

    normalized_data = data / np.max(np.abs(data))
    alpha_data = np.abs(normalized_data)

    # Generate list of lattice sites and positions
    x = []
    y = []
    graph, pos = disclination_graph(nx, disc_type)

    # Order the node list by the x index so the plot makes sense
    ordered_nodes = list(graph.nodes)
    ordered_nodes.sort(key=lambda s: s[1])

    for site in ordered_nodes:
        if close_disc:
            coords = pos[site]
        else:
            coords = site
        x.append(coords[0])
        y.append(coords[1])

    # Make colormap
    cmap = plt.cm.bwr
    my_cmap = cmap(np.arange(cmap.N // 2, cmap.N))
    my_cmap[:, -1] = np.linspace(0, 1, cmap.N // 2)
    my_cmap = ListedColormap(my_cmap)

    # Plot charge density
    fig, ax = plt.subplots(figsize=(6, 4))

    marker_scale = 250
    # im = ax.scatter(x, y, s=marker_scale * np.abs(normalized_data), c=np.abs(data), cmap=my_cmap, marker='o', vmin=0)
    dmax = np.max(np.abs(data))
    im = ax.scatter(x, y, s=marker_scale * np.abs(normalized_data), c=data, cmap='bwr', marker='o', vmax=dmax,
                    vmin=-dmax)
    ax.scatter(x, y, s=2, c='black')
    ax.set_aspect('equal')

    cbar = utils.add_colorbar(im, aspect=15, pad_fraction=1.0)
    cbar.ax.set_title(r'$|\rho|$', size=14)
    cbar.ax.tick_params(labelsize=14)

    ax.margins(x=0.2)

    plt.axis('off')
    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def plot_charge_per_layer(data_fname='ed_disclination_ldos', save=True, fig_fname='ed_disclination_rho_z', ylim=None):
    rho, params = utils.load_results(data_fname)
    nz, nx, mass, phs_mass, disc_type, half_sign, spin = params

    if half_sign is not None and half_sign != 0:
        norb = 2
    else:
        norb = 4

    data = np.sum(rho - norb, axis=1)

    plt.style.use(styles_dir / 'line_plot.mplstyle')

    fig, ax = plt.subplots(2, 1, figsize=(6, 8))
    ax[0].plot(np.arange(len(data) + 2), np.zeros(len(data) + 2), 'k--')
    ax[0].plot(np.arange(len(data)) + 1, np.zeros(len(data)), 'ro-', fillstyle='none', markersize=8, markeredgewidth=2)

    ax[1].plot(np.arange(len(data) + 2), np.zeros(len(data) + 2), 'k--')
    ax[1].plot(np.arange(len(data)) + 1, data, 'ro-', fillstyle='none', markersize=8, markeredgewidth=2)

    for axis in ax:
        axis.set_xticks((1, len(data)))

        axis.set_ylabel(r'$Q(z)$')
        axis.set_xlabel(r'$z$')

        if ylim is not None:
            axis.set_ylim((-ylim, ylim))

    # plt.margins(x=0.05)
    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


def plot_q_vs_mass(nz: int, nx: int, disc_type, half_sign, spinful: bool, data_folder_name=None, mod=True, save=True,
                   fig_fname="q_vs_mass", ylim=None):
    if half_sign is not None and half_sign != 0:
        norb = 2
    else:
        norb = 4

    mass_data = {}

    if data_folder_name is not None:
        data_location = data_dir / data_folder_name
    else:
        data_location = data_dir

    filenames = [f for f in listdir(data_location) if isfile(join(data_location, f))]

    for fname in filenames:
        with open(data_location / fname, 'rb') as handle:
            rho, params = pkl.load(handle)

        temp_nz, temp_nx, mass, phs_mass, temp_disc_type, temp_half_sign, spin = params

        if temp_nz != nz or temp_nx != nx or temp_disc_type != disc_type or temp_half_sign != half_sign:
            print('skip')
            continue

        # total_charge = np.sum(rho[:nz // 2] - norb)
        total_charge = disc.maissam_bound_charge(nz, nx, rho, norb)

        if spinful:
            if spin is None or spin == 0:
                continue
            mass_data[mass, spin] = total_charge
        else:
            if spin is not None and spin != 0:
                continue
            mass_data[mass] = total_charge

    if spinful:
        temp_data = {}

        for key in mass_data:
            # initialize spin-summed data dictionary
            temp_data[key[0]] = 0

            # check for missing spins missing their pairs
            if (key[0], -key[1]) not in mass_data:
                warnings.warn(f'Missing spin data for m={key[0]}!')

        # Sum mass data over spins
        for key in mass_data:
            temp_data[key[0]] += mass_data[key]

        mass_data = temp_data

    masses = []
    qs = []

    for key in mass_data:
        masses.append(key)
        qs.append(mass_data[key])

    qs = [x for _, x in sorted(zip(masses, qs))]
    masses.sort()

    plt.style.use(styles_dir / 'line_plot.mplstyle')
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(masses, qs, 'r.')

    for m in [-3, -1, 1, 3]:
        plt.axvline(x=m, color='k', linewidth=2, linestyle='--')

    ax.set_xlabel(r'$M$')
    ax.set_xticks((-3, -1, 0, 1, 3))

    ax.set_ylabel(r'$Q_0$')
    ax.grid(axis='y')

    if half_sign is not None and half_sign != 0:
        tick_increment = 1 / 16
    elif spinful:
        tick_increment = 1 / 4
    else:
        tick_increment = 1 / 8

    loc = plticker.MultipleLocator(base=tick_increment)
    ax.yaxis.set_major_locator(loc)

    if ylim is not None:
        ax.set_ylim(ylim)

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()


# def plot_q_vs_mass_summed_halves(data_folder_name: str, mod=True, save=True, fig_fname="q_vs_mass_summed_halves"):
#     norb = 2
#
#     masses_p = []
#     qs_p = []
#
#     masses_m = []
#     qs_m = []
#
#     filenames = [f for f in listdir(data_dir / data_folder_name) if isfile(join(data_dir / data_folder_name, f))]
#
#     for fname in filenames:
#         rho, params = utils.load_results(data_folder_name / fname)
#
#         nz, nx, mass, phs_mass, half_model, other_half = params
#
#         total_charge = np.sum(rho[:nz // 2] - norb)
#
#         if mod:
#             if half_model:
#                 modded_total_charge = (total_charge % (1 / 8)) * 8
#             else:
#                 modded_total_charge = (total_charge % (1 / 4)) * 4
#
#             temp_charge = modded_total_charge
#         else:
#             temp_charge = total_charge
#
#         if other_half:
#             masses_m.append(mass)
#             qs_m.append(temp_charge)
#         else:
#             masses_p.append(mass)
#             qs_p.append(temp_charge)
#
#     qs_p = [x for _, x in sorted(zip(masses_p, qs_p))]
#     qs_m = [x for _, x in sorted(zip(masses_m, qs_m))]
#     masses = masses_m.sort()
#
#     qs = [a + b for a, b in zip(qs_p, qs_m)]
#
#     plt.style.use(styles_dir / 'line_plot.mplstyle')
#     fig, ax = plt.subplots(figsize=(6, 4))
#
#     # ax.plot(masses, qs, 'o--', mec='red', mfc='red', color='k')
#     ax.plot(masses, qs, 'r.')
#
#     ax.set_xlabel(r'$M$')
#     ax.set_xticks((-3, -1, 0, 1, 3))
#     for m in [-3, -1, 1, 3]:
#         plt.axvline(x=m, color='k', linewidth=2, linestyle='--')
#
#     ax.set_ylabel(r'$Q(M)$')
#     if mod:
#         ax.set_yticks((0.25, 0.125, 0.0))
#         ax.set_yticklabels(('1/4', '1/8', '0'))
#     else:
#         ax.set_yticks((0.375, 0.25, 0.125, 0.0, -0.125, -0.25))
#         ax.set_yticklabels(('3/8', '1/4', '1/8', '0', '-1/8', '-1/4'))
#
#     plt.tight_layout()
#
#     if save:
#         plt.savefig(figure_dir / (fig_fname + '.pdf'))
#         plt.savefig(figure_dir / (fig_fname + '.png'))
#
#     plt.show()


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
    # nz, mass, phs_mass, energy_axis, eta, ks, k_nodes, nnn = params
    nz, mass, phs_mass, energy_axis, eta, ks, k_nodes = params
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


def plot_defect_free_rho(fig_fname='defect_free_rho', save=True):
    data_fname = 'defect_free_rho'

    with open(data_dir / (data_fname + '.pickle'), 'rb') as handle:
        data = pkl.load(handle)

    results, params = data
    rho = 2 * (results - 4)
    nx, mass, hoti_mass = params

    print(f'Corner charge (0, 0, 0): {np.sum(rho[:5, :5, :5])}')
    print(f'Corner charge (1, 0, 0): {np.sum(rho[5:, :5, :5])}')
    print(f'Corner charge (0, 1, 0): {np.sum(rho[:5, 5:, :5])}')
    print(f'Corner charge (1, 1, 0): {np.sum(rho[5:, 5:, :5])}')
    print(f'Corner charge (0, 0, 1): {np.sum(rho[:5, :5, 5:])}')
    print(f'Corner charge (1, 0, 1): {np.sum(rho[5:, :5, 5:])}')
    print(f'Corner charge (0, 1, 1): {np.sum(rho[:5, 5:, 5:])}')
    print(f'Corner charge (1, 1, 1): {np.sum(rho[5:, 5:, 5:])}')

    dmax = np.max(np.abs(rho))

    x, y, z = rho.nonzero()

    cmap = plt.cm.bwr
    my_cmap = cmap(np.arange(0, cmap.N // 2))
    my_cmap[:, -1] = np.linspace(1, 0, cmap.N // 2)
    my_cmap = ListedColormap(my_cmap)

    plt.style.use(styles_dir / 'line_plot.mplstyle')

    fig = plt.figure(figsize=(6, 4))
    ax = plt.axes(projection='3d')

    ax.view_init(20, 60)

    p = ax.scatter(x, y, z, c=rho, cmap=my_cmap, vmax=0, vmin=-dmax)
    ax.grid(True)

    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_zticks(())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    cp = fig.colorbar(p)

    plt.tight_layout()

    if save:
        plt.savefig(figure_dir / (fig_fname + '.pdf'))
        plt.savefig(figure_dir / (fig_fname + '.png'))

    plt.show()
