import numpy as np
import numpy.linalg as nlg
from numpy import pi, ndarray

import cupy as cp
import cupy.linalg as clg

from scipy import linalg as slg

import matplotlib.pyplot as plt

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


def disclination_hamiltonian_blocks(nx: int, mass: float, phs_mass: float, half_model=False, other_half=False
                                    , nnn=False):
    if nx % 2 == 1:
        nx += 1

    ny = nx

    if other_half:
        sigma_factor = -1
    else:
        sigma_factor = 1

    if half_model:
        gamma_xy = -1j * np.dot(gamma_x, gamma_y)
        u_4 = slg.expm(1j * pi / 4 * (gamma_xy + sigma_factor * np.identity(4, dtype=complex)))

        h_onsite = sigma_factor * mass * gamma_0
        h_phs_mass = phs_mass * gamma_5

        h_x = 1j / 2 * gamma_x + 1 / 2 * gamma_0 * sigma_factor
        h_y = 1j / 2 * gamma_y + 1 / 2 * gamma_0 * sigma_factor
        h_z = 1j / 2 * gamma_z + 1 / 2 * gamma_0 * sigma_factor

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

    # Z-Hopping
    h01 = np.kron(np.identity(n_tot, dtype=complex), h_z)

    # NNN Stuff
    if nnn and half_model:
        print("Can't implement nnn hoppings in the half model.")
    elif nnn:
        print("Adding next-nearest neighbor hoppings to Hamiltonian")
        h_xz = -1 / 4 * np.kron(gamma_0, sigma_x)
        h_yz = -1 / 4 * np.kron(gamma_0, sigma_y)

        h01 += np.kron(hop_x, h_xz) + np.kron(-hop_x, h_xz)
        h01 += np.kron(hop_y, h_yz) + np.kron(-hop_y, h_yz)

    return h00, h01


def disclination_hamiltonian(nz: int, nx: int, mass: float, phs_mass: float, half_model=False, other_half=False
                             , nnn=False):
    h00, h01 = disclination_hamiltonian_blocks(nx, mass, phs_mass, half_model, other_half, nnn)

    h = np.kron(np.identity(nz), np.array(h00))
    h += np.kron(np.diag(np.ones(nz - 1), k=1), h01)
    h += np.kron(np.diag(np.ones(nz - 1), k=-1), h01.conj().T)

    return h


def calculate_disclination_rho(nz: int, nx: int, mass: float, phs_mass: float, half_model=False, other_half=False,
                               nnn=False, use_gpu=True, fname='ed_disclination_ldos'):
    if half_model:
        norb = 4
    else:
        norb = 8

    if use_gpu:
        print('Building Hamiltonian and sending to GPU')
        h = cp.asarray(disclination_hamiltonian(nz, nx, mass, phs_mass, half_model, other_half, nnn))

        print('Solving for eigenvectors and eigenvalues')
        evals, evecs = clg.eigh(h)
        evals = evals.get()
        evecs = evecs.get()
    else:
        print('Building Hamiltonian')
        h = disclination_hamiltonian(nz, nx, mass, phs_mass, half_model, other_half, nnn)

        print('Solving for eigenvectors and eigenvalues')
        evals, evecs = nlg.eigh(h)

    rho = np.zeros((nz, (3 * nx * nx) // 4))

    for ii, energy in enumerate(evals):
        if energy <= 0:
            wf = evecs[:, ii]
            temp_rho = np.reshape(np.multiply(np.conj(wf), wf), (nz, -1, norb))
            rho += np.sum(temp_rho, axis=-1).real

    results = rho
    params = (nz, nx, mass, phs_mass, half_model, other_half, nnn)
    data = (results, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)

    return rho
