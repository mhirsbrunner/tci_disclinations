# TODO: Add documentation strings
import numpy as np
import numpy.linalg as nlg
from numpy import sin, cos, pi
from scipy import linalg as slg

from pathlib import Path
import pickle as pkl

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
def disclination_dimensions(nx: int, disc_type='plaq'):
    if disc_type.lower() == 'plaq':
        if nx % 2 != 0:
            raise ValueError('Lattice dimension must be an even integer for plaquette-centered disclinations.')

        bottom_width = nx
        top_width = nx // 2
        left_height = nx
        right_height = nx // 2

    elif disc_type.lower() == 'site':
        if nx % 2 != 1:
            raise ValueError('Lattice dimension must be an odd integer for site-centered disclinations.')

        bottom_width = nx
        top_width = nx // 2
        left_height = nx
        right_height = nx // 2 + 1

    else:
        raise ValueError('Disclination type must be "plaq" or "site".')

    return bottom_width, top_width, left_height, right_height


def side_surface_indices(nx: int, disc_type='plaq'):
    """
    Returns a list of ones and zeros where ones indicate that the index corresponds to a surface site
    :param nx: Number of sites along each axis
    :param disc_type: string specifying if the disclination is site- or plaquette-centered.
    :return: A list of zeros and ones, where ones indicate lattice sites on the side surfaces of the material
    """
    bottom_width, top_width, left_height, right_height = disclination_dimensions(nx, disc_type)

    # bottom surface
    surf_sites = np.ones(bottom_width)

    # sides below disclination
    for ii in range(right_height - 1):
        temp = np.concatenate((np.ones(1), np.zeros(bottom_width - 2), np.ones(1)))
        surf_sites = np.concatenate((surf_sites, temp))

    # sides above disclination
    for ii in range(left_height - right_height - 1):
        temp = np.concatenate((np.ones(1), np.zeros(top_width - 1)))
        surf_sites = np.concatenate((surf_sites, temp))

    # top edge
    temp = np.ones(top_width)
    surf_sites = np.concatenate((surf_sites, temp))

    return surf_sites


def number_of_sites(nx: int, disc_type='plaq'):
    return len(side_surface_indices(nx, disc_type))


def x_hopping_matrix(nx, disc_type='plaq', core_hopping=True):
    bottom_width, top_width, left_height, right_height = disclination_dimensions(nx, disc_type)

    x_hopping_sites = np.zeros(0)

    for ii in range(left_height - 1):
        if ii < right_height:
            x_hopping_sites = np.concatenate((x_hopping_sites, np.ones(bottom_width - 1, dtype=complex), (0,)))
        else:
            x_hopping_sites = np.concatenate((x_hopping_sites, np.ones(top_width - 1, dtype=complex), (0,)))

    x_hopping_sites = np.concatenate((x_hopping_sites, np.ones(top_width - 1, dtype=complex)))

    if not core_hopping and disc_type.lower() == 'site':
        x_hopping_sites[bottom_width * (right_height - 1) + top_width - 1] = 0
        x_hopping_sites[bottom_width * (right_height - 1) + top_width] = 0

    hop_mat = np.diag(x_hopping_sites, k=1)
    return hop_mat.T


def y_hopping_matrix(nx, disc_type='plaq', core_hopping=True):
    bottom_width, top_width, left_height, right_height = disclination_dimensions(nx, disc_type)

    y_hopping_sites_1 = np.concatenate((np.ones(bottom_width * (right_height - 1) + top_width, dtype=complex),
                                        np.zeros(top_width * (left_height - right_height - 1), dtype=complex)))
    y_hopping_sites_2 = np.concatenate((np.zeros(bottom_width * right_height, dtype=complex),
                                        np.ones(top_width * (left_height - right_height - 1), dtype=complex)))

    if not core_hopping and disc_type.lower() == 'site':
        y_hopping_sites_1[bottom_width * (right_height - 2) + top_width] = 0

    hop_mat = np.diag(y_hopping_sites_1, k=nx) + np.diag(y_hopping_sites_2, k=nx // 2)
    return hop_mat.T


def disclination_hopping_matrix(nx, disc_type='plaq'):
    bottom_width, top_width, left_height, right_height = disclination_dimensions(nx, disc_type)
    n_tot = number_of_sites(nx, disc_type)

    hopping_matrix = np.zeros((n_tot, n_tot))

    num_disc_sites = min(bottom_width - top_width, left_height - right_height)

    ind_1 = [bottom_width * right_height - ii - 1 for ii in range(num_disc_sites)]
    ind_2 = [n_tot - 1 - top_width * ii for ii in range(num_disc_sites)]

    for (ii, jj) in zip(ind_2, ind_1):
        hopping_matrix[ii, jj] = 1

    return hopping_matrix


def disclination_hamiltonian_blocks(nx: int, mass: float, phs_mass: float, disc_type='plaq', half_sign=None, spin=None,
                                    z_surface=False):
    if half_sign is not None:
        if half_sign != 1 and half_sign != -1 and half_sign != 0:
            raise ValueError('Parameter "half" must be either -1, 0, or 1')

    if spin is not None:
        if spin != 1 and spin != -1 and spin != 0:
            raise ValueError('Parameter "spin" must be either -1, 0, or 1')

    if (half_sign == 1 or half_sign == -1) and (spin == 1 or spin == -1):
        raise ValueError('Cannot implement spinful half model.')

    # Build Hamiltonian blocks
    if half_sign is None or half_sign == 0:
        gamma_xy = -1j * np.dot(gamma_x, gamma_y)

        if spin is None or spin == 0:
            u_4 = slg.expm(1j * pi / 4 * (np.kron(gamma_xy, sigma_0) + np.kron(np.identity(4, dtype=complex), sigma_z)))
        else:
            u_4 = slg.expm(1j * pi / 4 * (np.kron(gamma_xy, sigma_0) + np.kron(np.identity(4, dtype=complex), sigma_z)
                                          + spin * np.kron(np.identity(4, dtype=complex), sigma_0)))

        h_onsite = mass * np.kron(gamma_0, sigma_z)
        h_phs_mass = phs_mass * np.kron(gamma_5, sigma_0)

        h_x = -1j / 2 * np.kron(gamma_x, sigma_0) + 1 / 2 * np.kron(gamma_0, sigma_z)
        h_y = -1j / 2 * np.kron(gamma_y, sigma_0) + 1 / 2 * np.kron(gamma_0, sigma_z)
        h_z = -1j / 2 * np.kron(gamma_z, sigma_0) + 1 / 2 * np.kron(gamma_0, sigma_z)
        h_core = 1 / 2 * np.kron(gamma_0, sigma_z)

        norb = 8
    else:
        gamma_xy = -1j * np.dot(gamma_x, gamma_y)
        u_4 = slg.expm(1j * pi / 4 * (gamma_xy + half_sign * np.identity(4, dtype=complex)))

        h_onsite = half_sign * mass * gamma_0
        h_phs_mass = phs_mass * gamma_5

        h_x = -1j / 2 * gamma_x + 1 / 2 * gamma_0 * half_sign
        h_y = -1j / 2 * gamma_y + 1 / 2 * gamma_0 * half_sign
        h_z = -1j / 2 * gamma_z + 1 / 2 * gamma_0 * half_sign
        h_core = 1 / 2 * gamma_0 * half_sign

        norb = 4

    h_disc = np.dot(nlg.inv(u_4), h_y)

    # Arrange blocks into full Hamiltonian
    n_sites = number_of_sites(nx, disc_type)

    h00 = np.zeros((n_sites * norb, n_sites * norb), dtype=complex)

    # Onsite Hamiltonian
    h00 += np.kron(np.identity(n_sites, dtype=complex), h_onsite)

    # PHS Breaking on all surfaces
    if z_surface:
        phs_mass_sites = np.ones(n_sites)
    else:
        phs_mass_sites = side_surface_indices(nx, disc_type)

    h00 += np.kron(np.diag(phs_mass_sites), h_phs_mass)

    # X-Hopping
    if disc_type.lower() == 'plaq':
        x_hopping = x_hopping_matrix(nx, disc_type)
        h00 += np.kron(x_hopping, h_x) + np.kron(x_hopping, h_x).conj().T
    else:
        x_hopping = x_hopping_matrix(nx, disc_type, core_hopping=False)
        h00 += np.kron(x_hopping, h_x) + np.kron(x_hopping, h_x).conj().T

        # Add cos hoppings to the disclination core
        # core_hopping = x_hopping_matrix(nx, disc_type, core_hopping=False)
        # - x_hopping_matrix(nx, disc_type, core_hopping=True)
        # h00 += np.kron(core_hopping, np.dot(nlg.inv(u_4), h_core)) \
        #        + np.kron(core_hopping, np.dot(nlg.inv(u_4), h_core)).conj().T

    # Y-Hopping
    if disc_type.lower() == 'plaq':
        y_hopping = y_hopping_matrix(nx, disc_type)
        h00 += np.kron(y_hopping, h_y) + np.kron(y_hopping, h_y).conj().T
    else:
        y_hopping = y_hopping_matrix(nx, disc_type, core_hopping=False)
        h00 += np.kron(y_hopping, h_y) + np.kron(y_hopping, h_y).conj().T

        # Add cos hoppings to the disclination core
        # core_hopping = y_hopping_matrix(nx, disc_type, core_hopping=False)
        # - y_hopping_matrix(nx, disc_type, core_hopping=True)
        # h00 += np.kron(core_hopping, h_core) + np.kron(core_hopping, h_core).conj().T

    # Disclination Hopping
    disc_hopping = disclination_hopping_matrix(nx, disc_type)
    h00 += np.kron(disc_hopping, h_disc) + np.kron(disc_hopping, h_disc).conj().T

    # Z-Hopping
    h01 = np.kron(np.identity(n_sites, dtype=complex), h_z)

    # NNN Z- and Disclination-Hoppings
    if half_sign is None or half_sign == 0:
        h_xz = -1 / 4 * np.kron(gamma_0, sigma_x)
        h_yz = -1 / 4 * np.kron(gamma_0, sigma_y)
        h_disc_nnn_y = np.dot(nlg.inv(u_4), h_yz)
        h_disc_nnn_x = np.dot(h_xz, u_4)

        nnn_x_hopping = x_hopping_matrix(nx, disc_type, core_hopping=False)
        nnn_y_hopping = y_hopping_matrix(nx, disc_type, core_hopping=False)

        h01 += np.kron(nnn_x_hopping, h_xz) - np.kron(nnn_x_hopping.T, h_xz)
        h01 += np.kron(nnn_y_hopping, h_yz) - np.kron(nnn_y_hopping.T, h_yz)

        h01 += (np.kron(disc_hopping, h_disc_nnn_y) -
                np.kron(disc_hopping.T, h_disc_nnn_x))

    return h00, h01


def disclination_hamiltonian(nz: int, nx: int, mass: float, phs_mass: float, disc_type='plaq', half_sign=None,
                             spin=None):
    h00, h01 = disclination_hamiltonian_blocks(nx, mass, phs_mass, disc_type, half_sign, spin, z_surface=False)
    h00_surf, h01_surf = disclination_hamiltonian_blocks(nx, mass, phs_mass, disc_type, half_sign, spin, z_surface=True)

    surface_z_indices = np.concatenate(((1,), np.zeros(nz - 2), (1,)))
    bulk_z_indices = 1 - surface_z_indices

    h = np.kron(np.diag(surface_z_indices), np.array(h00_surf))
    h += np.kron(np.diag(bulk_z_indices), np.array(h00))

    h += np.kron(np.diag(np.ones(nz - 1), k=-1), h01)
    h += np.kron(np.diag(np.ones(nz - 1), k=1), h01.conj().T)

    return h


def calculate_disclination_rho(nz: int, nx: int, mass: float, phs_mass: float, disc_type='plaq', half_sign=None,
                               spin=None, use_gpu=True, fname='ed_disclination_ldos'):
    if half_sign is None or half_sign == 0:
        norb = 8
    else:
        norb = 4

    if use_gpu:
        import cupy as cp
        import cupy.linalg as clg

        print('Building Hamiltonian and sending to GPU')
        h = cp.asarray(disclination_hamiltonian(nz, nx, mass, phs_mass, disc_type, half_sign, spin))

        print('Solving for eigenvectors and eigenvalues')
        evals, evecs = clg.eigh(h)
        evals = evals.get()
        evecs = evecs.get()

        # Clear GPU memory
        mempool = cp.get_default_memory_pool()
        mempool.free_all_blocks()

    else:
        print('Building Hamiltonian')
        h = disclination_hamiltonian(nz, nx, mass, phs_mass, disc_type, half_sign, spin)

        print('Solving for eigenvectors and eigenvalues')
        evals, evecs = nlg.eigh(h)

    rho = np.zeros((nz, number_of_sites(nx, disc_type)))

    for ii, energy in enumerate(evals):
        if energy <= 0:
            wf = evecs[:, ii]
            temp_rho = np.reshape(np.multiply(np.conj(wf), wf), (nz, -1, norb))
            rho += np.sum(temp_rho, axis=-1).real

    results = rho
    params = (nz, nx, mass, phs_mass, disc_type, half_sign, spin)
    data = (results, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)

    return rho


def defect_free_hamiltonian(nx: int, mass: float, hoti_mass: float):
    # if spin is not None:
    #     if spin != 1 and spin != -1 and spin != 0:
    #         raise ValueError('Parameter "spin" must be either -1, 0, or 1')

    # gamma_xy = -1j * np.dot(gamma_x, gamma_y)

    # if spin is None or spin == 0:
    #     u_4 = slg.expm(1j * pi / 4 * (np.kron(gamma_xy, sigma_0) + np.kron(np.identity(4, dtype=complex), sigma_z)))
    # else:
    #     u_4 = slg.expm(1j * pi / 4 * (np.kron(gamma_xy, sigma_0) + np.kron(np.identity(4, dtype=complex), sigma_z)
    #                                   + spin * np.kron(np.identity(4, dtype=complex), sigma_0)))

    h_onsite = mass * np.kron(gamma_0, sigma_z)
    x_hoti_mass = hoti_mass * np.kron(gamma_0, sigma_x)
    y_hoti_mass = hoti_mass * np.kron(gamma_0, sigma_y)
    z_hoti_mass = hoti_mass * np.kron(gamma_5, sigma_0)

    h_x = 1j / 2 * np.kron(gamma_x, sigma_0) + 1 / 2 * np.kron(gamma_0, sigma_z)
    h_y = 1j / 2 * np.kron(gamma_y, sigma_0) + 1 / 2 * np.kron(gamma_0, sigma_z)
    h_z = 1j / 2 * np.kron(gamma_z, sigma_0) + 1 / 2 * np.kron(gamma_0, sigma_z)

    h_xz = -1 / 4 * np.kron(gamma_0, sigma_x)
    h_yz = -1 / 4 * np.kron(gamma_0, sigma_y)

    norb = 8

    # Arrange blocks into full Hamiltonian
    h = np.zeros((nx, nx, nx, norb, nx, nx, nx, norb), dtype=complex)

    # Onsite Hamiltonian
    for ii in range(nx):
        for jj in range(nx):
            for kk in range(nx):
                h[ii, jj, kk, :, ii, jj, kk, :] += h_onsite

                if ii == 0:
                    h[ii, jj, kk, :, ii, jj, kk, :] += x_hoti_mass
                elif ii == nx - 1:
                    h[ii, jj, kk, :, ii, jj, kk, :] -= x_hoti_mass

                if jj == 0:
                    h[ii, jj, kk, :, ii, jj, kk, :] += y_hoti_mass
                elif jj == nx - 1:
                    h[ii, jj, kk, :, ii, jj, kk, :] -= y_hoti_mass

                if kk == 0:
                    h[ii, jj, kk, :, ii, jj, kk, :] += z_hoti_mass
                elif kk == nx - 1:
                    h[ii, jj, kk, :, ii, jj, kk, :] -= z_hoti_mass

    # X-Hopping
    for ii in range(nx - 1):
        for jj in range(nx):
            for kk in range(nx):
                h[ii + 1, jj, kk, :, ii, jj, kk, :] += h_x
                h[ii, jj, kk, :, ii + 1, jj, kk, :] += h_x.conj().T

    # Y-Hopping
    for ii in range(nx):
        for jj in range(nx - 1):
            for kk in range(nx):
                h[ii, jj + 1, kk, :, ii, jj, kk, :] += h_y
                h[ii, jj, kk, :, ii, jj + 1, kk, :] += h_y.conj().T

    # Z-Hopping
    for ii in range(nx):
        for jj in range(nx):
            for kk in range(nx - 1):
                h[ii, jj, kk + 1, :, ii, jj, kk, :] += h_z
                h[ii, jj, kk, :, ii, jj, kk + 1, :] += h_z.conj().T

    # Need to work out signs and complex conjugations here. Missing half the terms currently
    # XZ-Hopping
    for ii in range(nx - 1):
        for jj in range(nx):
            for kk in range(nx - 1):
                h[ii + 1, jj, kk + 1, :, ii, jj, kk, :] += h_xz
                h[ii, jj, kk, :, ii + 1, jj, kk + 1, :] += h_xz.conj().T

                h[ii + 1, jj, kk, :, ii, jj, kk + 1, :] -= h_xz
                h[ii, jj, kk + 1, :, ii + 1, jj, kk, :] -= h_xz.conj().T

    # YZ-Hopping
    for ii in range(nx):
        for jj in range(nx - 1):
            for kk in range(nx - 1):
                h[ii, jj + 1, kk + 1, :, ii, jj, kk, :] += h_yz
                h[ii, jj, kk, :, ii, jj + 1, kk + 1, :] += h_yz.conj().T

                h[ii, jj + 1, kk, :, ii, jj, kk + 1, :] -= h_yz
                h[ii, jj, kk + 1, :, ii, jj + 1, kk, :] -= h_yz.conj().T

    return np.reshape(h, (norb * nx * nx * nx, norb * nx * nx * nx))


def calculate_defect_free_rho(nx: int, mass: float, hoti_mass: float, use_gpu=True):
    print('Building Hamiltonian')
    h = defect_free_hamiltonian(nx, mass, hoti_mass)

    norb = 8

    if use_gpu:
        import cupy as cp
        import cupy.linalg as clg

        print('Sending Hamiltonian to GPU')
        h = cp.asarray(h)

        print('Solving for eigenvectors and eigenvalues')
        evals, evecs = clg.eigh(h)
        evals = evals.get()
        evecs = evecs.get()
    else:
        print('Solving for eigenvectors and eigenvalues')
        evals, evecs = nlg.eigh(h)

    rho = np.zeros((nx, nx, nx))

    for ii, energy in enumerate(evals):
        if energy <= 0:
            wf = evecs[:, ii]
            prob_density = np.multiply(np.conj(wf), wf)
            temp_rho = np.reshape(prob_density, (nx, nx, nx, norb))
            rho += np.sum(temp_rho, axis=-1).real

    fname = 'defect_free_rho'
    results = rho
    params = (nx, mass, hoti_mass)
    data = (results, params)

    with open(data_dir / (fname + '.pickle'), 'wb') as handle:
        pkl.dump(data, handle)

    return rho
