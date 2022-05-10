import kwant
from itertools import product
import tinyarray
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('../src/bands.mplstyle')

# define Pauli-matrices for convenience
sigma_0 = tinyarray.array([[1, 0], [0, 1]])
sigma_x = tinyarray.array([[0, 1], [1, 0]])
sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
sigma_z = tinyarray.array([[1, 0], [0, -1]])


def make_ti_syst(ns, mass, flux, zeeman_fields, attach_lead=True):
    nx, ny, nz = ns

    lat = kwant.lattice.cubic(norbs=4)

    syst = kwant.Builder()

    # build list of sites
    sites = []

    z_surf_sites = []

    for (x, y, z) in product(range(nx), range(ny), range(nz)):
        if x == 0 or x == nx - 1:
            syst[lat(x, y, z)] = mass * np.kron(sigma_z, sigma_x) + zeeman_fields[0] * np.kron(sigma_z, sigma_y)
        elif y == 0 or y == ny - 1:
            syst[lat(x, y, z)] = mass * np.kron(sigma_z, sigma_x) + zeeman_fields[1] * np.kron(sigma_z, sigma_y)
        elif z == 0 or z == nz - 1:
            syst[lat(x, y, z)] = mass * np.kron(sigma_z, sigma_x) + zeeman_fields[2] * np.kron(sigma_z, sigma_y)
        else:
            syst[lat(x, y, z)] = mass * np.kron(sigma_z, sigma_x)

    # x-hopping
    syst[kwant.builder.HoppingKind((1, 0, 0), lat, lat)] = (-1j / 2 * np.kron(sigma_x, sigma_0)
                                                            + 1 / 2 * np.kron(sigma_z, sigma_x))

    # y-hopping
    syst[kwant.builder.HoppingKind((0, 1, 0), lat, lat)] = (-1j / 2 * np.kron(sigma_y, sigma_0)
                                                            + 1 / 2 * np.kron(sigma_z, sigma_x))

    # z-hopping
    syst[kwant.builder.HoppingKind((0, 0, 1), lat, lat)] = (-1j / 2 * np.kron(sigma_z, sigma_z)
                                                            + 1 / 2 * np.kron(sigma_z, sigma_x))

    if attach_lead:
        # build and attach the bottom lead
        lead = kwant.Builder(kwant.TranslationalSymmetry((0, 0, -1)))

        for (x, y) in product(range(nx), range(ny)):
            if x == 0 or x == nx - 1:
                lead[lat(x, y, 0)] = mass * np.kron(sigma_z, sigma_x) + zeeman_fields[0] * np.kron(sigma_z, sigma_y)
            elif y == 0 or y == ny - 1:
                lead[lat(x, y, 0)] = mass * np.kron(sigma_z, sigma_x) + zeeman_fields[1] * np.kron(sigma_z, sigma_y)
            else:
                lead[lat(x, y, 0)] = mass * np.kron(sigma_z, sigma_x)

        lead[kwant.builder.HoppingKind((1, 0, 0), lat, lat)] = (-1j / 2 * np.kron(sigma_x, sigma_0)
                                                                + 1 / 2 * np.kron(sigma_z, sigma_x))

        lead[kwant.builder.HoppingKind((0, 1, 0), lat, lat)] = (-1j / 2 * np.kron(sigma_y, sigma_0)
                                                                + 1 / 2 * np.kron(sigma_z, sigma_x))

        lead[kwant.builder.HoppingKind((0, 0, 1), lat, lat)] = (-1j / 2 * np.kron(sigma_z, sigma_z)
                                                                + 1 / 2 * np.kron(sigma_z, sigma_x))

        syst.attach_lead(lead)

    # attach virtual lead for device Green function
    vlead_interface = []
    for x in range(nx):
        for y in range(ny):
            vlead_interface.append(lat(x, y, nz - 1))

    mount_vlead(syst, vlead_interface, 4)

    # return the system
    return syst


def make_ti_disc_syst(length, width, height, mass, flux, zeeman):
    lat = kwant.lattice.cubic(norbs=4)

    syst = kwant.Builder()

    # build list of sites
    sites = []
    surf_sites = []
    for (x, y, z) in product(range(length), range(width), range(height)):
        if x < length / 2 or y < width / 2:
            sites.append(lat(x, y, z))
        if z == height - 1:
            surf_sites.append(lat(x, y, z))

    # onsite Hamiltonian
    syst[(site for site in sites)] = mass * np.kron(sigma_z, sigma_x)
    syst[(site for site in surf_sites)] = mass * np.kron(sigma_z, sigma_x) + zeeman * np.kron(sigma_z, sigma_y)

    # x-hopping
    syst[kwant.builder.HoppingKind((1, 0, 0), lat, lat)] = -1j / 2 * np.kron(sigma_x, sigma_0)

    # y-hopping
    syst[kwant.builder.HoppingKind((0, 1, 0), lat, lat)] = -1j / 2 * np.kron(sigma_y, sigma_0)

    # z-hopping
    syst[kwant.builder.HoppingKind((0, 0, 1), lat, lat)] = -1j / 2 * np.kron(sigma_z, sigma_z)

    # build and attach the bottom lead
    lead = kwant.Builder(kwant.TranslationalSymmetry((0, 0, -1)))

    lead[(lat(x, y, 0) for x in range(length) for y in range(height))] = mass * np.kron(sigma_z, sigma_x)
    lead[kwant.builder.HoppingKind((1, 0, 0), lat, lat)] = -1j / 2 * np.kron(sigma_x, sigma_0)
    lead[kwant.builder.HoppingKind((0, 1, 0), lat, lat)] = -1j / 2 * np.kron(sigma_y, sigma_0)
    lead[kwant.builder.HoppingKind((0, 0, 1), lat, lat)] = -1j / 2 * np.kron(sigma_z, sigma_z)

    syst.attach_lead(lead)

    # return the system
    return syst


def get_no_lead_dos(syst, energies=None):
    spectrum = kwant.kpm.SpectralDensity(syst)
    if energies is None:
        energies, densities = spectrum()
    else:
        densities = spectrum(energies)

    return energies, densities


def plot_dos(energies, densities):
    fig, ax = plt.subplots()
    ax.plot(energies, densities, 'b-')
    ax.plot(energies, np.zeros(len(energies)), 'k--')

    plt.tight_layout()

    plt.show()


def get_ldos(syst, dims, norbs, energies):
    nx, ny, nz = dims

    ldos = []

    for e in energies:
        green_function = kwant.solvers.default.greens_function(syst, energy=e)
        vlead_ind = len(green_function.out_leads) - 1
        g = green_function.submatrix(vlead_ind, vlead_ind)
        a = -2 * g.imag
        ldos.append(np.sum(np.reshape(np.diag(a), (nx, ny, norbs)), axis=-1))

    return ldos


def mount_vlead(sys, vlead_interface, norb):
    """Mounts virtual lead to interfaces provided.

    :sys: kwant.builder.Builder
        An unfinalized system to mount leads
    :vlead_interface: sequence of kwant.builder.Site
        Interface of lead
    :norb: integer
        Number of orbitals in system hamiltonian.
    """
    dim = len(vlead_interface)*norb
    zero_array = np.zeros((dim, dim), dtype=float)

    def selfenergy_func(energy, args=()):
        return zero_array

    vlead = kwant.builder.SelfEnergyLead(selfenergy_func, vlead_interface, ())
    sys.leads.append(vlead)
