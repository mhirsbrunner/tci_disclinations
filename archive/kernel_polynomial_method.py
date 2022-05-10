# Tutorial 2.8. Calculating spectral density with the Kernel Polynomial Method
# ============================================================================
#
# Physics background
# ------------------
#  - Chebyshev polynomials, random trace approximation, spectral densities.
#
# Kwant features highlighted
# --------------------------
#  - kpm module,kwant operators.

import scipy

# For plotting
from matplotlib import pyplot as plt

# necessary imports
import kwant
import numpy as np


# define the system
def make_syst(r=30, t=-1, a=1):
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(a, norbs=1)

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    syst[lat.shape(circle, (0, 0))] = 0.
    syst[lat.neighbors()] = t
    syst.eradicate_dangling()

    return syst

# define a Haldane system
def make_syst_topo(r=30, a=1, t=1, t2=0.5):
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(a, norbs=1, name=['a', 'b'])

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    syst[lat.shape(circle, (0, 0))] = 0.
    syst[lat.neighbors()] = t
    # add second neighbours hoppings
    syst[lat.a.neighbors()] = 1j * t2
    syst[lat.b.neighbors()] = -1j * t2
    syst.eradicate_dangling()

    return lat, syst.finalized()


# define the system
def make_syst_staggered(r=30, t=-1, a=1, m=0.1):
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(a, norbs=1)

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    syst[lat.a.shape(circle, (0, 0))] = m
    syst[lat.b.shape(circle, (0, 0))] = -m
    syst[lat.neighbors()] = t
    syst.eradicate_dangling()

    return syst

# Plot several density of states curves on the same axes.
def plot_dos(labels_to_data):
    plt.figure(figsize=(5,4))
    for label, (x, y) in labels_to_data:
        plt.plot(x, y.real, label=label, linewidth=2)
    plt.legend(loc=2, framealpha=0.5)
    plt.xlabel("energy [t]")
    # plt.ylabel(ylabel)
    plt.show()
    plt.clf()


# Plot fill density of states plus curves on the same axes.
def plot_dos_and_curves(dos, labels_to_data):
    plt.figure(figsize=(5,4))
    plt.fill_between(dos[0], dos[1], label="DoS [a.u.]",
                     alpha=0.5, color='gray')
    for label, (x, y) in labels_to_data:
        plt.plot(x, y, label=label, linewidth=2)
    plt.legend(loc=2, framealpha=0.5)
    plt.xlabel("energy [t]")
    # plt.ylabel(ylabel)
    plt.show()
    plt.clf()

def site_size_conversion(densities):
    return 3 * np.abs(densities) / max(densities)


# Plot several local density of states maps in different subplots
def plot_ldos(syst, densities, file_name=None):
    fig, axes = plt.subplots(1, len(densities), figsize=(7*len(densities), 7))
    for ax, (title, rho) in zip(axes, densities):
        kwant.plotter.density(syst, rho.real, ax=ax)
        ax.set_title(title)
        ax.set(adjustable='box', aspect='equal')
    plt.show()
    plt.clf()

def simple_dos_example():
    fsyst = make_syst().finalized()

    spectrum = kwant.kpm.SpectralDensity(fsyst)

    energies, densities = spectrum()

    energy_subset = np.linspace(0, 2)
    density_subset = spectrum(energy_subset)

    plot_dos([
        ('densities', (energies, densities)),
        ('density subset', (energy_subset, density_subset)),
    ])


def dos_integrating_example(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)

    print('identity resolution:', spectrum.integrate())

    # Fermi energy 0.1 and temperature 0.2
    fermi = lambda E: 1 / (np.exp((E - 0.1) / 0.2) + 1)

    print('number of filled states:', spectrum.integrate(fermi))


def increasing_accuracy_example(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)
    original_dos = spectrum()  # get unaltered DoS

    spectrum.add_moments(energy_resolution=0.03)

    increased_resolution_dos = spectrum()

    plot_dos([
        ('density', original_dos),
        ('higher energy resolution', increased_resolution_dos),
    ])

    spectrum.add_moments(100)
    spectrum.add_vectors(5)

    increased_moments_dos = spectrum()

    plot_dos([
        ('density', original_dos),
        ('higher number of moments', increased_moments_dos),
    ])


def operator_example(fsyst):
    # identity matrix
    matrix_op = scipy.sparse.eye(len(fsyst.sites))
    matrix_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=matrix_op)

    # 'sum=True' means we sum over all the sites
    kwant_op = kwant.operator.Density(fsyst, sum=True)
    operator_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=kwant_op)

    plot_dos([
        ('identity matrix', matrix_spectrum()),
        ('kwant.operator.Density', operator_spectrum()),
    ])


def ldos_example(fsyst):
    # 'sum=False' is the default, but we include it explicitly here for clarity.
    kwant_op = kwant.operator.Density(fsyst, sum=False)
    local_dos = kwant.kpm.SpectralDensity(fsyst, operator=kwant_op)

    zero_energy_ldos = local_dos(energy=0)
    finite_energy_ldos = local_dos(energy=1)
    plot_ldos(fsyst, [
        ('energy = 0', zero_energy_ldos),
        ('energy = 1', finite_energy_ldos)
    ])


def ldos_sites_example():
    fsyst = make_syst_staggered().finalized()
    # find 'A' and 'B' sites in the unit cell at the center of the disk
    center_tag = np.array([0, 0])
    where = lambda s: s.tag == center_tag
    # make local vectors
    vector_factory = kwant.kpm.LocalVectors(fsyst, where)

    # 'num_vectors' can be unspecified when using 'LocalVectors'
    local_dos = kwant.kpm.SpectralDensity(fsyst, num_vectors=None,
                                          vector_factory=vector_factory,
                                          mean=False)
    energies, densities = local_dos()
    plot_dos([
        ('A sublattice', (energies, densities[:, 0])),
        ('B sublattice', (energies, densities[:, 1])),
    ])


def vector_factory_example(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)
    # construct a generator of vectors with n random elements -1 or +1.
    n = fsyst.hamiltonian_submatrix(sparse=True).shape[0]
    def binary_vectors():
        while True:
           yield np.rint(np.random.random_sample(n)) * 2 - 1

    custom_factory = kwant.kpm.SpectralDensity(fsyst,
                                               vector_factory=binary_vectors())
    plot_dos([
        ('default vector factory', spectrum()),
        ('binary vector factory', custom_factory()),
    ])


def bilinear_map_operator_example(fsyst):
    rho = kwant.operator.Density(fsyst, sum=True)

    # sesquilinear map that does the same thing as `rho`
    def rho_alt(bra, ket):
        return np.vdot(bra, ket)

    rho_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=rho)
    rho_alt_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=rho_alt)

    plot_dos([
        ('kwant.operator.Density', rho_spectrum()),
        ('bilinear operator', rho_alt_spectrum()),
    ])

def conductivity_example():
    # construct the Haldane model
    lat, fsyst = make_syst_topo()
    # find 'A' and 'B' sites in the unit cell at the center of the disk
    where = lambda s: np.linalg.norm(s.pos) < 1

    # component 'xx'
    s_factory = kwant.kpm.LocalVectors(fsyst, where)
    cond_xx = kwant.kpm.conductivity(fsyst, alpha='x', beta='x', mean=True,
                                     num_vectors=None, vector_factory=s_factory)
    # component 'xy'
    s_factory = kwant.kpm.LocalVectors(fsyst, where)
    cond_xy = kwant.kpm.conductivity(fsyst, alpha='x', beta='y', mean=True,
                                     num_vectors=None, vector_factory=s_factory)

    energies = cond_xx.energies
    cond_array_xx = np.array([cond_xx(e, temperature=0.01) for e in energies])
    cond_array_xy = np.array([cond_xy(e, temperature=0.01) for e in energies])

    # area of the unit cell per site
    area_per_site = np.abs(np.cross(*lat.prim_vecs)) / len(lat.sublattices)
    cond_array_xx /= area_per_site
    cond_array_xy /= area_per_site
    # ldos
    s_factory = kwant.kpm.LocalVectors(fsyst, where)
    spectrum = kwant.kpm.SpectralDensity(fsyst, num_vectors=None,
                                         vector_factory=s_factory)

    plot_dos_and_curves(
    (spectrum.energies, spectrum.densities * 8),
    [
        (r'Longitudinal conductivity $\sigma_{xx} / 4$',
         (energies, cond_array_xx / 4)),
        (r'Hall conductivity $\sigma_{xy}$',
         (energies, cond_array_xy))],
        ylabel=r'$\sigma [e^2/h]$',
        file_name='kpm_cond'
    )


def main():
    simple_dos_example()

    fsyst = make_syst().finalized()

    dos_integrating_example(fsyst)
    increasing_accuracy_example(fsyst)
    operator_example(fsyst)
    ldos_example(fsyst)
    ldos_sites_example()
    vector_factory_example(fsyst)
    bilinear_map_operator_example(fsyst)
    conductivity_example()


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
