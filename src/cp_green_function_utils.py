import cupy as cp
import cupy.linalg as clg


def retarded_green_function(hamiltonian: cp.ndarray, energy: float, eta=1e-6) -> cp.ndarray:
    n = hamiltonian.shape[0]
    return clg.inv((energy + 1j * eta) * cp.identity(n) - hamiltonian)


def surface_green_function(energy, h00, h01, surf_pert=None):
    it_max = 100
    tol = 1e-8

    if surf_pert is None:
        surf_pert = cp.zeros(h00.shape)

    energy = energy * cp.identity(h00.shape[0])

    eps_s = h00

    eps = h00
    alpha = h01.conj().T
    beta = h01

    alpha_norm = 0
    beta_norm = 0

    for ii in range(it_max):
        g0_beta = clg.solve(energy - eps, beta)
        g0_alpha = clg.solve(energy - eps, alpha)

        eps_s = eps_s + alpha @ g0_beta
        eps = eps + alpha @ g0_beta + beta @ g0_alpha

        alpha = alpha @ g0_alpha
        beta = beta @ g0_beta

        alpha_norm = clg.norm(alpha)
        beta_norm = clg.norm(beta)
        if cp.max(alpha_norm) < tol or cp.max(beta_norm) < tol:
            gs = clg.solve(energy - eps_s - surf_pert, cp.identity(h00.shape[0]))
            gb = clg.solve(energy - eps, cp.identity(h00.shape[0]))
            return gs, gb

    print(f'Max iterations reached. alpha_norm: {alpha_norm}, beta_norm: {beta_norm}')

    gs = clg.solve(energy - eps_s - surf_pert, cp.identity(h00.shape[0]))
    gb = clg.solve(energy - eps, cp.identity(h00.shape[0]))

    return gs, gb


def spectral_function(g=None, ham=None, energy=None, eta=None) -> cp.ndarray:
    if g is None:
        if ham is not None:
            if energy is None or eta is None:
                raise ValueError('Hamiltonian, energy, and broadening must be passed'
                                 ' if the Green function is not specified.')
            else:
                g = retarded_green_function(ham, energy, eta=eta)
        else:
            raise ValueError('Either Green function or Hamiltonian must be given.')
    elif ham is not None:
        print('Both Green function and Hamiltonian specified, defaulting to using the Green function.')

    return -2 * cp.imag(g)
