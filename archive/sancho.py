# This function seems to not work correctly, keeping it here for posterity
def sancho(energy, h0, s0, h1, s1, h_surf=None):
    # Source: http://www.pitt.edu/~djb145/python,/transport/2018/07/22/RGF-1/
    """
    Generate surface Green's function through recursive Sancho method.
    INPUT
        energy: (complex float) Complex energy.
        H0: (numpy matrix) Intra-layer Hamiltonian.
        S0: (numpy matrix) Intra-layer overlap matrix. Make this the identity.
        H1: (numpy matrix) Inter-layer coupling Hamiltonian.
        S1: (numpy matrix) Inter-layer coupling overlap matrix. Make this a zero matrix.
    OUTPUT
        greenSurface: (complex numpy matrix) Surface Green's function.
        greenBulk: (complex numpy matrix) Bulk Green's function.
    REFERENCES
    Sancho, MP Lopez, et al. "Highly convergent schemes for the calculation of bulk and surface Green functions."
    Journal of Physics F: Metal Physics 15.4 (1985): 851.
    """

    # Variables
    invMat = nlg.inv(energy * s0 - h0)
    tol = 1e-9
    itCounter = 0
    maxIter = 50

    # Initialize recursion parameters
    epSurfaceOld = h0 + h1 * invMat * h1.T.conj()
    if h_surf is not None:
        epSurfaceOld += h_surf
    epBulkOld = epSurfaceOld + h1.T.conj() * invMat * h1
    alphaOld = h1 * invMat * h1
    betaOld = h1.T.conj() * invMat * h1.T.conj()

    # Set dummy updated matrices
    epSurface = 10 * tol + epSurfaceOld
    epBulk = 10 * tol + epBulkOld
    alpha = 10 * tol + alphaOld
    beta = 10 * tol + betaOld

    # Initial update difference norms
    diff_epSurface = nlg.norm(epSurface - epSurfaceOld)
    diff_epBulk = nlg.norm(epBulk - epBulkOld)
    diff_alpha = nlg.norm(alpha - alphaOld)
    diff_beta = nlg.norm(beta - betaOld)

    # Iterate until convergence criteria satisfied
    while ((itCounter < maxIter) and (
            (diff_epSurface > tol) or (diff_epBulk > tol) or (diff_alpha > tol) or (diff_beta > tol))):
        # Calculate recursion parameters
        invMat = nlg.inv(energy * s0 - epBulkOld)
        alpha = alphaOld * invMat * alphaOld
        beta = betaOld * invMat * betaOld
        epBulk = epBulkOld + alphaOld * invMat * betaOld + betaOld * invMat * alphaOld
        epSurface = epSurfaceOld + alphaOld * invMat * betaOld

        # Check convergence
        diff_epSurface = nlg.norm(epSurface - epSurfaceOld)
        diff_epBulk = nlg.norm(epBulk - epBulkOld)
        diff_alpha = nlg.norm(alpha - alphaOld)
        diff_beta = nlg.norm(beta - betaOld)

        # Set new values to old
        alphaOld = alpha
        betaOld = beta
        epBulkOld = epBulk
        epSurfaceOld = epSurface

        # Update itCounter
        itCounter += 1

    print(f'iterations: {itCounter}')
    print(f'max diff: {np.max((diff_beta, diff_epBulk, diff_alpha, diff_epSurface))}')

    # Calculate Greens functions
    greenSurface = nlg.inv(energy * s0 - epSurface)
    greenBulk = nlg.inv(energy * s0 - epBulk)

    print(f'alpha: {alpha}')
    print(f'beta: {beta}')

    return greenSurface, greenBulk