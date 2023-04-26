"""set of function to perform gaussian inference"""

import numpy as np
from scipy.linalg import block_diag
from . import kernels as kk

# We consider the flavors
# xSigma, xg, V, V3, V8, V15, T3, T8, T15
# in NNPDF the labels are [1, 2, 3, 4, 5, 6, 9, 10, 11], since for now we only
# use these ones (DIS only), I ll use the following labels instead
flavor_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
nflavs = len(flavor_labels)
flavs_valence = [2, 3, 4, 5]  # flavs for which a valence sumrule is implemented
flavs_momentum = [0, 1]  # flavs involved in momentum sumrule


# default hyperparameters
theta = {
    "sigma": [3, 3, 3, 3, 3, 3, 3, 3, 3],
    "l0": [4, 4, 4, 4, 4, 4, 4, 4, 4],
    "alpha": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

# grid for sumrules
gridt = np.asarray(
    [
        0.0,
        1,
    ]
)
ngridt = gridt.size


# linear transformation for momentum sumrules


def build_sumrules(nflavs, flav_valence, flav_momentum=None):
    """
    Parameters:
        flav_valence (list) : list of int denoting the flavors for which a valence sum rule should be implemented
        flav_momentum (list): list of int denoting the flavors involved in momentum sum rule
    Returns:
        S (np.narray): narray of shape (num_sumrules, 2*num_flavors) implementing linear combination of primitives
    """
    S = np.zeros((len(flav_valence), 2 * nflavs))
    for sumrule, flav in enumerate(flav_valence):
        S[sumrule, 2 * flav] = -1.0
        S[sumrule, 2 * flav + 1] = 1.0

    if flav_momentum is not None:
        S_ = np.zeros(2 * nflavs)
        for flav in flav_momentum:
            S_[2 * flav] = -1.0
            S_[2 * flav + 1] = 1.0
        S = np.block([[S], [S_]])

    return S


S = build_sumrules(nflavs, flavs_valence, flavs_momentum)


def conditioning(mean, cov, y):
    """Given the Gaussian variable v=(x,z) with mean (mux,muz) and covariance cov
    (x,z) ~ N((mux, muz),cov)
    by applying conditioning it returns mean and covariance for the Gaussian
    variable x|z=y
    """

    # size of the vector to fix
    m = y.size
    # size of the output Gaussian variable
    n = mean.size - m

    # decompose mean vector in two blocks
    mux = mean[:n]
    muz = mean[n:]

    # decompose cov in four blocks
    Sxx = cov[:n, :n]
    Sxz = cov[:n, n:]
    Szx = Sxz.T
    Szz = cov[n:, n:]

    # invert Szz
    l, u = np.linalg.eig(Szz)  # usa np.linalg.eigh
    Szz_inv = u @ np.diag(1.0 / l) @ u.T
    id_test = Szz @ Szz_inv

    # compute mean posterior
    mu = mux + Sxz @ Szz_inv @ (y - muz)
    # compute cov posterior
    cov = Sxx - Sxz @ Szz_inv @ Szx

    return mu, cov, id_test


def cov_from_blocks(lines):
    """Given a list containing the lines of a matrix
    expressed as block contributions, it returns the full matrix"""
    mat = []
    for line in lines:
        mat.append(np.concatenate(line, axis=1))
    return np.concatenate(mat)


def block_diag_from_list(blocks):
    """Given a list of blocks create the corresponding block diagonal matrix.
    Parameters:
        blocks (list): list of numpy narrays
    Returns:
        res (np.narray): block diagonal matrix
    """
    nblocks = len(blocks)
    if nblocks == 1:
        return blocks[0]
    else:
        res = block_diag(blocks[0], blocks[1])
        for i in range(2, nblocks):
            res = block_diag(res, blocks[i])
    return res


def evaluate_on_grid(k, x, y, *args):
    """Evaluate the function k having arguments args on
    the grids x and y
    Parameters:
        k : function to be evaluated
        x (np.array): grid of points
        y (np.array): grid of points
        args : additional args of the function
    Returns:
        res (np.narray): narray containing values of k(x,y,args)
    """
    return k(x[:, None], y[None, :], *args)


class GP_generator:

    """Class to generate gaussian processes defined by
    - kernel (together with corresponding hyperparameters)
    - x-grids corresponding to the point of f
    - fk table for the full dataset
    - experimental information
    """

    def __init__(self, kernel, grid, y, Cy, fk, kin_lim=True, theta=theta):
        self.kernel = kernel
        self.grid = grid
        self.theta = theta
        self.ndata = y.size

        # impose f(1)=0 for all the flavors involved
        if kin_lim:
            ngrid = grid.size
            Af = np.zeros(ngrid)
            Af[ngrid - 1] = 1.0
            A_ = [Af for f in range(0, nflavs)]
            A = block_diag_from_list(A_)
            self.fk = np.block([[fk], [A]])
            y_kinlim = np.zeros(nflavs)
            self.y = np.concatenate([y, y_kinlim])
            self.Cy = block_diag(Cy, np.zeros((nflavs, nflavs)))

        else:
            self.fk = fk
            self.Cy = Cy
            self.y = y

    def build_blocks(self, grids):
        r"""
        Compute the blocks composing the prior covmat.
        Construct only the part connected to the kernel

        Parameters
        ----------
        kernel : string
            name of the kernel
        grid : numpy.ndarray
            grid for f (i.e. grid of the FK table)

        Returns
        -------
        A series of numpy.ndarray representing the different blocks
        """

        if self.kernel == "Gibbs":
            # if you want, you can describe each flavor with a different kernel
            # k_flavs = [kk.dxdyK for f in range(0,nflavs)] # kernels for flavors
            # use the same kernel for all flavors for now
            k = kk.dxdyK

            # add here kernel for the primitives
            K = kk.K
            dxK = kk.dxK

        else:
            raise ValueError(f"A kernel named {kernel} is not implemented")

        # construct the kernel dependent part of the blocks
        Sxx_ = []
        Sxsx_ = []
        Sxsxs_ = []

        dxsTxsxt_ = []
        dxTxxt_ = []
        Txtxt_ = []

        for sigma, l0, alpha in zip(
            self.theta["sigma"], self.theta["l0"], self.theta["alpha"]
        ):

            Sxx_.append(evaluate_on_grid(k, self.grid, self.grid, sigma, l0, alpha))
            Sxsx_.append(evaluate_on_grid(k, grids, self.grid, sigma, l0, alpha))
            Sxsxs_.append(evaluate_on_grid(k, grids, grids, sigma, l0, alpha))

            dxsTxsxt_.append(evaluate_on_grid(dxK, grids, gridt, sigma, l0, alpha))
            dxTxxt_.append(evaluate_on_grid(dxK, self.grid, gridt, sigma, l0, alpha))
            Txtxt_.append(evaluate_on_grid(K, gridt, gridt, sigma, l0, alpha))

        # construct the block diagonal blocks
        Sxx = block_diag_from_list(Sxx_)
        Sxsxs = block_diag_from_list(Sxsxs_)
        Sxsx = block_diag_from_list(Sxsx_)
        dxsTxsxt = block_diag_from_list(dxsTxsxt_)
        dxTxxt = block_diag_from_list(dxTxxt_)
        Txtxt = block_diag_from_list(Txtxt_)

        # complete construction of blocks adding fk tables, exp information
        # and sumrules operator

        A00 = Sxsxs
        A01 = Sxsx @ self.fk.T
        A02 = dxsTxsxt @ S.T
        A11 = self.fk @ Sxx @ self.fk.T + self.Cy
        A12 = self.fk @ dxTxxt @ S.T
        A22 = S @ Txtxt @ S.T

        return A00, A01, A02, A11, A12, A22

    def get_prior(self, grids, sumrules=False):
        A00, A01, A02, A11, A12, A22 = self.build_blocks(grids)

        if sumrules:
            lines = [
                [A00, A01, A02],
                [A01.T, A11, A12],
                [A02.T, A12.T, A22],
            ]
        else:
            lines = [
                [A00, A01],
                [A01.T, A11],
            ]

        cov_prior = cov_from_blocks(lines)
        mean_prior = np.zeros(cov_prior.shape[0])

        return mean_prior, cov_prior

    def get_posterior(self, grids, sumrules=False):
        if sumrules:
            mean_prior, cov_prior = self.get_prior(grids, sumrules=sumrules)

            # apply conditioning first time

            xi = np.asarray([1, 3, 1, 3, 3])  # sum rule f'(1)-f'(0) = 1
            mean_posterior, cov_posterior, _ = conditioning(mean_prior, cov_prior, xi)

            # apply conditioning second time
            mean_prior = mean_posterior
            cov_prior = cov_posterior
            xi = self.y  # FK @ f + Cy = y
            mean_posterior, cov_posterior, _ = conditioning(mean_prior, cov_prior, xi)

        else:
            mean_prior, cov_prior = self.get_prior(grids)
            # apply conditioning
            xi = self.y  # FK @ f + Cy = y
            mean_posterior, cov_posterior, _ = conditioning(mean_prior, cov_prior, xi)

        return mean_posterior, cov_posterior

    def compute_th_prediction(self, sumrules=False):
        # compute posterior distribution for ff = f | Af+eps=y
        mean_posterior_f, cov_posterior_f = self.get_posterior(self.grid, sumrules)

        # the th prediction is the mean of the gaussian variable A @ ff, with ff = f | Af+eps=y
        # By construction it is y
        th_prediction = self.fk @ mean_posterior_f

        # the pdf error is given by the covariance of the gaussian variable A @ ff
        pdf_error = self.fk @ cov_posterior_f @ self.fk.T
        return th_prediction[: self.ndata], pdf_error[: self.ndata]
