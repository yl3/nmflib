"""Test NMF code."""

import numpy as np
import nmflib.nmf


class SimpleNMFData:
    """Three channels and two signatures.

    The first signature only creates mutations in the first channel. The second
    signature creates mutations evenly in all channels.
    """
    X = np.array([[15, 40, 12, 85],
                  [15, 20, 0,  25],
                  [0,  10, 6,  30],
                  [0,  20, 12, 60]])
    W_true = np.array([[.5, 0.4],
                       [.5, 0],
                       [0,  0.2],
                       [0,  0.4]])
    H_true = np.array([[30, 40, 0,  50],
                       [0,  50, 30, 150]])

    def check_answer(W, H):
        """Check if the fitted matrices are close to the true one."""
        TOL = 1e-4
        assert W.shape == (4, 2)
        assert H.shape == (2, 4)

        # Order columns in W to match W_true.
        if W[0, 0] < W[0, 1]:
            W = W[:, [1, 0]]
            H = H[[1, 0], :]

        assert np.all(np.abs(W - SimpleNMFData.W_true) < TOL)
        assert np.all(np.abs(H - SimpleNMFData.H_true) < TOL)


def test_fit_nmf():
    W, H, n_iter, errors = nmflib.nmf.fit_nmf(
        SimpleNMFData.X, 2, max_iter=10000, tol=1e-10)
    SimpleNMFData.check_answer(W, H)

    W, H, n_iter, errors = nmflib.nmf.fit_nmf(
        SimpleNMFData.X, 2, max_iter=10000, tol=1e-10, use_numba=True)
    SimpleNMFData.check_answer(W, H)
