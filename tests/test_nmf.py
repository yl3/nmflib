"""Test NMF code."""

import numpy as np
import nmflib.nmf


class SimpleNMFData:
    """Three channels and two signatures.

    The first signature only creates mutations in the first two channels. The
    second signature creates mutations in all channels.
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

    @classmethod
    def check_answer(cls, W, H):
        """Check if the fitted matrices are close to the true one."""
        TOL = 1e-4
        assert W.shape == (4, 2)
        assert H.shape == (2, 4)

        # Order columns in W to match W_true.
        if W[0, 0] < W[0, 1]:
            W = W[:, [1, 0]]
            H = H[[1, 0], :]

        assert np.all(np.abs(W - cls.W_true) < TOL)
        assert np.all(np.abs(H - cls.H_true) < TOL)


class ScaledNMFData(SimpleNMFData):
    """Three channels and two signatures.

    The first signature only creates mutations in the first two channels. The
    second signature creates mutations in all channels.

    The third and fourth samples are "exome" patients where only 25% of 50% of
    the possible mutations can be observed. Overall counts are however adjusted
    to match that of SimpleNMFData.
    """
    S = np.array([[1, 1, 0.5, 0.25],
                  [1, 1, 0.5, 0.25],
                  [1, 1, 0.5, 0.25],
                  [1, 1, 0.5, 0.25]])
    H_true = np.array([[30, 40, 0,  200],
                       [0,  50, 60, 600]])


def test_fit_nmf():
    """Test fitting a regular NMF."""
    W, H, n_iter, errors = nmflib.nmf.fit_nmf(
        SimpleNMFData.X, 2, max_iter=10000, tol=1e-10)
    SimpleNMFData.check_answer(W, H)


def test_fit_nmf_scaled():
    """Test fitting NMF with a scaling matrix."""
    W, H, n_iter, errors = nmflib.nmf.fit_nmf(
        ScaledNMFData.X, 2, ScaledNMFData.S, max_iter=10000, tol=1e-10)
    ScaledNMFData.check_answer(W, H)


def test_match_signatures():
    # Generate some random signatures
    np.random.seed(0)
    K = 10
    signatures = np.random.dirichlet([0.5] * 96, K)
    scramble_idx = np.random.choice(K, K, False)
    scrambled_signatures = signatures[:, scramble_idx]
    found_idx = nmflib.nmf.match_signatures(scrambled_signatures, signatures)
    assert np.all(found_idx == scramble_idx)
