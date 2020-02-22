"""Test NMF code."""
# Copyright (C) 2019 Yilong Li (yilong.li.yl3@gmail.com) - All Rights Reserved

import pytest

import logging
import re
import time

import numpy as np
import pandas as pd
import scipy.stats

import nmflib.nmf


def test_validate_is_ndarray():
    """Test nmf._validate_is_ndarray()."""
    assert nmflib.nmf._validate_is_ndarray(None) is None
    arr = np.arange(6).reshape(3, 2)
    df = pd.DataFrame(arr)
    assert isinstance(nmflib.nmf._validate_is_ndarray(arr), np.ndarray)
    assert isinstance(nmflib.nmf._validate_is_ndarray(df), np.ndarray)
    with pytest.raises(ValueError):
        nmflib.nmf._validate_is_ndarray("Not an array")


class SimpleNMFData:
    """Three channels and two signatures. No sampling noise whatsoever.

    The first signature only creates mutations in the first two channels. The
    second signature creates mutations in all channels.
    """
    W_true = np.array([[.5, 0.4], [.5, 0], [0, 0.2], [0, 0.4]])
    k = W_true.shape[1]
    H_true = np.array([[30, 40, 0, 50], [0, 50, 30, 150]])
    S = np.array([[1, 1, 0.5, 0.25], [1, 1, 0.5, 0.25], [1, 1, 0.5, 0.25],
                  [1, 1, 0.5, 0.25]])

    def __init__(self, use_S=False, use_O=False):
        self.W = SimpleNMFData.W_true
        self.H = SimpleNMFData.H_true
        self.rank = SimpleNMFData.k
        X_exp = np.matmul(self.W, self.H)
        if use_O:
            # Add an offset that's identical to 2 * WH...
            self.O = 2 * np.matmul(self.W, self.H)  # noqa: E741
            X_exp += self.O
        else:
            self.O = None  # noqa: E741
        if use_S:
            self.S = SimpleNMFData.S
            X_exp *= self.S
        else:
            self.S = None
        self.X = np.round(X_exp)

    def check_answer(self, W, H):
        """Check if the fitted matrices are close to the true one."""
        RELTOL = 0.05  # Needs to be quite high since local maxima are real...
        ABSTOL = 0.01
        assert W.shape == (4, 2)
        assert H.shape == (2, 4)

        # Order columns in W to match W_true.
        if W[0, 0] < W[0, 1]:
            W = W[:, [1, 0]]
            H = H[[1, 0], :]

        # Check the closeness of the parameters.
        assert np.allclose(self.W_true, W, RELTOL, ABSTOL)
        assert np.allclose(self.H_true, H, RELTOL, ABSTOL)

        # Check the closeness of the solution.
        X_exp = nmflib.nmf.nmf_mu(W, H, self.S, self.O)
        assert np.allclose(X_exp, self.X, RELTOL, ABSTOL)


class SyntheticPCAWG():
    """Synthetic mutational signature convolutions from PCAWG.

    Offsets and scales are optionally simulated.

    Args:
        random_state (int): Random state for simulating counts. Default: 0.
    """
    def __init__(self,
                 datafiles,
                 use_S=False,
                 use_O=False,
                 r=None,
                 random_state=0):
        self.r = r
        self.S = None
        self.O = None  # noqa: E741
        self.random_state = random_state
        self._load_syn_data(datafiles, use_S, use_O)

    def simulate(self):
        """Simulate Poisson or nbinom (when r is provided) counts."""
        np.random.seed(self.random_state)
        if self.r is None:
            rvs = scipy.stats.poisson.rvs(self.X_exp)
        else:
            p = nmflib.nmf._nb_p(self.X_exp, self.r)
            rvs = scipy.stats.nbinom.rvs(self.r, p)
        return rvs

    def _load_syn_data(self, datafiles, use_S, use_O):
        self.W = pd.read_csv(datafiles + '/ground.truth.syn.sigs.csv.gz',
                             index_col=[0, 1],
                             header=0)
        self.H = pd.read_csv(datafiles + '/ground.truth.syn.exposures.csv.gz',
                             index_col=0,
                             header=0)
        X_exp = self.W.dot(self.H)
        if use_O:
            O_fraction = scipy.stats.uniform.rvs(0.5,
                                                 1 - 0.5,
                                                 X_exp.shape,
                                                 random_state=self.random_state)
            O = O_fraction * X_exp.values  # noqa: E741
            X_exp -= O
            self.O = O  # noqa: E741
        else:
            self.O = None  # noqa: E741
        if use_S:
            S = scipy.stats.uniform.rvs(0.05,
                                        1 - 0.05,
                                        X_exp.shape,
                                        random_state=self.random_state)
            X_exp *= S
            self.X_exp = X_exp
            self.S = S
        else:
            self.S = None
        self.X_exp = X_exp


synthetic_nmf_data_dir = pytest.mark.datafiles(
    'test_data/ground.truth.syn.sigs.csv.gz',
    'test_data/ground.truth.syn.exposures.csv.gz')


@synthetic_nmf_data_dir
@pytest.mark.parametrize("use_S", [True, False])
@pytest.mark.parametrize("use_O", [True, False])
@pytest.mark.parametrize("r", [None, 100])
def test_initialise_nmf(datafiles, use_S, use_O, r):
    """Test nmf.initialise_nmf() on the PCAWG mutation count dataset."""
    synthetic_pcawg_dataset = SyntheticPCAWG(datafiles, use_S, use_O, r)
    X = synthetic_pcawg_dataset.simulate()
    M, N = X.shape
    k = 21
    W_init, H_init = nmflib.nmf.initialise_nmf(X, k, synthetic_pcawg_dataset.S,
                                               synthetic_pcawg_dataset.O)
    assert W_init.shape == (M, k)
    assert np.all(W_init > 0)
    assert np.all(H_init > 0)
    assert H_init.shape == (k, N)
    assert np.allclose(np.sum(W_init, 0), 1)
    if use_S:
        X_exp = X / synthetic_pcawg_dataset.S
    else:
        X_exp = X
    if not use_O:
        # This is not accurate, since O can be larger than X sometimes.
        assert np.allclose(np.sum(H_init, 0), np.sum(X_exp, 0))


def test_iterate_nbinom_nmf_r():
    """
    Test nmf._iterate_nbinom_nmf_r_ml().
    """
    mu = 20
    true_r = 10
    p = nmflib.nmf._nb_p(mu, true_r)
    np.random.seed(0)
    x = scipy.stats.nbinom.rvs(true_r, p, size=int(1e5))
    r = nmflib.nmf._initialise_nb_r(x, mu)
    for _ in range(10):
        r = nmflib.nmf._iterate_nbinom_nmf_r_ml(x, mu, r)
    assert 9.9 < r < 10.1  # Should be ~10.0


@synthetic_nmf_data_dir
@pytest.mark.parametrize("use_S", [True, False])
@pytest.mark.parametrize("use_O", [True, False])
@pytest.mark.parametrize("r", [None, 100])
@pytest.mark.slow
def test_nmf_updates_monotonicity(datafiles, use_S, use_O, r):
    """
    Make sure that each individual W, H and r update step is monotonous.
    """
    synthetic_pcawg_dataset = SyntheticPCAWG(datafiles, use_S, use_O, r)
    X_obs = synthetic_pcawg_dataset.simulate()
    S = synthetic_pcawg_dataset.S
    O = synthetic_pcawg_dataset.O  # noqa: E741
    TARGET_RANK = 20

    # Run 10 iterations and make sure each individual update is monotonous.
    W, H = nmflib.nmf.initialise_nmf(
        X_obs, TARGET_RANK, random_state=synthetic_pcawg_dataset.random_state)
    X_exp = nmflib.nmf.nmf_mu(W, H, S, O)
    r = nmflib.nmf._initialise_nb_r(X_obs, X_exp)
    loglik = prev_loglik = np.sum(nmflib.nmf.loglik(X_obs, X_exp, r))
    for _ in range(10):
        W = nmflib.nmf._multiplicative_update_W(X_obs, W, H, S, O, r=r)
        X_exp = nmflib.nmf.nmf_mu(W, H, S, O)
        loglik = np.sum(nmflib.nmf.loglik(X_obs, X_exp, r))
        assert loglik > prev_loglik
        prev_loglik = loglik

        H = nmflib.nmf._multiplicative_update_H(X_obs, W, H, S, O, r=r)
        X_exp = nmflib.nmf.nmf_mu(W, H, S, O)
        loglik = np.sum(nmflib.nmf.loglik(X_obs, X_exp, r))
        assert loglik > prev_loglik
        prev_loglik = loglik

        X_exp = nmflib.nmf.nmf_mu(W, H, S, O)
        r = nmflib.nmf._iterate_nbinom_nmf_r_ml(X_obs, X_exp, r)
        loglik = np.sum(nmflib.nmf.loglik(X_obs, X_exp, r))
        assert loglik > prev_loglik
        prev_loglik = loglik


@synthetic_nmf_data_dir
@pytest.mark.parametrize("use_S", [True, False])
@pytest.mark.parametrize("use_O", [True, False])
@pytest.mark.parametrize("r", [None, 100])
@pytest.mark.slow
def test_fit_nmf_monotonicity(datafiles, use_S, use_O, r):
    """Make sure the nmf.fit() errors are monotonous."""
    synthetic_pcawg_dataset = SyntheticPCAWG(datafiles, use_S, use_O, r)
    X_obs = synthetic_pcawg_dataset.simulate()
    S = synthetic_pcawg_dataset.S
    O = synthetic_pcawg_dataset.O  # noqa: E741
    TARGET_RANK = 10

    # Test the monotonicity of overall errors.
    nbinom_fit = r is not None
    max_iter = 100
    epoch_len = 10
    W, H, r, n_iter, errors = nmflib.nmf.fit(X_obs,
                                             TARGET_RANK,
                                             S,
                                             O,
                                             max_iter=max_iter,
                                             epoch_len=epoch_len,
                                             nbinom_fit=nbinom_fit,
                                             verbose=True,
                                             max_iter_error=False)

    assert len(errors) <= (max_iter + epoch_len - 1) // epoch_len

    for i in range(len(errors) - 1):
        assert errors[i + 1] <= errors[i]


@pytest.mark.parametrize("use_S", [True, False])
@pytest.mark.parametrize("use_O", [True, False])
def test_fit_nmf_small_dataset(use_S, use_O, caplog):
    """Test fitting a simple NMF dataset with no sampling noise."""
    caplog.set_level(logging.INFO)
    nmf_dataset = SimpleNMFData(use_S, use_O)
    abstol = 1e-30  # Ensure very good convergence.
    if use_S:
        S = nmf_dataset.S
    else:
        S = None
    if use_O:
        O = nmf_dataset.O  # noqa: E741
    else:
        O = None  # noqa: E741
    W, H, r, n_iter, errors = nmflib.nmf.fit(nmf_dataset.X,
                                             nmf_dataset.rank,
                                             S,
                                             O,
                                             max_iter=float("inf"),
                                             abstol=abstol)
    nmf_dataset.check_answer(W, H)

    # Test logging.
    W, H, r, n_iter, errors = nmflib.nmf.fit(nmf_dataset.X,
                                             nmf_dataset.rank,
                                             S,
                                             O,
                                             max_iter=100,
                                             epoch_len=10,
                                             verbose=True,
                                             abstol=abstol)
    assert re.search(r"Iteration 10 after \S+ seconds, error: ", caplog.text)
    assert re.search(r"Iteration 20 after \S+ seconds, error: ", caplog.text)


@synthetic_nmf_data_dir
@pytest.mark.slow
@pytest.mark.parallel
def test_mpfit(datafiles):
    """
    Make sure nmf.mpfit() works and is (slightly) faster than just fit().
    """
    synthetic_pcawg_data = SyntheticPCAWG(datafiles, True, True, 10)
    X_obs = synthetic_pcawg_data.simulate()
    S = synthetic_pcawg_data.S
    O = synthetic_pcawg_data.O  # noqa: E741
    TARGET_RANK = 20

    # Measure time without multiprocessing.
    start_time = time.time()
    sp_res = []
    for i in range(2):
        sp_res.append(
            nmflib.nmf.mpfit(
                random_inits=2,
                n_processes=1,
                n_threads=1,
                random_state=0,
                verbose=False,
                X=X_obs,
                k=TARGET_RANK,
                S=S,
                O=O,  # noqa: E741
                nbinom_fit=True))
    single_process_elapsed = time.time() - start_time

    # Measure time with multiprocessing.
    start_time = time.time()
    mp_res = nmflib.nmf.mpfit(
        random_inits=2,
        n_processes=2,
        n_threads=1,
        random_state=0,
        verbose=False,
        X=X_obs,
        k=TARGET_RANK,
        S=S,
        O=O,  # noqa: E741
        nbinom_fit=True)
    multiprocess_elapsed = time.time() - start_time

    assert len(mp_res) == len(sp_res) == 2
    assert single_process_elapsed / multiprocess_elapsed > 1.5


def test_match_components():
    # Generate some random components
    np.random.seed(0)
    K = 10
    components = np.random.dirichlet([0.5] * 96, K)
    scramble_idx = np.random.choice(K, K, False)
    scrambled_components = components[:, scramble_idx]
    found_idx = nmflib.nmf.match_components(scrambled_components, components)
    assert np.all(found_idx == scramble_idx)


def test_nmf_gof():
    """Test nmf.gof() function."""
    simple_nmf_data = SimpleNMFData()
    X = simple_nmf_data.X
    X_exp = np.matmul(SimpleNMFData.W_true, SimpleNMFData.H_true)

    # Observed is *exactly* as expected - P value should be 0 since no
    # simulation should get this high a likelihood.
    _, pval, data, _ = nmflib.nmf.gof(X_exp, X_exp, random_state=0)
    assert pval == 0.0
    assert np.all(np.array(data) == 1.0)

    # Observed is very far from expected - P value should be 0 since no
    # simulation should be always more likely than the expected value.
    _, pval, data, _ = nmflib.nmf.gof(X, X_exp * 100, random_state=0)
    assert pval == 0.0
    assert np.all(np.array(data) == 0.0)

    # Create 10 simulations and choose the middle one. That one should have
    # a midrange likelihood on average.
    sim_X = [scipy.stats.poisson.rvs(X_exp, random_state=0) for k in range(11)]
    sim_X_logliks = [
        np.sum(scipy.stats.poisson.logpmf(X, X_exp)) for X in sim_X
    ]
    sim_X = list(zip(sim_X_logliks, sim_X))
    sim_X = sorted(sim_X, key=lambda x: x[0])
    _, pval, _, _ = nmflib.nmf.gof(sim_X[5][1], X_exp, random_state=0)
    assert 0.2 <= pval <= 0.8


@synthetic_nmf_data_dir
@pytest.mark.slow
@pytest.mark.parallel
def test_parallel_gof(datafiles):
    """Make sure nmf.mpfit() works and is faster than just fit()."""
    true_r = 10
    synthetic_pcawg_data = SyntheticPCAWG(datafiles, true_r)
    X_obs = synthetic_pcawg_data.simulate()
    X_exp = synthetic_pcawg_data.X_exp

    # Measure time with and without multiprocessing.
    start_time = time.time()
    nmflib.nmf.gof(X_obs, X_exp, 100, r=true_r, n_processes=2, n_threads=1)
    multiprocess_elapsed = time.time() - start_time
    start_time = time.time()
    nmflib.nmf.gof(X_obs, X_exp, 200, r=true_r, n_processes=1, n_threads=1)
    single_process_elapsed = time.time() - start_time
    assert single_process_elapsed / multiprocess_elapsed > 1.5


def test_signatures_model():
    """Test SignaturesModel and SingleNMFModel classes."""
    X_exp = np.matmul(SimpleNMFData.W_true, SimpleNMFData.H_true)
    sim_X = scipy.stats.poisson.rvs(X_exp, random_state=0)
    sig_models = nmflib.nmf.SignaturesModel(sim_X, [1, 2, 3])
    sig_models.fit()
    assert (sig_models.fitted.loc[2, 'gof_pval'] >
            sig_models.fitted.loc[1, 'gof_pval'])
    assert (sig_models.fitted.loc[2, 'gof_pval'] >
            sig_models.fitted.loc[3, 'gof_pval'])

    # Explicitly check SingleNMFModel.__str__().
    assert (sig_models.fitted.loc[1, 'nmf_model'].__str__() ==
            'Poisson-NMF(M=4, N=4, K=1) *')


def test_h_confint():
    """Test LRT and confidence interval computation.

    Tests nmf.hk_confint() and nmf.hk_lrt() also.
    """
    simple_nmf_data = SimpleNMFData()
    W, H, _, _, _ = nmflib.nmf.fit(simple_nmf_data.X, SimpleNMFData.k)
    idx = nmflib.nmf.match_components(simple_nmf_data.W_true, W)
    W = W[:, idx]
    H = H[idx, :]

    # Sample 3 has none of signature 1 but only signature 2. Check that
    # - P value for exposure = 0 should be "insignificant".
    # - The lower confidence interval should include 0.
    # - The upper confidence interval should roughly have a LRT P-value of 0.05.
    sig_idx = 0
    sample_idx = 2
    x_obs = simple_nmf_data.X[:, sample_idx]
    h_hat = H[:, sample_idx]
    output_tuple = nmflib.nmf.hk_confint(x_obs, W, sig_idx, h_hat=h_hat)
    (_, _, _, pval, _, ml_loglik, restricted_loglik, _, cint_low, cint_high, _,
     _) = output_tuple
    assert pval > 0.2
    assert cint_low == 0
    target_f = nmflib.nmf._NMFProfileLoglikFitter(x_obs.reshape(-1, 1), W,
                                                  h_hat.reshape(-1, 1),
                                                  sig_idx)._profile_ml
    cint_high_loglik = target_f(cint_high)
    chi2_stat = 2 * (ml_loglik - cint_high_loglik)
    assert 0.949 <= scipy.stats.chi2.cdf(chi2_stat, 1) <= 0.951

    # Similarly, 3 has strong exposure of signature 2.
    # - P value for exposure = 0 should be "significant".
    # - The lower confidence interval should not 0.
    # - The lower and upper confidence intervals should roughly have a
    #   LRT P-value of 0.05.
    # - The restricted log-likelihood should be smaller than the unrestricted
    #   maximum likelihood.
    sig_idx = 1
    sample_idx = 2
    x_obs = simple_nmf_data.X[:, sample_idx]
    h_hat = H[:, sample_idx]
    output_tuple = nmflib.nmf.hk_confint(x_obs, W, sig_idx, h_hat=h_hat)
    (_, _, _, pval, _, ml_loglik, restricted_loglik, _, cint_low, cint_high, _,
     _) = output_tuple
    assert ml_loglik > restricted_loglik
    assert pval < 0.01
    assert cint_low > 0
    target_f = nmflib.nmf._NMFProfileLoglikFitter(x_obs.reshape(-1, 1), W,
                                                  h_hat.reshape(-1, 1),
                                                  sig_idx)._profile_ml
    cint_low_loglik = target_f(cint_low)
    chi2_stat = 2 * (ml_loglik - cint_low_loglik)
    assert 0.949 <= scipy.stats.chi2.cdf(chi2_stat, 1) <= 0.951
    cint_high_loglik = target_f(cint_high)
    chi2_stat = 2 * (ml_loglik - cint_high_loglik)
    assert 0.949 <= scipy.stats.chi2.cdf(chi2_stat, 1) <= 0.951
