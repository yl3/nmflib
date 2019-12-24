"""Test NMF code."""

import pytest

import logging
import numpy as np
import pandas as pd
import re
import scipy.stats
import sklearn.decomposition._nmf
import time

import nmflib.constants
import nmflib.nmf


class SimpleNMFData:
    """Three channels and two signatures.

    The first signature only creates mutations in the first two channels. The
    second signature creates mutations in all channels.
    """
    X = np.array([[15, 40, 12, 85], [15, 20, 0, 25], [0, 10, 6, 30],
                  [0, 20, 12, 60]])
    W_true = np.array([[.5, 0.4], [.5, 0], [0, 0.2], [0, 0.4]])
    H_true = np.array([[30, 40, 0, 50], [0, 50, 30, 150]])

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
    S = np.array([[1, 1, 0.5, 0.25], [1, 1, 0.5, 0.25], [1, 1, 0.5, 0.25],
                  [1, 1, 0.5, 0.25]])
    H_true = np.array([[30, 40, 0, 200], [0, 50, 60, 600]])


class SyntheticPCAWG():
    """Synthetic mutational signature convolutions from PCAWG.

    Offsets and scales are simulated.

    Args:
        random_state (int): Random state for simulating counts. Default: 0.
    """
    def __init__(self, datadir, r=None, random_state=0):
        self.r = r
        self.random_state = random_state
        self._load_syn_data(datadir)

    def simulate(self):  # noqa: E741
        """Simulate Poisson or nbinom (when r is provided) counts."""
        np.random.seed(self.random_state)
        if self.r is None:
            rvs = scipy.stats.poisson.rvs(self.X_exp)
        else:
            p = nmflib.nmf._nb_p(self.X_exp, self.r)
            rvs = scipy.stats.nbinom.rvs(self.r, p)
        return rvs

    def _load_syn_data(self, datadir):
        self.catalog = pd.read_csv(datadir + '/ground.truth.syn.catalog.csv.gz',
                                   index_col=0,
                                   header=0)
        self.sigs = pd.read_csv(datadir + '/ground.truth.syn.sigs.csv.gz',
                                index_col=[0, 1],
                                header=0)
        self.exposures = pd.read_csv(datadir +
                                     '/ground.truth.syn.exposures.csv.gz',
                                     index_col=0,
                                     header=0)
        X_exp = self.sigs.dot(self.exposures)
        S = scipy.stats.uniform.rvs(0.05,
                                    1 - 0.05,
                                    X_exp.shape,
                                    random_state=self.random_state)
        X_exp *= S
        O = scipy.stats.uniform.rvs(  # noqa: E741
            0.5,
            1 - 0.5,
            X_exp.shape,
            random_state=self.random_state)
        O *= X_exp.values  # noqa: E741
        X_exp -= O
        self.X_exp = X_exp
        self.S = S
        self.O = O  # noqa: E741


def test_fit_nmf(caplog):
    """Test fitting a regular NMF."""
    caplog.set_level(logging.INFO)
    W, H, r, n_iter, errors = nmflib.nmf.fit(SimpleNMFData.X,
                                             2,
                                             max_iter=10000,
                                             abstol=1e-10)
    SimpleNMFData.check_answer(W, H)

    # Test logging.
    W, H, r, n_iter, errors = nmflib.nmf.fit(SimpleNMFData.X,
                                             2,
                                             max_iter=21,
                                             abstol=1e-10,
                                             verbose=True)
    assert re.search(r"Iteration 10 after \S+ seconds, error: ", caplog.text)
    assert re.search(r"Iteration 20 after \S+ seconds, error: ", caplog.text)


def test_fit_nmf_scaled():
    """Test fitting NMF with a scaling matrix."""
    W, H, r, n_iter, errors = nmflib.nmf.fit(ScaledNMFData.X,
                                             2,
                                             ScaledNMFData.S,
                                             abstol=1e-10)
    ScaledNMFData.check_answer(W, H)


def test_iterate_nbinom_nmf_r_ml():
    """Test nmf._iterate_nbinom_nmf_r_ml() using simple data."""
    mu = 10
    true_r = 10
    p = nmflib.nmf._nb_p(mu, true_r)
    np.random.seed(0)
    x = scipy.stats.nbinom.rvs(true_r, p, size=int(1e5))
    r = nmflib.nmf._initialise_nb_r(x, mu)
    for _ in range(10):
        r = nmflib.nmf._iterate_nbinom_nmf_r_ml(x, mu, r)
    assert 9.9 < r < 10.1  # Should be ~10.0


def test_fit_nbinom_r_mm():
    """Test to make sure nmf.fit_nbinom_r_mm() works using simple data."""
    mu = 20
    true_r = 10
    p = nmflib.nmf._nb_p(mu, true_r)
    np.random.seed(0)
    x = scipy.stats.nbinom.rvs(true_r, p, size=int(1e5))
    r, _ = nmflib.nmf.fit_nbinom_r_mm(x, mu, len(x) - 1)
    assert 9.9 < r < 10.1  # Should be ~10.00


@pytest.mark.datafiles('test_data/ground.truth.syn.catalog.csv.gz',
                       'test_data/ground.truth.syn.sigs.csv.gz',
                       'test_data/ground.truth.syn.exposures.csv.gz')
def test_fit_nmf_monotonicity(datafiles):
    """Make sure nmf.fit() is monotonous."""
    datafiles = str(datafiles)
    true_r = 10
    synthetic_pcawg_data = SyntheticPCAWG(datafiles, true_r)
    X_obs = synthetic_pcawg_data.simulate()
    S = synthetic_pcawg_data.S
    O = synthetic_pcawg_data.O  # noqa: E741
    TARGET_RANK = 20

    # Test the monotonicity of overall errors.
    W, H, r, n_iter, errors = nmflib.nmf.fit(X_obs,
                                             TARGET_RANK,
                                             S,
                                             O,
                                             nbinom_fit=True,
                                             verbose=True)

    # Skip check on iterations 1-5, which are typically fitted using method
    # of moments and are not guaranteed to increase log-likelihood.
    for i in range(5, len(errors) - 1):
        assert errors[i + 1] <= errors[i]


@pytest.mark.datafiles('test_data/ground.truth.syn.catalog.csv.gz',
                       'test_data/ground.truth.syn.sigs.csv.gz',
                       'test_data/ground.truth.syn.exposures.csv.gz')
def test_nmf_updates_monotonicity(datafiles):
    """Make sure that each individual W, H and r update step is monotonous."""
    datafiles = str(datafiles)
    true_r = 10
    synthetic_pcawg_data = SyntheticPCAWG(datafiles, true_r)
    X_obs = synthetic_pcawg_data.simulate()
    S = synthetic_pcawg_data.S
    O = synthetic_pcawg_data.O  # noqa: E741
    TARGET_RANK = 20

    # Run 10 iterations and make sure each individual update is monotonous.
    W, H = sklearn.decomposition._nmf._initialize_nmf(
        X_obs,
        TARGET_RANK,
        'random',
        random_state=synthetic_pcawg_data.random_state)
    X_exp = nmflib.nmf._nmf_mu(W, H, S, O)
    r = nmflib.nmf._initialise_nb_r(X_obs, X_exp)
    loglik = prev_loglik = np.sum(nmflib.nmf.loglik(X_obs, X_exp, r))
    for _ in range(10):
        W = nmflib.nmf._multiplicative_update_W(X_obs, W, H, S, O, r=r)
        X_exp = nmflib.nmf._nmf_mu(W, H, S, O)
        loglik = np.sum(nmflib.nmf.loglik(X_obs, X_exp, r))
        assert loglik > prev_loglik
        prev_loglik = loglik

        H = nmflib.nmf._multiplicative_update_H(X_obs, W, H, S, O, r=r)
        X_exp = nmflib.nmf._nmf_mu(W, H, S, O)
        loglik = np.sum(nmflib.nmf.loglik(X_obs, X_exp, r))
        assert loglik > prev_loglik
        prev_loglik = loglik

        X_exp = nmflib.nmf._nmf_mu(W, H, S, O)
        r = nmflib.nmf._iterate_nbinom_nmf_r_ml(X_obs, X_exp, r)
        loglik = np.sum(nmflib.nmf.loglik(X_obs, X_exp, r))
        assert loglik > prev_loglik
        prev_loglik = loglik


def test_match_signatures():
    # Generate some random signatures
    np.random.seed(0)
    K = 10
    signatures = np.random.dirichlet([0.5] * 96, K)
    scramble_idx = np.random.choice(K, K, False)
    scrambled_signatures = signatures[:, scramble_idx]
    found_idx = nmflib.nmf.match_signatures(scrambled_signatures, signatures)
    assert np.all(found_idx == scramble_idx)


def test_context_rate_matrix():
    # Test with sample indices.
    sample_names = ['s0', 's1', 's2', 's3']
    scale_mat = nmflib.nmf.context_rate_matrix(
        pd.Series([False, True, False, True], index=sample_names), False)
    assert all(scale_mat.columns == sample_names)
    scale_mat_trinucs = scale_mat.index.get_level_values(1)
    for k in [0, 2]:
        assert all(scale_mat[sample_names[k]].values ==
                   nmflib.constants.HUMAN_GENOME_TRINUCS[scale_mat_trinucs])
    for k in [1, 3]:
        assert all(scale_mat[sample_names[k]].values ==
                   nmflib.constants.HUMAN_EXOME_TRINUCS[scale_mat_trinucs])

    # Test relative rates
    scale_mat = nmflib.nmf.context_rate_matrix([False, True, False, True], True)
    scale_mat_trinucs = scale_mat.index.get_level_values(1)
    gw_relative = pd.Series(np.repeat(
        1.0, len(nmflib.constants.HUMAN_GENOME_TRINUCS)),
                            index=nmflib.constants.HUMAN_GENOME_TRINUCS.index)
    ew_relative = (nmflib.constants.HUMAN_EXOME_TRINUCS /
                   nmflib.constants.HUMAN_GENOME_TRINUCS)
    for k in [0, 2]:
        assert all(scale_mat[k].values == gw_relative[scale_mat_trinucs])
    for k in [1, 3]:
        assert all(scale_mat[k].values == ew_relative[scale_mat_trinucs])

    # Test without sample indices.
    scale_mat = nmflib.nmf.context_rate_matrix([False, True, False, True],
                                               False)
    assert all(scale_mat.columns == [0, 1, 2, 3])


def test_nmf_gof():
    """Test nmf.gof() function."""
    X = SimpleNMFData.X
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

    # Create 10 simulations and choose the middle one. That one should have a
    # midrange likelihood on average.
    sim_X = [scipy.stats.poisson.rvs(X_exp, random_state=0) for k in range(11)]
    sim_X_logliks = [
        np.sum(scipy.stats.poisson.logpmf(X, X_exp)) for X in sim_X
    ]
    sim_X = list(zip(sim_X_logliks, sim_X))
    sim_X = sorted(sim_X, key=lambda x: x[0])
    _, pval, _, _ = nmflib.nmf.gof(sim_X[5][1], X_exp, random_state=0)
    assert 0.2 <= pval <= 0.8


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


@pytest.mark.datafiles('test_data/ground.truth.syn.catalog.csv.gz',
                       'test_data/ground.truth.syn.sigs.csv.gz',
                       'test_data/ground.truth.syn.exposures.csv.gz')
def test_mpfit(datafiles):
    """Make sure nmf.mpfit() works and is faster than just fit()."""
    datafiles = str(datafiles)
    true_r = 10
    synthetic_pcawg_data = SyntheticPCAWG(datafiles, true_r)
    X_obs = synthetic_pcawg_data.simulate()
    S = synthetic_pcawg_data.S
    O = synthetic_pcawg_data.O  # noqa: E741
    TARGET_RANK = 20

    # Measure time with and without multiprocessing.
    start_time = time.time()
    mp_res = nmflib.nmf.mpfit(2,
                              X_obs,
                              TARGET_RANK,
                              S,
                              O,
                              True,
                              verbose=True,
                              random_state=0,
                              n_processes=2)
    multiprocess_elapsed = time.time() - start_time
    start_time = time.time()
    sp_res = []
    for i in range(2):
        sp_res.append(
            nmflib.nmf.fit(X_obs,
                           TARGET_RANK,
                           S,
                           O,
                           True,
                           verbose=True,
                           random_state=i))
    single_process_elapsed = time.time() - start_time
    assert len(mp_res) == len(sp_res) == 2
    assert single_process_elapsed / multiprocess_elapsed > 1.2


@pytest.mark.datafiles('test_data/ground.truth.syn.catalog.csv.gz',
                       'test_data/ground.truth.syn.sigs.csv.gz',
                       'test_data/ground.truth.syn.exposures.csv.gz')
def test_parallel_gof(datafiles):
    """Make sure nmf.mpfit() works and is faster than just fit()."""
    datafiles = str(datafiles)
    true_r = 10
    synthetic_pcawg_data = SyntheticPCAWG(datafiles, true_r)
    X_obs = synthetic_pcawg_data.simulate()
    X_exp = synthetic_pcawg_data.X_exp

    # Measure time with and without multiprocessing.
    start_time = time.time()
    mp_res = nmflib.nmf.gof(X_obs, X_exp, 100, r=true_r, n_processes=2)
    multiprocess_elapsed = time.time() - start_time
    start_time = time.time()
    sp_res = nmflib.nmf.gof(X_obs, X_exp, 200, r=true_r, n_processes=1)
    single_process_elapsed = time.time() - start_time
    assert single_process_elapsed / multiprocess_elapsed > 1.2


def test_validate_is_ndarray():
    """Test nmf._validate_is_ndarray()."""
    assert nmflib.nmf._validate_is_ndarray(None) is None
    arr = np.arange(6).reshape(3, 2)
    df = pd.DataFrame(arr)
    assert isinstance(nmflib.nmf._validate_is_ndarray(arr), np.ndarray)
    assert isinstance(nmflib.nmf._validate_is_ndarray(df), np.ndarray)
    with pytest.raises(ValueError):
        nmflib.nmf._validate_is_ndarray("Not an array")
