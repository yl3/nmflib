"""Nonnegative matrix factorisation code for count data."""
# Copyright (C) 2019 Yilong Li (yilong.li.yl3@gmail.com) - All Rights Reserved

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.special import digamma, polygamma
import sklearn.decomposition._nmf
import sklearn.metrics.pairwise
import sys

import logging
import time

import nmflib.constants


def _ensure_pos(arr, epsilon=np.finfo(np.float32).eps):
    """Replace zeroes in arr with epsilon in-place."""
    sel = arr == 0.0
    arr[sel] = epsilon


def _validate_is_ndarray(arr):
    """Return a numpy.ndarray version of the array."""
    if arr is None:
        return None
    elif isinstance(arr, pd.DataFrame):
        return arr.values
    elif isinstance(arr, np.ndarray):
        return arr
    else:
        raise ValueError('Expected numpy.ndarray or pandas.DataFrame as type.')


def _multiplicative_update_W(X, W, H, S=None, O=None, r=None):  # noqa: E741
    """Multiplicative KL-update for W.

    Multiplicative update for W to maximise E[X] = (WH + O) * S.

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        W (numpy.ndarray): A nonnegative matrix of shape (M, K) for current W.
        H (numpy.ndarray): A nonnegative matrix of shape (K, N) for current H.
        S (numpy.ndarray): A matrix of values [0, 1] of shape (M, N) indicating
            the proportion of mutations of each of the M contexts that cannot be
            observed.
        O (numpy.ndarray): A matrix of nonnegative values of shape (M, N),
            representing the additive offset term.
        r (float): Non-negative value that, if provided, is used as the
            overdispersion parameter for a negative binomial NMF model.

    Returns:
        numpy.ndarray: The multiplicative update for W.
    """
    # Precompute some helper variables.
    WHO = np.matmul(W, H)
    if O is not None:
        WHO += O
    if S is not None:
        WHOS = WHO * S
    else:
        WHOS = WHO
    if r is not None:
        WHOSr = WHOS + r
    else:
        WHOSr = WHOS
    _ensure_pos(WHO)
    _ensure_pos(WHOS)
    _ensure_pos(WHOSr)

    # Compute numerator.
    if r is not None:
        if S is not None:
            numerator = np.matmul(
                np.divide(X, WHO) - np.divide(X * S, WHOSr), H.T)
        else:
            numerator = np.matmul(np.divide(X, WHO) - np.divide(X, WHOSr), H.T)
    else:
        numerator = np.matmul(np.divide(X, WHO), H.T)
    _ensure_pos(numerator)

    # Compute denominator.
    if r is None and S is None:
        # Use shorthand for computing W column sums.
        H_sum = np.sum(H, axis=1)
        denominator = H_sum[np.newaxis, :]
    elif r is not None and S is None:
        denominator = np.matmul(r / WHOSr, H.T)
    elif r is not None and S is not None:
        denominator = np.matmul(r * S / WHOSr, H.T)
    else:
        # r is None and S is not None
        denominator = np.matmul(S, H.T)
    _ensure_pos(denominator)

    # Compute the update
    numerator /= denominator
    delta_W = numerator
    new_W = W * delta_W

    return new_W


def _multiplicative_update_H(X, W, H, S=None, O=None, r=None):  # noqa: 471
    """Multiplicative KL-update for H.

    Multiplicative update for H to maximise E[X] = (WH + O) * S.

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        W (numpy.ndarray): A nonnegative matrix of shape (M, K) for current W.
        H (numpy.ndarray): A nonnegative matrix of shape (K, N) for current H.
        S (numpy.ndarray): A matrix of values [0, 1] of shape (M, N) indicating
            the proportion of mutations of each of the M contexts that cannot be
            observed.
        O (numpy.ndarray): A matrix of nonnegative values of shape (M, N),
            representing the additive offset term.
        r (float): Non-negative value that, if provided, is used as the
            overdispersion parameter for a negative binomial NMF model.

    Returns:
        numpy.ndarray: The multiplicative update for H.
    """
    # Precompute some helper variables.
    WHO = np.matmul(W, H)
    if O is not None:
        WHO += O
    if S is not None:
        WHOS = WHO * S
    else:
        WHOS = WHO
    if r is not None:
        WHOSr = WHOS + r
    else:
        WHOSr = WHOS
    _ensure_pos(WHO)
    _ensure_pos(WHOS)
    _ensure_pos(WHOSr)

    # Compute numerator.
    if r is not None:
        if S is not None:
            numerator = np.matmul(W.T,
                                  np.divide(X, WHO) - np.divide(X * S, WHOSr))
        else:
            numerator = np.matmul(W.T, np.divide(X, WHO) - np.divide(X, WHOSr))
    else:
        numerator = np.matmul(W.T, np.divide(X, WHO))
    _ensure_pos(numerator)

    # Compute denominator.
    if r is None and S is None:
        # Use shorthand for computing W column sums.
        W_sum = np.sum(W, axis=0)
        denominator = W_sum[:, np.newaxis]
    elif r is not None and S is None:
        denominator = np.matmul(W.T, r / WHOSr)
    elif r is not None and S is not None:
        denominator = np.matmul(W.T, r * S / WHOSr)
    else:
        # r is None and S is not None
        denominator = np.matmul(W.T, S)
    _ensure_pos(denominator)

    # Compute the update
    numerator /= denominator
    delta_H = numerator
    new_H = H * delta_H

    return new_H


def _initialise_nb_r(X, mu):
    """Method of moment initialisation of the nbinom dispersion parameter."""
    numerator = np.sum(X.shape)
    denominator = np.sum((X / mu - 1)**2)
    return numerator / denominator


def _nb_r_score(r, X, mu):
    """Score function for the negative binomial theta parameter."""
    elems = np.prod(X.shape)
    score = (elems * (-digamma(r) + np.log(r) + 1) +
             np.sum(digamma(X + r) - np.log(mu + r) - (X + r) / (mu + r)))
    return score


def _nb_r_info(r, X, mu):
    """Fisher information for the negative binomial theta parameter."""
    def trigamma(x):
        return polygamma(1, x)

    elems = np.prod(X.shape)
    info = (elems * (-trigamma(r) + 1 / r) +
            np.sum(trigamma(X + r) - 2 / (mu + r) + (X + r) / ((mu + r)**2)))
    # For numerical stability
    if info > 0:
        logging.warning('Got info > 0. Replacing with info <- -info.')
        info = -info
    return info


def _iterate_nbinom_nmf_r_ml(X, mu, r):
    """Use Newton's method to iterate negative binomial dispersion parameter.

    Args:
        X (numpy.ndarray): An array of counts.
        mu (numpy.ndarray): The expected value for each observed count in X.
        r (float): The current value of the negative binomial dispersion
            parameter.

    Returns:
        float: The next iterated value of r.
    """

    score = _nb_r_score(r, X, mu)
    info = _nb_r_info(r, X, mu)
    new_r = r - score / info
    while new_r <= 0:
        # For numerical stability
        new_r = r - score / info / 2
    return new_r


def _nb_p(mu, r):
    """Calculate the negative binomial p from a mean and a dispersion parameter.
    """
    # p has the same shape as mu.
    # We need to do subtrace the value from 1, since Wikipedia counts number of
    # successes, but scipy counts number of failures.
    p = 1 - mu / (r + mu)
    return p


def _nmf_mu(W, H, S=None, O=None):  # noqa: E741
    """Helper function for computing E[X] = (WH + O) * S."""
    WHOS = np.matmul(W, H)
    if O is not None:
        WHOS += O
    if S is not None:
        WHOS *= S
    return WHOS


def _iterate_nmf_fit(
        X,
        W,
        H,
        S=None,
        O=None,  # noqa: 471
        r=None,
        r_fit_method=None):
    """Perform a single iteration of W, H and r updates.

    The goal is to approximate X ~ nbinom((WH + O) * S, r).

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        W (numpy.ndarray): A nonnegative matrix of shape (M, K) for current W.
        H (numpy.ndarray): A nonnegative matrix of shape (K, N) for current H.
        S (numpy.ndarray): A matrix of values [0, 1] of shape (M, N) indicating
            the proportion of mutations of each of the M contexts that cannot be
            observed.
        O (numpy.ndarray): A matrix of nonnegative values of shape (M, N),
            representing the additive offset term.
        r (float): Non-negative value that, if provided, is used as the
            overdispersion parameter for a negative binomial NMF model.
        r_fit_method (str): Method for fitting overdispersion parameter r. If
            None, fitting is not done. Otherwise, must either be 'ml' for
            maximum likelihood or 'mm' for method of moments.

    Returns:
        numpy.ndarray: The updated W.
        numpy.ndarray: The updated H.
        float: The updated r.
    """
    if r_fit_method not in (None, 'ml', 'mm'):
        raise ValueError('r_fit_method must be in (None, "ml", "mm").')
    W = _multiplicative_update_W(X, W, H, S, O, r)
    H = _multiplicative_update_H(X, W, H, S, O, r)
    mu = _nmf_mu(W, H, S, O)
    if r_fit_method == 'mm':
        r, _ = fit_nbinom_nmf_r_mm(X, mu, W.shape[1])
    elif r_fit_method == 'ml':
        r, _ = fit_nbinom_nmf_r_ml(X, mu, r)
    else:
        # Should never get here.
        pass
    return W, H, r


def fit_nbinom_r_mm(X, mu, dof, reltol=0.001):
    """Fit the negative binomial dispersion parameter via method of moments."""
    prev_r = _initialise_nb_r(X, mu)
    iter = 0
    while True:
        iter += 1
        numerator = np.sum((X - mu)**2 / (mu + mu**2 / prev_r)) - dof
        denominator = np.sum((X - mu)**2 / (mu + prev_r)**2)
        delta = numerator / denominator
        r = prev_r - delta
        if abs(r - prev_r) / prev_r < reltol:
            break
        prev_r = r
    return r, iter


def fit_nbinom_nmf_r_mm(X, mu, k, reltol=0.001):
    """Fit the negative binomial NMF dispersion parameter via method of moments.
    """
    free_params = np.prod(X.shape) - k * (np.sum(X.shape) - 1)
    r, iter = fit_nbinom_r_mm(X, mu, free_params, reltol)
    return r, iter


def fit_nbinom_nmf_r_ml(X, mu, r_init, reltol=0.001):
    """
    Fit the negative binomial NMF dispersion parameter via maximum likelihood.
    """
    prev_r = r_init
    n_iter = 0
    while True:
        n_iter += 1
        r = _iterate_nbinom_nmf_r_ml(X, mu, prev_r)
        if abs(r - prev_r) / prev_r < reltol:
            break
        prev_r = r
    return r, n_iter


def loglik(X, X_exp, r=None):
    """Compute poisson or negative binomial log-likelihood.

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        X_exp (numpy.ndarray): A nonnegative matrix for expected values of X.
        r (float): If provided, then negative binomial log-likelihood is
            computed using `r` as the dispersion parameter.

    Returns:
        numpy.ndarray: The log-likelihood for each element in X.
    """
    if r is None:
        logliks = scipy.stats.poisson.logpmf(X, X_exp)
    else:
        p = _nb_p(X_exp, r)  # p has the same shape as X_exp.
        logliks = scipy.stats.nbinom.logpmf(X, r, p)
    return logliks


def calc_aic(loglik, k):
    """Compute AIC.

    Args:
        loglik (float): Log-likelihood of the current model.
        k (int): Number of free parameters in the current model.

    Returns:
        float: The AIC value.
    """
    aic = 2 * k - 2 * loglik
    return aic


def calc_bic(loglik, k, n):
    """Compute BIC.

    Args:
        loglik (float): Log-likelihood of the current model.
        k (int): Number of free parameters in the current model.
        n (int): Number of observations, i.e. number of samples.

    Returns:
        float: The BIC value.
    """
    bic = np.log(n) * k - 2 * loglik
    return bic


def gof(X, X_exp, sim_count=None, random_state=None, r=None):
    """Bootstrapped goodness-of-fit for count data NMF.

    Given that X ~ Poisson(X_exp), compute the P value for each column in X
    given X_exp. Then Kolmogorov-Smirnov test to estimate goodness-of-fit.

    Args:
        X (array-like): Observed counts: a nonnegative integer matrix of shape
            (M, N).
        X_exp (array-like): A nonnegative expected values matrix of shape
            (M, N).
        sim_count (int): How many simulated instances of X should be generated?
            Default: 100.
        random_state (int): Random seed.
        r (float): A positive dispersion parameter. If None, then a Poisson
            model is assumed.

    Returns:
        gof_D (float): KS test statistic.
        gof_pval (float): Goodness-of-fit estimate for the entire matrix.
        gof_data (numpy.ndarray): Goodness-of-fit p-values for each sample
            (i.e. column of X) individually.
        sim_logliks (numpy.ndarray): A matrix of log-likelihoods for simulated
            observations for each sample with shape (`sim_count`, X.shape[1]).
    """
    if random_state is not None:
        np.random.seed(random_state)
    if sim_count is None:
        sim_count = 100

    # Observed log-likelihood of each sample - represented as a row vector.
    obs_loglik_mat = loglik(X, X_exp, r)
    obs_loglik_rowvec = obs_loglik_mat.sum(axis=0).reshape(1, -1)

    # Simulated log-likelihood matrices.
    sim_logliks = []
    for k in range(sim_count):
        if r is None:
            sim_X = scipy.stats.poisson.rvs(X_exp)
            sim_logliks.append(
                scipy.stats.poisson.logpmf(sim_X, X_exp).sum(axis=0))
        else:
            p = _nb_p(X_exp, r)
            sim_X = scipy.stats.nbinom.rvs(r, p, size=X.shape)
            sim_logliks.append(
                scipy.stats.nbinom.logpmf(sim_X, r, p).sum(axis=0))

    sim_logliks = np.array(sim_logliks)

    # Calculate empirical P values for row sample.
    signif_simuls = np.sum(sim_logliks <= obs_loglik_rowvec, axis=0)
    sample_pvals = signif_simuls / sim_count

    # Calculate overall goodness-of-fit P value.
    gof_D, gof_pval = scipy.stats.kstest(sample_pvals, 'uniform')
    return gof_D, gof_pval, sample_pvals, sim_logliks


def fit(
        X,
        k,
        S=None,
        O=None,  # noqa: E741
        nbinom_fit=False,
        max_iter=200,
        tol=1e-4,
        verbose=False,
        random_state=None):
    """Fit KL-NMF using multiplicative updates.

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        k (int): A positive integer for the number of signatures to use.
        S (numpy.ndarray): A matrix of values [0, 1] of shape (M, N) indicating
            the proportion of mutations of each of the M contexts that cannot be
            observed.
        O (numpy.ndarray): A matrix of nonnegative values of shape (M, N),
            representing the additive offset term.
        nbinom_fit (bool): Whether to fit a negative binomial model or a default
            Poisson model.
        max_iter (int): Maximum number of iterations to use.
        tol (float): Relative tolerance for convergence.
        verbose (bool): Whether to print progress updates every 10 iterations.
        random_state (int): The random seed to use in NMF initialisation.

    Returns:
        numpy.ndarray: The fitted W matrix of shape (M, k). The matrix is scaled
            to column sums of 1.
        numpy.ndarray: The fitted H matrix of shape (k, N). The matrix is scaled
            correspondingly to W matrix scaling.
        float: The overdispersion parameter r. Returns None if the Poisson model
            was fitted.
        int: Number of iterations used.
        list: A list of errors recorded every 10 iterations.
    """
    X = _validate_is_ndarray(X)
    S = _validate_is_ndarray(S)
    O = _validate_is_ndarray(O)  # noqa: E741
    W, H = sklearn.decomposition._nmf._initialize_nmf(X,
                                                      k,
                                                      'random',
                                                      random_state=random_state)
    r = None  # First iteration is fitted with Poissonian overdispersion.

    # Set up initial values.
    start_time = time.time()
    error_at_init = previous_error = None
    errors = []

    n_iter = 0
    while n_iter <= max_iter:
        if nbinom_fit:
            W, H, r = _iterate_nmf_fit(X, W, H, S, O, r, r_fit_method='mm')
        else:
            W, H, r = _iterate_nmf_fit(X, W, H, S, O, r, r_fit_method=None)
        n_iter += 1

        # Test for convergence every 10 iterations.
        if n_iter % 10 == 0:
            X_exp = _nmf_mu(W, H, S, O)
            error = -np.sum(loglik(X, X_exp, r))
            errors.append(error)
            if verbose:
                elapsed = time.time() - start_time
                logging.info(
                    "Iteration {} after {:.3f} seconds, error: {}".format(
                        n_iter, elapsed, error))
            if error_at_init is None:
                error_at_init = error
            elif (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    if nbinom_fit and n_iter <= max_iter:
        if verbose:
            logging.info("Converged with method of moments fit. Proceeding "
                         "with maximum likelihood.")

        while n_iter <= max_iter:
            W, H, r = _iterate_nmf_fit(X, W, H, S, O, r, r_fit_method='ml')
            n_iter += 1

            # Test for convergence every 10 iterations.
            if n_iter % 10 == 0:
                X_exp = _nmf_mu(W, H, S, O)
                error = -np.sum(loglik(X, X_exp, r))
                errors.append(error)
                if verbose:
                    elapsed = time.time() - start_time
                    logging.info(
                        "Iteration {} after {:.3f} seconds, error: {}".format(
                            n_iter, elapsed, error))
                if (previous_error - error) / error_at_init < tol:
                    break
                previous_error = error

    # Scale W and H such that W columns sum to 1.
    W_colsums = np.sum(W, axis=0)
    W /= W_colsums[np.newaxis, :]
    H *= W_colsums[:, np.newaxis]
    return W, H, r, n_iter, errors


def match_signatures(W1, W2):
    """Match mutational signatures (columns) between W1 and W2.

    The distance metric used in cosine similarity. Matching is done using
    Hungarian algorithm.

    Args:
        W1 (array-like): An array of shape (<mutation contexts>, <signatures>).
        W2 (array-like): An array of shape (<mutation contexts>, <signatures>).

    Returns:
        numpy.ndarray: An array of indices matching columns of W2 with columns
            of W1.
    """
    dist_mat = sklearn.metrics.pairwise.cosine_distances(W1.T, W2.T)
    _, W2_idx = scipy.optimize.linear_sum_assignment(dist_mat)
    return W2_idx


def mut_context_df(relative=True):
    """Compute the opportunity mutation type in each context.

    Currently only supports the 96 trinucleotide pyrimidine strand SNV types.

    Args:
        relative (bool): Whether to return the absolute opportunity counts. If
            False, the returned values are in [0, 1] and correspond to the
            proportion of each context in the whole genome that is (estimated to
            be) observable in whole-exome sequencing.

    Returns:
        pandas.DataFrame: A data frame giving the contexts counts in the genome
            or the exome.
    """
    # Merge the absolute mutation counts.
    mut_context_types = pd.MultiIndex.from_product(
        [nmflib.constants.MUTS_6, nmflib.constants.HUMAN_GENOME_TRINUCS.index])
    mut_context_types = mut_context_types.to_frame()
    sel = mut_context_types[0].str[0] == mut_context_types[1].str[1]
    mut_context_types = mut_context_types.loc[sel]

    if relative:
        gw_rates = pd.Series([1.0] * len(nmflib.constants.HUMAN_GENOME_TRINUCS),
                             index=nmflib.constants.HUMAN_GENOME_TRINUCS.index)
        ew_rates = pd.Series(nmflib.constants.HUMAN_EXOME_TRINUCS /
                             nmflib.constants.HUMAN_GENOME_TRINUCS)
    else:
        gw_rates = nmflib.constants.HUMAN_GENOME_TRINUCS
        ew_rates = nmflib.constants.HUMAN_EXOME_TRINUCS
    mut_context_types = mut_context_types.merge(gw_rates.to_frame(),
                                                left_on=1,
                                                right_index=True)
    mut_context_types = mut_context_types.merge(ew_rates.to_frame(),
                                                left_on=1,
                                                right_index=True)

    # After merging the counts, drop the first two columns that correspond to
    # the index anyways. Then name the remaining columns properly.
    mut_context_types.drop(columns=mut_context_types.columns[:2], inplace=True)
    mut_context_types.columns = ['gw_rate', 'ew_rate']

    return mut_context_types


def context_rate_matrix(sample_is_exome, relative=True):
    """Make context matrix S for mutation opportunity.

    Make a matrix S such that each column corresponds to mutation context counts
    depending of a sample depending on whether the sample underwent whole-exome
    or whole-genome sequencing.

    Currently only SNVs in trinucleotide context (96 types) is supported.

    Args:
        sample_is_exome (array-like): A boolean array for whether the sample
            underwent exome or whole-genome sequencing.
        relative (bool): Whether to return the absolute opportunity counts. If
            False, the returned values are in [0, 1] and correspond to the
            proportion of each context in the whole genome that is (estimated to
            be) observable in whole-exome sequencing.

    Returns:
        pandas.DataFrame: A data frame of shape (96, len(sample_is_exome)) of
            the relative or absolute opportunities).
    """
    context_counts = mut_context_df(relative)
    counts_to_use = [
        context_counts['ew_rate'] if x else context_counts['gw_rate']
        for x in sample_is_exome
    ]
    scale_mat = pd.DataFrame(counts_to_use).T
    if isinstance(sample_is_exome, pd.Series):
        scale_mat.columns = sample_is_exome.index
    else:
        scale_mat = scale_mat.T.reset_index(drop=True).T
    return scale_mat


class SingleNMFModel:
    """An NMF model for a mutation count matrix.

    Args:
        X (numpy.ndarray): A non-negative mutation counts matrix of shape
            (M, N).
        rank (int): Rank of the NMF model.
        S (numpy.ndarray): A matrix of values [0, 1] of shape (M, N) indicating
            the proportion of mutations of each of the M contexts that cannot be
            observed. See also :func:`nmflib.nmf.context_rate_matrix`.
        O (numpy.ndarray): A matrix of nonnegative values of shape (M, N),
            representing the additive offset term.
        nbinom (bool): Whether to fit a negative binomial model or a default
            Poisson model.
        random_inits (int): Number of random initialisations per rank to fit.
            The best fit of each random initialisation is kept.
        gof_sim_count (int): Number of simulations for the goodness-of-fit
            computation. Defaults to the value used in :func:`nmflib.nmf.gof`.

    Attributes:
        X (numpy.ndarray): A non-negative mutation counts matrix of shape
            (M, N).
        rank (int): Rank of the NMF model.
        S (numpy.ndarray): A matrix of values [0, 1] of shape (M, N) indicating
            the proportion of mutations of each of the M contexts that cannot be
            observed. See also :func:`nmflib.nmf.context_rate_matrix`.
        O (numpy.ndarray): A matrix of nonnegative values of shape (M, N),
            representing the additive offset term.
        nbinom (bool): Whether to fit a negative binomial model or a default
            Poisson model.
        random_inits (int): Number of random initialisations per rank to fit.
            The best fit of each random initialisation is kept.
        gof_sim_count (int): Number of simulations for the goodness-of-fit
            computation.
        fitted (dict): A dictionary of keys ('W', 'H', 'r', 'gof', 'sample_gof',
            'n_iter', 'errors', 'elapsed') for the fitted W matrix of shape
            (M, k), the fitted H matrix of shape (k, N), the overall simulated
            goodness-of-fit estimate of the model, the simulated
            goodness-of-fit values for each sample, the number of iterations,
            the errors and the total time spent to fit all random
            initialisations, respectively.
    """
    def __init__(
            self,
            X,
            rank,
            S=None,
            O=None,  # noqa: 471
            nbinom=False,
            random_inits=20,
            gof_sim_count=None):
        self.X = X
        self.rank = rank
        self.S = S
        self.O = O  # noqa: 471
        self.nbinom = nbinom
        self.random_inits = random_inits
        self.gof_sim_count = gof_sim_count
        self.fitted = None

    def fit(self, print_dots=True, **kwargs):
        """Fit the current NMF model and compute goodness-of-fit.

        The results are stored in :attr:`fitted`.

        Args:
            print_dots (bool): Whether to print a dot for each random
                initiation to STDERR.
            **kwargs: Keyword arguments for :func:`nmflib.nmf.fit`.
        """
        start_time = time.time()

        # Compute the best decomposition.
        errors_best = None
        for _ in range(self.random_inits):
            if print_dots:
                sys.stderr.write('.')
            W, H, r, n_iter, errors = fit(self.X, self.rank, self.S, self.O,
                                          self.nbinom, **kwargs)
            if errors_best is None or errors[-1] < errors_best[-1]:
                W_best, H_best, r_best, n_iter_best = W, H, r, n_iter
                errors_best = errors
        sys.stderr.write('\n')

        # Calculate goodness-of-fit.
        X_pred = _nmf_mu(W_best, H_best, self.S, self.O)
        gof_D, gof_pval, sample_gof_pval, sim_logliks = gof(self.X,
                                                            X_pred,
                                                            self.gof_sim_count,
                                                            r=r_best)

        # Calculate AIC and BIC.
        fitted_loglik = np.sum(loglik(self.X, X_pred, r_best))
        M, N, K = self.X.shape + (self.rank, )
        param_count = (M - 1 + N) * K
        if self.nbinom:
            param_count += 1
        aic = calc_aic(fitted_loglik, param_count)
        bic = calc_bic(fitted_loglik, param_count, np.prod(self.X.shape))

        elapsed = time.time() - start_time

        # Save results.
        self.fitted = {
            'W': W_best,
            'H': H_best,
            'r': r_best,
            'gof_D': gof_D,
            'gof_pval': gof_pval,
            'sample_gof': sample_gof_pval,
            'n_iter': n_iter_best,
            'errors': errors_best,
            'aic': aic,
            'bic': bic,
            'elapsed': elapsed
        }

    def __str__(self):
        M, N = self.X.shape
        if self.nbinom is False:
            out_str = f'Poisson-NMF(M={M}, N={N}, K={self.rank})'
        else:
            out_str = f'nbinom-NMF(M={M}, N={N}, K={self.rank})'
        if self.fitted is not None:
            out_str += ' *'
        return out_str


class SignaturesModel:
    """Generic class for fitting signatures and model selection.

    Args:
        X (numpy.ndarray): A non-negative mutation counts matrix of shape
            (M, N).
        ranks_to_test (iterable): A list of ranks to test.
        S (numpy.ndarray): A matrix of values [0, 1] of shape (M, N) indicating
            the proportion of mutations of each of the M contexts that cannot be
            observed. See also :func:`nmflib.nmf.context_rate_matrix`.
        O (numpy.ndarray): A matrix of nonnegative values of shape (M, N),
            representing the additive offset term.
        nbinom (bool): Whether to fit a negative binomial model or a default
            Poisson model.
        random_inits (int): Number of random initialisations per rank to fit.
            The best fit of each random initialisation is kept.
        gof_sim_count (int): Number of simulations for the goodness-of-fit
            computation. Defaults to the value used in :func:`nmflib.nmf.gof`.

    Attributes:
        X (numpy.ndarray): A non-negative mutation counts matrix of shape
            (M, N).
        ranks_to_test (iterable): A list of ranks to test.
        S (numpy.ndarray): A matrix of values [0, 1] of shape (M, N) indicating
            the proportion of mutations of each of the M contexts that cannot be
            observed. See also :func:`nmflib.nmf.context_rate_matrix`.
        O (numpy.ndarray): A matrix of nonnegative values of shape (M, N),
            representing the additive offset term.
        nbinom (bool): Whether to fit a negative binomial model or a default
            Poisson model.
        random_inits (int): Number of random initialisations per rank to fit.
            The best fit of each random initialisation is kept.
        gof_sim_count (int): Number of simulations for the goodness-of-fit
            computation.
        fitted (pandas.DataFrame): A data frame indexed by the ranks to be
            tested with columns for the actual model and the model fitting
            outputs.
    """
    def __init__(
            self,
            X,
            ranks_to_test,
            S=None,
            O=None,  # noqa:471
            nbinom=False,
            random_inits=20,
            gof_sim_count=None):
        self.X = X
        self.ranks_to_test = ranks_to_test
        self.S = S
        self.O = O  # noqa: 471
        self.nbinom = nbinom
        self.random_inits = random_inits
        self.gof_sim_count = gof_sim_count
        self.fitted = None

    def fit(self, **kwargs):
        """Fit all NMF ranks and store their goodness-of-fit values.

        Args:
            **kwargs: Keyword arguments for :func:`nmflib.nmf.fit`.
        """
        model_of_rank = {}
        for rank in self.ranks_to_test:
            sys.stderr.write(str(rank))
            model_of_rank[rank] = SingleNMFModel(self.X, rank, self.S, self.O,
                                                 self.nbinom, self.random_inits,
                                                 self.gof_sim_count)
            model_of_rank[rank].fit(print_dots=True)
        model_tuples = []
        for rank in self.ranks_to_test:
            m = model_of_rank[rank]
            model_tuples.append(
                (m, m.fitted['r'], m.fitted['gof_D'], m.fitted['gof_pval'],
                 m.fitted['aic'], m.fitted['bic'], m.fitted['n_iter'],
                 m.fitted['errors'][-1], m.fitted['elapsed']))
        columns = ('nmf_model', 'dispersion', 'gof_D', 'gof_pval',
                   'aic', 'bic', 'n_iter', 'final_error', 'elapsed')
        out_df = pd.DataFrame(model_tuples,
                              index=self.ranks_to_test,
                              columns=columns)
        self.fitted = out_df
