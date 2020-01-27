"""Nonnegative matrix factorisation code for count data."""
# Copyright (C) 2019 Yilong Li (yilong.li.yl3@gmail.com) - All Rights Reserved

import logging
import os
import multiprocessing
import sys
import time

import numpy as np
import pandas as pd
import scipy.optimize
from scipy.special import digamma, polygamma
import sklearn.metrics.pairwise

import tqdm


def _ensure_pos(arr, epsilon=np.finfo(np.float32).eps, inplace=True):
    """Ensure a (very low) epsilon floor for arr."""
    sel = arr < epsilon
    if inplace:
        arr[sel] = epsilon
    else:
        arr2 = arr.copy()
        arr2[sel] = epsilon
        return arr2


def initialise_nmf(X, k, random_state=None):
    """Initialise count NMF.

    W is initialised from a flat Dirichlet distribution and H is initialised by
    uniformly distributing the counts across all *k* signatures.

    Args:
        X (numpy.ndarray): Matrix of shape (M, N) to be decomposed.
        k (int): Rank of the intended NMF model.
        random_state (int): Optional random state.

    Returns
        numpy.ndarray: Randomised matrix W of shape (M, k), where each column
            is drawn from a Dirichlet distribution with alpha = 1.
        numpy.ndarray: Randomised matrix H of shape (k, N), where the columns
            are sampled from a flat multinomial distribution with the total
            mutation count equal to the corresponding row sums in X.
    """
    np.random.seed(random_state)
    M, N = X.shape
    W = scipy.stats.dirichlet.rvs([1] * M, size=k).T
    H = np.array([scipy.stats.multinomial.rvs(mut_count, [1 / k] * k)
                  for mut_count in np.sum(X, 0)]).T
    return W, H


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


def _ensure_df(arr):
    """Return a Pandas DataFrame version of the array."""
    if arr is None:
        return None
    elif isinstance(arr, pd.DataFrame):
        return arr
    elif isinstance(arr, np.ndarray):
        return pd.DataFrame(arr)
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


def nmf_mu(W, H, S=None, O=None):  # noqa: E741
    """Helper function for computing E[X] = (WH + O) * S."""
    WHOS = np.matmul(W, H)
    if O is not None:
        WHOS += O
    if S is not None:
        WHOS *= S
    return WHOS


def _divergence(X, X_exp, r=None):
    """Matrix divergence for Poisson or negative binomial."""

    if r is None:
        divergence = -np.sum(X * np.log(X_exp) - X_exp)
    else:
        # Since we are back to Wikipedia formulation, where p corresponds to
        # scipy.stats' 1 - p.
        one_minus_p = _nb_p(X_exp, r)
        p = 1 - one_minus_p
        _ensure_pos(p)
        _ensure_pos(one_minus_p)
        MN = np.prod(X.shape)
        divergence = -(np.sum(scipy.special.gammaln(X + r)) -
                       MN * scipy.special.gammaln(r) +
                       r * np.sum(np.log(one_minus_p)) + np.sum(X * np.log(p)))
    return divergence


def _iterate_nmf_fit(
        X,
        W,
        H,
        S=None,
        O=None,  # noqa: 471
        r=None,
        update_r=False,
        update_W=True):
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
        update_r (bool): Whether to update `r`.

    Returns:
        numpy.ndarray: The updated W.
        numpy.ndarray: The updated H.
        float: The updated r.
    """
    if update_W:
        W = _multiplicative_update_W(X, W, H, S, O, r)
    H = _multiplicative_update_H(X, W, H, S, O, r)
    mu = nmf_mu(W, H, S, O)
    if update_r:
        r, _ = _iterate_nbinom_nmf_r_ml(X, mu, r)

    return W, H, r


def _get_cpu_count(multiprocess):
    """Determine number of processes to use.

    Args:
        multiprocess (bool or int): Whether to use multiprocessing. If int, then
            then that number of parallel processes are spawned. If True, then
            the number of processes is set to :func:`os.cpu_count` - 1.

    Returns:
        int: Number of processes to use.
    """
    if (not isinstance(multiprocess, bool)
            and not (isinstance(multiprocess, int) and multiprocess >= 2)):
        raise ValueError("Parameter 'multiprocess' must either be a "
                         "boolean or >= 2.")
    if multiprocess is False:
        processes_to_use = 1
    elif multiprocess is True:
        processes_to_use = os.cpu_count() - 1
    else:
        processes_to_use = multiprocess
    return processes_to_use


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


def loglik(X, X_exp=None, r=None, W=None, H=None, S=None, O=None):  # noqa: E741
    """Compute poisson or negative binomial log-likelihood.

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        X_exp (numpy.ndarray): A nonnegative matrix for expected values of X.
        r (float): If provided, then negative binomial log-likelihood is
            computed using `r` as the dispersion parameter.
        W (numpy.ndarray): A nonnegative signatures matrix of shape (M, k).
        H (numpy.ndarray): A nonnegative exposures matrix of shape (k, N).
        S (numpy.ndarray): A nonnegative scaling matrix of shape (M, N). Ignored
            if `X_exp` is provided.
        O (numpy.ndarray): A nonnegative offset matrix of shape (M, N). Ignored
            if `X_exp` is provided.

    Note:
        Either `X_exp` or `W` and `H` must be provided.

    Returns:
        numpy.ndarray: The log-likelihood for each element in X.
    """
    if X_exp is None:
        if W is None or H is None:
            raise ValueError("Either 'X_exp' or both W and H must be provided.")
        else:
            X_exp = nmf_mu(W, H, S, O)
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


def _sim_loglik_helper_func(params):
    """A helper function for computing simulated log-likelihoods.

    Note that all parameters below must be provided as a tuple to satisfy
    multiprocessing.Pool.imap_unordered().

    Args:
        X (array-like): Observed counts: a nonnegative integer matrix of shape
            (M, N).
        sim_count (int): How many simulated instances of X should be generated?
        X_exp (array-like): A nonnegative expected values matrix of shape
            (M, N).
        r (float): A positive dispersion parameter. If None, then a Poisson
            model is assumed.
        p (float): A positive negative binomial success probability.

    Returns:
        numpy.ndarray: An array of simulated log-likelihoods of shape
            (sim_count, X.shape[1]).
    """
    X, sim_count, X_exp, r, p = params
    sim_logliks = []
    for _ in range(sim_count):
        if r is not None and p is not None:
            # Simulate from negative binomial distribution.
            sim_X = scipy.stats.nbinom.rvs(r, p, size=X.shape)
            sim_logliks.append(
                scipy.stats.nbinom.logpmf(sim_X, r, p).sum(axis=0))
        else:
            # Simulate from Poisson distribution.
            sim_X = scipy.stats.poisson.rvs(X_exp)
            sim_logliks.append(
                scipy.stats.poisson.logpmf(sim_X, X_exp).sum(axis=0))
    sim_logliks = np.array(sim_logliks)
    return sim_logliks


def gof(X, X_exp, sim_count=None, random_state=None, r=None, n_processes=1):
    """Bootstrapped goodness-of-fit for count data NMF.

    Given that X ~ Poisson(X_exp), compute the P value for each column in X
    given X_exp. Then Kolmogorov-Smirnov test to estimate goodness-of-fit.

    Args:
        X (array-like): Observed counts: a nonnegative integer matrix of shape
            (M, N).
        X_exp (array-like): A nonnegative expected values matrix of shape
            (M, N).
        sim_count (int): How many simulated instances of X should be generated
            per process? Default: 100.
        random_state (int): Random seed.
        r (float): A positive dispersion parameter. If None, then a Poisson
            model is assumed.
        n_processes (int): Number of processes to use.

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
    if not isinstance(n_processes, int) or n_processes < 1:
        raise ValueError("Parameter n_processes must be a positive " "integer.")

    # Observed log-likelihood of each sample - represented as a row vector.
    obs_loglik_mat = loglik(X, X_exp, r)
    obs_loglik_rowvec = obs_loglik_mat.sum(axis=0).reshape(1, -1)

    # Simulated log-likelihood matrices.
    if r is not None:
        p = _nb_p(X_exp, r)
    else:
        p = None
    if n_processes == 1:
        sim_logliks = _sim_loglik_helper_func((X, sim_count, X_exp, r, p))
        sim_logliks = np.array(sim_logliks)
    else:
        process_pool = multiprocessing.get_context("spawn").Pool(n_processes)
        args_list = [(X, sim_count, X_exp, r, p)] * n_processes
        sim_logliks = process_pool.imap_unordered(_sim_loglik_helper_func,
                                                  args_list)
        sim_logliks = np.concatenate(list(sim_logliks), axis=0)

    # Calculate empirical P values for row sample.
    signif_simuls = np.sum(sim_logliks <= obs_loglik_rowvec, axis=0)
    sample_pvals = signif_simuls / sim_count

    # Calculate overall goodness-of-fit P value.
    gof_D, gof_pval = scipy.stats.kstest(sample_pvals, 'uniform')
    return gof_D, gof_pval, sample_pvals, sim_logliks


class MaxIterError(Exception):
    """Maximum iteration reached without convergence in NMF fit."""
    pass


def fit(
        X,
        k=None,
        S=None,
        O=None,  # noqa: E741
        nbinom_fit=False,
        max_iter=1000,
        abstol=1e-5,
        epoch_len=10,
        random_state=None,
        W_fixed=None,
        H_init=None,
        r=None,
        verbose=False,
        max_iter_error=False):
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
        abstol (float): Absolute tolerance for convergence in the log-space of
            likelihood per iteration. Fitting is considered converged when error
            improves by less than `abstol` x `epoch_len` over an `epoch_len`.
        epoch_len (int): Compute divergence, check convergence and optionally
            update negative binomial dispersion every this many iterations.
        random_state (int): The random seed to use in NMF initialisation.
        W_fixed (numpy.ndarray): A matrix of shape (M, k). If provided, then
            W is not updated but this matrix is used as a constant W instead.
        H_init (numpy.ndarray): The initial H matrix values.
        r (float): If `nbinom_fit` is `False`, then this value is used as the
            constant overdispersion parameter for negative binomial NMF. If
            `nbinom_fit` is True, this parameter is ignored.
        verbose (bool): Whether to print progress updates every `epoch_len`
            iterations.
        max_iter_error (bool): If True, an error as opposed to a warning is
            raised if maximum iteration limit is reached.

    Returns:
        numpy.ndarray: The fitted W matrix of shape (M, k). The matrix is scaled
            to column sums of 1.
        numpy.ndarray: The fitted H matrix of shape (k, N). The matrix is scaled
            correspondingly to W matrix scaling.
        float: The overdispersion parameter r. Returns None if the Poisson model
            was fitted.
        int: Number of iterations used.
        list: A list of errors recorded every `epoch_len` iterations.

    Raises:
        MaxIterError: If `max_iter_error` is True and the maximum iteration
            limit is reached.
    """
    if k is None and W_fixed is None:
        raise ValueError("'r' must be provided if 'W_fixed' is not.")
    elif k is None:
        k = W_fixed.shape[1]

    # Make sure the matrices are numpy arrays.
    X = _validate_is_ndarray(X)
    S = _validate_is_ndarray(S)
    O = _validate_is_ndarray(O)  # noqa: E741

    # Initialise W and H.
    W, H = initialise_nmf(X, k, random_state=random_state)
    if W_fixed is not None:
        update_W = False
        W = W_fixed
    else:
        update_W = True
    if H_init is not None:
        H = H_init
    if nbinom_fit:
        X_exp = nmf_mu(W, H, S, O)
        r = _initialise_nb_r(X, X_exp)

    # Set up initial values.
    start_time = time.time()
    previous_error = None
    errors = []
    n_iter = 0

    while True:
        # Test for convergence and optionally update r every epoch_len
        # iterations.
        X_exp = nmf_mu(W, H, S, O)

        if nbinom_fit:
            r, _ = fit_nbinom_nmf_r_ml(X, X_exp, r)
            if verbose:
                logging.info(f'Iteration {n_iter}, updated r to {r}.')

        # Perform `epoch_len` updates (without updating r).
        for _ in range(epoch_len):
            n_iter += 1
            W, H, r = _iterate_nmf_fit(X, W, H, S, O, r, update_W=update_W)

        # Check for convergence.
        error = _divergence(X, X_exp, r)
        errors.append(error)
        if verbose:
            elapsed = time.time() - start_time
            logging.info(
                "Iteration {} after {:.3f} seconds, error: {}".format(
                    n_iter, elapsed, error))
        if previous_error is None:
            previous_error = error
        elif previous_error - error < abstol * epoch_len:
            break
        elif n_iter >= max_iter:
            break
        previous_error = error

    if n_iter >= max_iter:
        msg = "Maximum iteration reached."
        if max_iter_error:
            raise MaxIterError(msg)
        else:
            logging.warning(msg)

    # Scale W and H such that W columns sum to 1.
    if update_W:
        W_colsums = np.sum(W, axis=0)
        W /= W_colsums[np.newaxis, :]
        H *= W_colsums[:, np.newaxis]
    return W, H, r, n_iter, errors


def _fit_worker_wrapper(args_kwargs_tuple):
    """Run fit(), except unpack arguments first."""
    args, kwargs = args_kwargs_tuple
    return fit(*args, **kwargs)


def mpfit(
        random_inits,
        n_processes=None,
        random_state=None,
        verbose=False,
        *args,
        **kwargs):
    """Parallelised NMF fitting.

    Args:
        random_inits (int): The number of random initialisations to fit using
            NMF.
        n_processes (int): Number of processes to spawn. By default,
            :func:`os.cpu_count` - 1  is used.
        random_state (int): If provided, then the random states used are for
            each random initialisation are `random_state` +
            np.array(range(random_inits)).
        verbose (bool): Whether to show a progress bar.
        args (list): Positional arguments for :func:`fit`.
        kwargs (dict): Keyword arguments for :func:`fit`.

    Returns:
        list: A list of outputs from :func:`fit`.

    See also:
        Additional argument documentation is found in function :func:`fit`.
    """
    if n_processes is None:
        n_processes = os.cpu_count() - 1

    # Create argument list for fit(). The last three arguments are 'verbose',
    # and 'random_state'.
    if random_state is None:
        args_list = [(args, kwargs)] * random_inits
    else:
        args_list = []
        for i in range(random_inits):
            cur_kwargs = kwargs.copy()
            cur_kwargs['random_state'] = random_state + i
            args_list.append((args, cur_kwargs))
    process_pool = multiprocessing.get_context("spawn").Pool(n_processes)
    if verbose:
        fitted_res = list(
            tqdm.tqdm(process_pool.imap_unordered(_fit_worker_wrapper,
                                                  args_list),
                      total=random_inits))
    else:
        fitted_res = list(
            process_pool.imap_unordered(_fit_worker_wrapper, args_list))
    return fitted_res


def hk_lrt(
        x_obs,
        W,
        sig_idx,
        S=None,
        O=None,  # noqa: E741
        r=None,
        h_hat=None,
        fit_kwargs=None):
    """Compute LRT for whether exposure of signature `sig_idx` is zero.

    Args:
        x_obs (numpy.ndarray): 1D array of observed mutation counts across M
            mutation types.
        W (numpy.ndarray): A signatures matrix of shape (M, k).
        sig_idx (int): Which signature with the statistics be computed on?
        r (float): If provided, negative binomial model will be used.
        h_hat (numpy.ndarray): 1D array of previously fitted signature exposures
            across k signatures.
        fit_kwargs (dict): Additional keyword arguments for :func:`fit`.

    Returns:
        float: Likelihood ratio test P value.
        float: (Asymptotic) Chi-squared test statistic for the likelihood ratio.
        float: Log-likelihood of the (provided or fitted) ML exposures vector.
        float: Profile log-likelihood for exposure of signature `sig_idx` being
            zero.
        numpy.ndarray: The restricted maximum-likelihood estimate of `h`.
    """
    # Initialise some arguments.
    if fit_kwargs is None:
        fit_kwargs = {}
    M, k = W.shape

    # Convert variables into appropriate column vectors.
    x_obs = x_obs.reshape(-1, 1)
    if h_hat is not None:
        h_hat = h_hat.reshape(-1, 1)
    else:
        _, h_hat, _, _, _ = fit(x_obs,
                                None,
                                S,
                                O,
                                W_fixed=W,
                                r=r,
                                **fit_kwargs)

    # Overall ML log-likelihood.
    ml_loglik = np.sum(loglik(x_obs, None, r, W, h_hat, S, O))

    # Calculate log-likelihood and LRT for the exposure of signature k being 0.
    W_fixed = W[:, np.arange(k) != sig_idx]
    h = h_hat[np.arange(k) != sig_idx]
    _, profile_h_hat, _, _, _ = fit(x_obs,
                                    None,
                                    S,
                                    O,
                                    W_fixed=W_fixed,
                                    H_init=h,
                                    r=r,
                                    **fit_kwargs)
    restricted_h_hat = h_hat.copy()
    restricted_h_hat[sig_idx] = 0
    restricted_h_hat[np.arange(k) != sig_idx] = profile_h_hat
    h0_loglik = np.sum(loglik(x_obs, None, r, W, restricted_h_hat, S, O))
    h0_chi2_stat = max(2 * (ml_loglik - h0_loglik), 0.0)
    h0_chi2_pval = 1 - scipy.stats.chi2.cdf(h0_chi2_stat, 1)

    return (h0_chi2_pval, h0_chi2_stat, ml_loglik, h0_loglik,
            restricted_h_hat.reshape(-1))


class _NMFProfileLoglikFitter:
    """
    Helper class that provides a h_i -> profile log-likelihood function.

    Args:
        x_obs (numpy.ndarray): A column matrix of shape (M, 1).
        W (numpy.ndarray): A matrix of signatures of shape (M, k).
        h_hat (numpy.ndarray): A column matrix of maximum likelihood estimates
            for exposures. Shape is (k, 1).
        sig_idx (numpy.ndarray): Index in h_hat or W columns for which profile
            likelihood is to be computed.
        S (numpy.ndarray): A column matrix of scales of shape (M, 1).
        O (numpy.ndarray): A column matrix of offsets of shape (M, 1).
        r (float): Fixed dispersion parameter for negative binomial NMF
            optimisation. If not provided, then Poisson-NMF is used instead.
        fit_kwargs (dict): Additional keyword arguments for :func:`fit`.
    """
    def __init__(
            self,
            x_obs,
            W,
            h_hat,
            sig_idx,
            S=None,
            O=None,  # noqa: E741
            r=None,
            **fit_kwargs):
        self.x = x_obs
        sel = np.arange(W.shape[1]) != sig_idx
        self.W_nuisance = W[:, sel]
        self.Wi = W[:, [sig_idx]]
        self.h_nuisance = h_hat[sel, :]
        self.hi = h_hat[sig_idx, 0]
        self.S = S
        self.O = O  # noqa: E741
        self.r = r
        self.fit_kwargs = fit_kwargs

    def _profile_ml(self, cur_hi):
        """Compute the profile maximum log-likelihood for cur_h_idx.

        Args:
            cur_hi (float): Current value for h_i, which is the exposure for
                signature W_i.

        Returns:
            float: Maximum log-likelihood with the restriction that
                h[self.sig_idx] == cur_hi.
        """
        O_plus_Whi = self.Wi * cur_hi
        if self.O is not None:
            O_plus_Whi += self.O
        _, h_nuisance_hat, _, _, _ = fit(self.x,
                                         None,
                                         self.S,
                                         O_plus_Whi,
                                         W_fixed=self.W_nuisance,
                                         H_init=self.h_nuisance,
                                         r=self.r,
                                         **self.fit_kwargs)
        profile_ml = np.sum(
            loglik(self.x, None, self.r, self.W_nuisance, h_nuisance_hat,
                   self.S, O_plus_Whi))
        return profile_ml


def hk_confint(
        x_obs,
        W,
        sig_idx,
        S=None,
        O=None,  # noqa: E741
        alpha=0.05,
        r=None,
        h_hat=None,
        brentq_xtol=1e-5,
        brentq_rtol=1e-5,
        fit_kwargs=None):
    """Compute LRT and confidence intervals for a signature's exposure.

    Compute the likelihood-ratio test for the null hypothesis that signature `k`
    has zero exposure. Then compute the (1 - `alpha`)-confidence interval for
    the estimated exposure value.

    Args:
        x_obs (numpy.ndarray): 1D array of observed mutation counts across M
            mutation types.
        W (numpy.ndarray): A signatures matrix of shape (M, k).
        sig_idx (int): Which signature with the statistics be computed on?
        S (numpy.ndarray): 1D array of scaling factors.
        O (numpy.ndarray): 1D array of offsets.
        alpha (float): The cumulative probability that should lie outside the
            confidence intervals (in total for both sides).
        r (float): If provided, negative binomial model will be used.
        h_hat (numpy.ndarray): 1D array of previously fitted signature exposures
            across k signatures.
        brentq_xtol (float): Parameter `xtol` for :func:`scipy.optimize.brentq`.
        brentq_rtol (float): Parameter `rtol` for :func:`scipy.optimize.brentq`.
        fit_kwargs (dict): Additional keyword arguments for :func:`fit`.
            By default, `max_iter` is infinite, `abstol` is 0.01 and `epoch_len`
            is 5.

    Returns:
        float: Estimated or provided exposure of signature `sig_idx`.
        float: Lower end of the confidence interval.
        float: Upper end of the confidence interval.
        float: Likelihood ratio test P value.
        float: (Asymptotic) Chi-squared test statistic for the likelihood ratio.
        float: Log-likelihood of the (provided or fitted) ML exposures vector.
        float: Profile log-likelihood for exposure of signature `sig_idx` being
            zero.
        float: :cls:`scipy.stats.RootResults` of the lower bound root finding.
        float: :cls:`scipy.stats.RootResults` of the upper bound root finding.
    """
    # Convert variables into appropriate column vectors.
    x_obs = x_obs.reshape(-1, 1)
    if S is not None:
        S = S.reshape(-1, 1)
    if O is not None:
        O = O.reshape(-1, 1)  # noqa: E741
    if fit_kwargs is None:
        fit_kwargs = {}
    if 'max_iter' not in fit_kwargs:
        fit_kwargs['max_iter'] = float('inf')
    if 'abstol' not in fit_kwargs:
        fit_kwargs['abstol'] = 0.01
    if h_hat is not None:
        h_hat = h_hat.reshape(-1, 1)
    else:
        _, h_hat, _, _, _ = fit(x_obs,
                                None,
                                S,
                                O,
                                W_fixed=W,
                                r=r,
                                **fit_kwargs)

    # Compute LRT for signature sig_idx having zero exposure.
    h0_chi2_pval, h0_chi2_stat, ml_loglik, h0_loglik, _ = hk_lrt(
        x_obs, W, sig_idx, S, O, r, h_hat, fit_kwargs)

    # Calculate the minimum null hypothesis log-likelihood to be considered to
    # be within the alpha-confidence interval.
    chisq_cutoff = scipy.stats.chi2.ppf(1 - alpha, 1)
    h0_loglik_cutoff = ml_loglik - chisq_cutoff / 2

    # Compute the lower end of the confidence interval. First check if zero
    # is included in the confidence interval.
    if h0_loglik >= h0_loglik_cutoff:
        confint_lower_bound = 0
        r_lower_bound = None
    else:
        target_f = _NMFProfileLoglikFitter(x_obs, W, h_hat, sig_idx, S, O, r,
                                           **fit_kwargs)._profile_ml
        confint_lower_bound, r_lower_bound = scipy.optimize.brentq(
            lambda h_i: target_f(h_i) - h0_loglik_cutoff,
            h_hat[sig_idx, 0],
            0,
            xtol=brentq_xtol,
            rtol=brentq_rtol,
            full_output=True,
            disp=True)

    # Compute the upper end of the confidence interval.
    target_f = _NMFProfileLoglikFitter(x_obs, W, h_hat, sig_idx, S, O, r,
                                       **fit_kwargs)._profile_ml
    upper_bound_upper_bound = np.sum(h_hat)
    while target_f(upper_bound_upper_bound) - h0_loglik_cutoff > 0:
        upper_bound_upper_bound *= 2
    confint_upper_bound, r_upper_bound = scipy.optimize.brentq(
        lambda h_i: target_f(h_i) - h0_loglik_cutoff,
        h_hat[sig_idx, 0],
        upper_bound_upper_bound,
        xtol=brentq_xtol,
        rtol=brentq_rtol,
        full_output=True,
        disp=True)

    return (h_hat[sig_idx, 0], confint_lower_bound, confint_upper_bound,
            h0_chi2_pval, h0_chi2_stat, ml_loglik, h0_loglik, r_lower_bound,
            r_upper_bound)


def h_confint(
        x_obs,
        W,
        S=None,
        O=None,  # noqa: E741
        alpha=0.05,
        r=None,
        h_hat=None,
        brentq_xtol=1e-5,
        brentq_rtol=1e-5,
        fit_kwargs=None):
    """Compute LRT and confidence intervals for each signature's exposure.

    Compute the likelihood-ratio test for the null hypothesis for each signature
    that the signature's exposure is zero. Then compute the (1 - `alpha`)-
    confidence interval for the estimated exposure value.

    Args:
        x_obs (numpy.ndarray): 1D array of observed mutation counts across M
            mutation types.
        W (numpy.ndarray): A signatures matrix of shape (M, k).
        S (numpy.ndarray): 1D array of scaling factors.
        O (numpy.ndarray): 1D array of offsets.
        alpha (float): The cumulative probability that should lie outside the
            confidence intervals (in total for both sides).
        r (float): If provided, negative binomial model will be used.
        h_hat (numpy.ndarray): 1D array of previously fitted signature exposures
            across k signatures.
        brentq_xtol (float): Parameter `xtol` for :func:`scipy.optimize.brentq`.
        brentq_rtol (float): Parameter `rtol` for :func:`scipy.optimize.brentq`.
        fit_kwargs (dict): Additional keyword arguments for :func:`fit`. See
            also :func:`hk_confint`.

    Returns:
        pandas.DataFrame: A data frame of confidence intervals and likelihood
            ratio results.
    """
    if fit_kwargs is None:
        fit_kwargs = dict()
    if S is None:
        S_colvec = None
    else:
        S_colvec = S.reshape(-1, 1)
    if O is None:
        O_colvec = None
    else:
        O_colvec = O.reshape(-1, 1)
    if h_hat is None:
        _, h_hat, _, _, _ = fit(x_obs.reshape(-1, 1),
                                None,
                                S_colvec,
                                O_colvec,
                                W_fixed=W,
                                r=r,
                                **fit_kwargs)
        h_hat = h_hat.reshape(-1)  # hk_confint() requires an 1D array.
    output_tuples = []
    for sig_idx in range(len(h_hat)):
        output_tuple = hk_confint(x_obs, W, sig_idx, S, O, alpha, r, h_hat,
                                  brentq_xtol, brentq_rtol, fit_kwargs)
        output_tuples.append(output_tuple)
    columns = ('h', 'cint_low', 'cint_high', 'pval', 'chi2_stat', 'ml_loglik',
               'h0_loglik', 'r_lower_bound', 'r_upper_bound')
    out_df = pd.DataFrame(output_tuples, columns=columns)
    return out_df


def match_components(W1, W2):
    """Match NMF components (columns) between W1 and W2.

    The distance metric used in cosine similarity. Matching is done using
    Hungarian algorithm.

    Args:
        W1 (array-like): An array of shape (<mutation contexts>, <component>).
        W2 (array-like): An array of shape (<mutation contexts>, <component>).

    Returns:
        numpy.ndarray: An array of indices matching columns of W2 with columns
            of W1.
    """
    dist_mat = sklearn.metrics.pairwise.cosine_distances(W1.T, W2.T)
    _, W2_idx = scipy.optimize.linear_sum_assignment(dist_mat)
    return W2_idx


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

    def fit(self, verbose=False, multiprocess=False, **kwargs):
        """Fit the current NMF model and compute goodness-of-fit.

        The results are stored in :attr:`fitted`.

        Args:
            verbose (bool): Whether to show a progress bar for progress in
                random initialisation fits.
            multiprocess (bool or int): Whether to use multiprocessing. If int,
                then that number of parallel processes are spawned. If True,
                then the number of processes is set to :func:`os.cpu_count` - 1.
            **kwargs: Keyword arguments for :func:`nmflib.nmf.fit`.
        """
        processes_to_spawn = _get_cpu_count(multiprocess)
        start_time = time.time()

        # Compute the best decomposition.
        if processes_to_spawn == 1:
            errors_best = None
            if verbose == 1:
                random_init_range = tqdm.tqdm(range(self.random_inits))
            else:
                random_init_range = range(self.random_inits)
            fit_verbose = verbose > 1
            for _ in random_init_range:
                W, H, r, n_iter, errors = fit(self.X, self.rank, self.S, self.O,
                                              self.nbinom, verbose=fit_verbose,
                                              **kwargs)
                if errors_best is None or errors[-1] < errors_best[-1]:
                    W_best, H_best, r_best, n_iter_best = W, H, r, n_iter
                    errors_best = errors
        else:
            fitted_models = mpfit(self.random_inits,
                                  self.X,
                                  self.rank,
                                  self.S,
                                  self.O,
                                  self.nbinom,
                                  n_processes=processes_to_spawn,
                                  verbose=verbose,
                                  **kwargs)
            best_model_idx = np.argmax([m[4][-1] for m in fitted_models])
            W_best, H_best, r_best, n_iter_best, errors_best = \
                fitted_models[best_model_idx]

        elapsed = time.time() - start_time

        X_pred = nmf_mu(W_best, H_best, self.S, self.O)
        fitted_loglik = np.sum(loglik(self.X, X_pred, r_best))

        # Save results.
        self.fitted = {
            'W': W_best,
            'H': H_best,
            'X_pred': X_pred,
            'r': r_best,
            'n_iter': n_iter_best,
            'errors': errors_best,
            'loglik': fitted_loglik,
            'elapsed': elapsed
        }

    def gof(self, multiprocess=False):
        """Compute simulated goodness-of-fit.

        Args:
            multiprocess (bool or int): Whether to use multiprocessing. If int,
                then that number of parallel processes are spawned. If True,
                then the number of processes is set to :func:`os.cpu_count` - 1.

        Returns:
            float: Goodness-of-fit KS statistic.
            float: Goodness-of-fit KS P value.
            numpy.ndarray: Goodness-of-fit p-values for each sample (i.e.
                column of X) individually.
            numpy.ndarray: A matrix of log-likelihoods for simulated
                observations for each sample with shape (`sim_count`,
                X.shape[1]).
        """
        if self.fitted is None:
            raise ValueError(
                'self.fitted is None. Please run self.fit() first.')
        processes_to_spawn = _get_cpu_count(multiprocess)
        gof_D, gof_pval, sample_gof_pval, sim_logliks = gof(
            self.X,
            self.fitted['X_pred'],
            self.gof_sim_count,
            r=self.fitted['r'],
            n_processes=processes_to_spawn)
        return gof_D, gof_pval, sample_gof_pval, sim_logliks

    def aic(self):
        """Calculate AIC."""
        if self.fitted is None:
            raise ValueError(
                'self.fitted is None. Please run self.fit() first.')
        param_count = self._n_params()
        aic = calc_aic(self.fitted['loglik'], param_count)
        return aic

    def bic(self):
        """Calculate BIC."""
        if self.fitted is None:
            raise ValueError(
                'self.fitted is None. Please run self.fit() first.')
        param_count = self._n_params()
        sample_count = np.prod(self.X.shape)
        bic = calc_bic(self.fitted['loglik'], param_count, sample_count)
        return bic

    def predict(self, X_new=None, S_new=None, O_new=None, alpha=0.05,
                verbose=False, **kwargs):
        """Fit H MLEs and alpha-confidence intervals for a new count matrix.

        Args:
            alpha (float): The total probability that should lie outside the
                confidence intervals.
            X_new (array-like): A 2D array of shape `M`, `N_new` of nonnegative
                mutation counts. If None, then :attr:`X` is use instead.
            S_new (array-like): Scaling matrix for `X_new`.
            O_new (array-like): Offset matrix for `X_new`.
            verbose (bool): Whether to print a progress bar.
            kwargs (dict): Additional keyword arguments for :func:`h_confint`.

        Returns:
            pd.DataFrame: A "long format" data frame of confidence intervals and
                likelihood ratio results for each sample and signature.
        """
        if self.fitted is None:
            raise ValueError(
                'self.fitted is None. Please run self.fit() first.')
        if X_new is None:
            X_new = self.X
            S_new = self.S
            O_new = self.O
        else:
            X_new = _ensure_df(X_new)
            S_new = _ensure_df(S_new)
            O_new = _ensure_df(O_new)
        confint_dfs = []
        if verbose:
            sample_list = tqdm.tqdm(X_new.columns)
        else:
            sample_list = X_new.columns
        for sample in sample_list:
            if S_new is None:
                cur_S = None
            else:
                cur_S = S_new[sample].values
            if O_new is None:
                cur_O = None
            else:
                cur_O = O_new[sample].values
            confint_df = h_confint(X_new[sample].values, self.fitted['W'],
                                   cur_S, cur_O, alpha, self.fitted['r'])
            confint_df.index = pd.MultiIndex.from_frame(
                pd.DataFrame({'signature': range(self.rank), 'sample': sample}))
            confint_dfs.append(confint_df)
        full_df = pd.concat(confint_dfs)
        return full_df

    def _nmf_fitter_helper_func(self, print_dots=False, **kwargs):
        """Fit an NMF model from a random initialisation.

        Args:
            kwargs (dict): Additional parameters for :func:`fit` apart from
                `X`, `k`, `S`, `O` and `nbinom_fit`.
        """
        def fitter_helper_func(_=None):
            if print_dots:
                sys.stderr.write('.')
            W, H, r, n_iter, errors = fit(self.X, self.rank, self.S, self.O,
                                          self.nbinom, **kwargs)
            return W, H, r, n_iter, errors

        return fitter_helper_func

    def _n_params(self):
        """Number of model parameters."""
        M, N = self.X.shape
        param_count = (M - 1 + N) * self.rank
        if self.nbinom:
            param_count += 1
        return param_count

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

    def fit(self, verbose=False, multiprocess=False, **kwargs):
        """Fit all NMF ranks and store their goodness-of-fit values.

        Args:
            verbose (bool): Whether to show a progress bar for progress in
                random initialisation fits.
            multiprocess (bool or int): Whether to use multiprocessing. See
                :meth:`SingleNMFModel.fit`.
            **kwargs: Keyword arguments for :func:`nmflib.nmf.fit`.
        """
        model_of_rank = {}
        for rank in self.ranks_to_test:
            logging.info(f'Fitting rank {rank}')
            model_of_rank[rank] = SingleNMFModel(self.X, rank, self.S, self.O,
                                                 self.nbinom, self.random_inits,
                                                 self.gof_sim_count)
            model_of_rank[rank].fit(verbose, multiprocess, **kwargs)
        model_tuples = []
        for rank in self.ranks_to_test:
            m = model_of_rank[rank]
            tuple_to_append = [m, m.fitted['r'], m.fitted['loglik']]
            tuple_to_append += m.gof(multiprocess)[:2]
            tuple_to_append += (m.aic(), m.bic())
            tuple_to_append += [m.fitted['n_iter'], m.fitted['errors'][-1],
                                m.fitted['elapsed']]
            model_tuples.append(tuple_to_append)
        columns = ('nmf_model', 'dispersion', 'log-likelihood', 'gof_D',
                   'gof_pval', 'aic', 'bic', 'n_iter', 'final_error', 'elapsed')
        out_df = pd.DataFrame(model_tuples,
                              index=self.ranks_to_test,
                              columns=columns)
        self.fitted = out_df
