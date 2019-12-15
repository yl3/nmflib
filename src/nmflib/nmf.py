"""Nonnegative matrix factorisation code for count data."""
# Copyright (C) 2019 Yilong Li (yilong.li.yl3@gmail.com) - All Rights Reserved


import numpy as np
import pandas as pd
import scipy.optimize
import sklearn.decomposition._nmf
import sklearn.metrics.pairwise

import logging
import time

import nmflib.constants


EPSILON = np.finfo(np.float32).eps


def _mu_W(X, W, H, S=None):
    """Multiplicative KL-update for W.

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        W (numpy.ndarray): A nonnegative matrix of shape (M, K) for current W.
        H (numpy.ndarray): A nonnegative matrix of shape (K, N) for current H.
        S (numpy.ndarray): A matrix of values [0, 1] of shape (M, N) indicating
            the proportion of mutations of each of the M contexts that cannot be
            observed.

    Returns:
        numpy.ndarray: The multiplicative update for W.
    """

    # Compute numerator.
    numerator = np.dot(W, H)
    numerator[numerator == 0] = EPSILON
    np.divide(X, numerator, out=numerator)
    numerator = np.dot(numerator, H.T)

    # Compute denominator
    if S is None:
        H_sum = np.sum(H, axis=1)
        denominator = H_sum[np.newaxis, :]
    else:
        denominator = np.dot(S, H.T)
    denominator[denominator == 0] = EPSILON

    # Compute the update
    numerator /= denominator
    delta_W = numerator

    return delta_W


def _mu_H(X, W, H, S=None):
    """Multiplicative KL-update for H.

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        W (numpy.ndarray): A nonnegative matrix of shape (M, K) for current W.
        H (numpy.ndarray): A nonnegative matrix of shape (K, N) for current H.
        S (numpy.ndarray): A matrix of values [0, 1] of shape (M, N) indicating
            the proportion of mutations of each of the M contexts that cannot be
            observed.

    Returns:
        numpy.ndarray: The multiplicative update for H.
    """
    # Compute numerator.
    numerator = np.dot(W, H)
    numerator[numerator == 0] = EPSILON
    np.divide(X, numerator, out=numerator)
    numerator = np.dot(W.T, numerator)

    # Compute denominator
    if S is None:
        W_sum = np.sum(W, axis=0)
        denominator = W_sum[:, np.newaxis]
    else:
        denominator = np.dot(W.T, S)
    denominator[denominator == 0] = EPSILON

    # Compute the update
    numerator /= denominator
    delta_H = numerator

    return delta_H


def _kl_divergence(X, W, H, S=None):
    """Kullback-Leibler divergence for X ~ WH.

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        W (numpy.ndarray): A nonnegative matrix of shape (M, K) for current W.
        H (numpy.ndarray): A nonnegative matrix of shape (K, N) for current H.
        S (numpy.ndarray): A matrix of values [0, 1] of shape (M, N) indicating
            the proportion of mutations of each of the M contexts that cannot be
            observed.

    Returns:
        float: Total KL divergence.
    """
    X_exp = np.dot(W, H)
    if S is not None:
        X_exp *= S
    X_exp_raveled = X_exp.ravel()
    X_raveled = X.ravel()

    # Prevent the algorithm from crashing.
    X_not_zero = X_raveled != 0
    X_exp_raveled = X_exp_raveled[X_not_zero]
    X_raveled = X_raveled[X_not_zero]
    X_exp_raveled[X_exp_raveled == 0] = EPSILON

    if S is None:
        sum_E_exp = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
    else:
        sum_E_exp = np.sum(X_exp)
    div = X_raveled / X_exp_raveled
    res = np.dot(X_raveled, np.log(div))
    res += sum_E_exp - np.sum(X_raveled)

    return res


def gof(X, X_exp, sim_count=None, random_state=None):
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

    Returns:
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
    obs_loglik_rowvec = (scipy.stats.poisson.logpmf(X, X_exp)
                         .sum(axis=0)
                         .reshape(1, -1))

    # Simulated log-likelihood matrices.
    sim_logliks = []
    for k in range(sim_count):
        sim_X = scipy.stats.poisson.rvs(X_exp)
        sim_logliks.append(scipy.stats.poisson.logpmf(sim_X, X_exp).sum(axis=0))
    sim_logliks = np.array(sim_logliks)

    # Calculate empirical P values for row sample.
    signif_simuls = np.sum(sim_logliks <= obs_loglik_rowvec, axis=0)
    sample_pvals = signif_simuls / sim_count

    # Calculate overall goodness-of-fit P value.
    gof_D, gof_pval = scipy.stats.kstest(sample_pvals, 'uniform')
    return gof_pval, sample_pvals, sim_logliks


def fit(X, k, S=None, max_iter=200, tol=1e-4, verbose=False, random_state=None):
    """Fit KL-NMF using multiplicative updates.

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        k (int): A positive integer for the number of signatures to use.
        S (numpy.ndarray): A matrix of values [0, 1] of shape (M, N) indicating
            the proportion of mutations of each of the M contexts that cannot be
            observed.
        max_iter (int): Maximum number of iterations to use.
        tol (float): Relative tolerance for convergence.
        verbose (bool): Whether to print progress updates every 10 iterations.
        random_state (int): The random seed to use in NMF initialisation.

    Returns:
        numpy.ndarray: The fitted W matrix of shape (M, k). The matrix is scaled
            to column sums of 1.
        numpy.ndarray: The fitted H matrix of shape (k, N). The matrix is scaled
            correspondingly to W matrix scaling.
        int: Number of iterations used.
        list: A list of errors recorded every 10 iterations.
    """
    W, H = sklearn.decomposition._nmf._initialize_nmf(
        X, k, 'random', random_state=random_state)

    # Set up initial values.
    start_time = time.time()
    error_at_init = _kl_divergence(X, W, H)
    previous_error = error_at_init
    errors = []

    for n_iter in range(1, max_iter + 1):
        delta_W = _mu_W(X, W, H, S)
        W *= delta_W
        delta_H = _mu_H(X, W, H, S)
        H *= delta_H

        # Test for convergence every 10 iterations.
        if n_iter % 10 == 0:
            error = _kl_divergence(X, W, H, S)
            errors.append(error)
            if verbose:
                elapsed = time.time() - start_time
                logging.info("Iteration {} after {:.3f} seconds, error: {}"
                             .format(n_iter, elapsed, error))
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # Scale W and H such that W columns sum to 1.
    W_colsums = np.sum(W, axis=0)
    W /= W_colsums[np.newaxis, :]
    H *= W_colsums[:, np.newaxis]
    return W, H, n_iter, errors


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
        ew_rates = pd.Series(nmflib.constants.HUMAN_EXOME_TRINUCS
                             / nmflib.constants.HUMAN_GENOME_TRINUCS)
    else:
        gw_rates = nmflib.constants.HUMAN_GENOME_TRINUCS
        ew_rates = nmflib.constants.HUMAN_EXOME_TRINUCS
    mut_context_types = mut_context_types.merge(gw_rates.to_frame(), left_on=1,
                                                right_index=True)
    mut_context_types = mut_context_types.merge(ew_rates.to_frame(), left_on=1,
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
    counts_to_use = [context_counts['ew_rate']
                     if x else context_counts['gw_rate']
                     for x in sample_is_exome]
    scale_mat = pd.DataFrame(counts_to_use).T
    if isinstance(sample_is_exome, pd.Series):
        scale_mat.columns = sample_is_exome.index
    else:
        scale_mat = scale_mat.T.reset_index(drop=True).T
    return scale_mat
