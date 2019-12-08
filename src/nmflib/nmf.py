"""Nonnegative matrix factorisation code for count data."""
# Copyright (C) 2019 Yilong Li (yilong.li.yl3@gmail.com) - All Rights Reserved


import logging
import numpy as np
import scipy.optimize
import sklearn.decomposition._nmf
import sklearn.metrics.pairwise
import time


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


def fit_nmf(X, k, S=None, max_iter=200, tol=1e-4, verbose=False,
            random_state=None):
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
                logging.info(
                    "Iteration %02d after %.3f seconds, error: %f".format(
                        n_iter, elapsed, error))
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    if n_iter % 10 != 0:
        error = _kl_divergence(X, W, H, S)
        errors.append(error)
        if verbose:
            elapsed = time.time() - start_time
            msg = ("Stopped after iteration %02d after %.3f seconds, error: %f"
                   .format(n_iter, elapsed, error))
            logging.info(msg)

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
