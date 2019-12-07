"""Nonnegative matrix factorisation code for count data."""
# Copyright (C) 2019 Yilong Li (yilong.li.yl3@gmail.com) - All Rights Reserved


import numba
import logging
import numpy as np
import sklearn.decomposition._nmf
import time


EPSILON = np.finfo(np.float32).eps


def _mu_W(X, W, H):
    """Multiplicative KL-update for W.

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        W (numpy.ndarray): A nonnegative matrix of shape (M, K) for current W.
        H (numpy.ndarray): A nonnegative matrix of shape (K, N) for current H.

    Returns:
        numpy.ndarray: The multiplicative update for W.
    """

    # Compute numerator.
    numerator = np.dot(W, H)
    numerator[numerator == 0] = EPSILON
    np.divide(X, numerator, out=numerator)
    numerator = np.dot(numerator, H.T)

    # Compute denominator
    H_sum = np.sum(H, axis=1)
    denominator = H_sum[np.newaxis, :]
    denominator[denominator == 0] = EPSILON

    # Compute the update
    numerator /= denominator
    delta_W = numerator

    return delta_W


def _mu_H(X, W, H):
    """Multiplicative KL-update for H.

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        W (numpy.ndarray): A nonnegative matrix of shape (M, K) for current W.
        H (numpy.ndarray): A nonnegative matrix of shape (K, N) for current H.

    Returns:
        numpy.ndarray: The multiplicative update for H.
    """
    # Compute numerator.
    numerator = np.dot(W, H)
    numerator[numerator == 0] = EPSILON
    np.divide(X, numerator, out=numerator)
    numerator = np.dot(W.T, numerator)

    # Compute denominator
    W_sum = np.sum(W, axis=0)
    denominator = W_sum[:, np.newaxis]
    denominator[denominator == 0] = EPSILON

    # Compute the update
    numerator /= denominator
    delta_H = numerator

    return delta_H


@numba.njit
def _mu_W_jit(X, W, H):
    """Multiplicative KL-update for W.

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        W (numpy.ndarray): A nonnegative matrix of shape (M, K) for current W.
        H (numpy.ndarray): A nonnegative matrix of shape (K, N) for current H.

    Returns:
        numpy.ndarray: The multiplicative update for W.
    """

    # Compute numerator.
    numerator = np.dot(W, H)
    numerator = np.where(numerator == 0, EPSILON, numerator)
    numerator = np.divide(X, numerator)
    numerator = np.dot(numerator, H.T)

    # Compute denominator
    H_sum = np.sum(H, axis=1)
    denominator = H_sum.reshape(1, -1)
    denominator = np.where(denominator == 0, EPSILON, denominator)

    # Compute the update
    numerator /= denominator
    delta_W = numerator

    return delta_W


@numba.njit
def _mu_H_jit(X, W, H):
    """Multiplicative KL-update for H.

    Args:
        X (numpy.ndarray): A nonnegative integer matrix of shape (M, N).
        W (numpy.ndarray): A nonnegative matrix of shape (M, K) for current W.
        H (numpy.ndarray): A nonnegative matrix of shape (K, N) for current H.

    Returns:
        numpy.ndarray: The multiplicative update for H.
    """
    # Compute numerator.
    numerator = np.dot(W, H)
    numerator = np.where(numerator == 0, EPSILON, numerator)
    numerator = np.divide(X, numerator)
    numerator = np.dot(W.T, numerator)

    # Compute denominator
    W_sum = np.sum(W, axis=0)
    denominator = W_sum.reshape(-1, 1)
    denominator = np.where(denominator == 0, EPSILON, denominator)

    # Compute the update
    numerator /= denominator
    delta_H = numerator

    return delta_H


def _kl_divergence(X, W, H):
    """KL divergence for X ~ WH."""
    WH = np.dot(W, H)
    WH_raveled = WH.ravel()
    X_raveled = X.ravel()

    # Prevent the algorithm from crashing.
    X_not_zero = X_raveled != 0
    WH_raveled = WH_raveled[X_not_zero]
    X_raveled = X_raveled[X_not_zero]
    WH_raveled[WH_raveled == 0] = EPSILON

    sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
    div = X_raveled / WH_raveled
    res = np.dot(X_raveled, np.log(div))
    res += sum_WH - np.sum(X_raveled)

    return res


def fit_nmf(X, k, max_iter=200, tol=1e-4, verbose=0, random_state=None,
            use_numba=False):
    """Fit KL-NMF using multiplicative updates."""
    W, H = sklearn.decomposition._nmf._initialize_nmf(
        X, k, 'random', random_state=random_state)

    # Set up initial values.
    start_time = time.time()
    error_at_init = _kl_divergence(X, W, H)
    previous_error = error_at_init
    errors = []

    for n_iter in range(1, max_iter + 1):
        if use_numba:
            delta_W = _mu_W_jit(X, W, H)
        else:
            delta_W = _mu_W(X, W, H)
        W *= delta_W

        if use_numba:
            delta_H = _mu_H_jit(X, W, H)
        else:
            delta_H = _mu_H(X, W, H)
        H *= delta_H

        # Test for convergence every 10 iterations.
        if n_iter % 10 == 0:
            error = _kl_divergence(X, W, H)
            errors.append(error)
            if verbose:
                elapsed = time.time() - start_time
                logging.info(
                    "Iteration %02d after %.3f seconds, error: %f".format(
                        n_iter, elapsed, error))
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    if verbose and n_iter % 10 != 0:
        elapsed = time.time() - start_time
        logging.info(
            "Stopped after iteration %02d after %.3f seconds, error: %f".format(
                n_iter, elapsed, error))

    # Scale W and H such that W columns sum to 1.
    W_colsums = np.sum(W, axis=0)
    W /= W_colsums[np.newaxis, :]
    H *= W_colsums[:, np.newaxis]
    return W, H, n_iter, errors
