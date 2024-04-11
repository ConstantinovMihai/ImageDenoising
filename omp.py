"""
This file contains an own implementation of the classical
Orthogonal Matching Pursuit (OMP) algorithm. 
Implemented mostly for learning purposes, as it is already implemented in 
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuit.html#sklearn.linear_model.OrthogonalMatchingPursuit
The results are a bit different, as sklearn uses Batch Orthogonal Matching Pursuit Algorithm
"""

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
import scipy as sp
import matplotlib.pyplot as plt
from time import time
from scipy import linalg

np.random.seed(42)

def unsparse(v, idx, length):
    #print(f"{v.shape=}; {len(idx)=}; {length=}")
    """Transform a vector-index pair to a dense representation"""
    x = np.zeros(length)
    x[idx] = v
    return x


def orthogonal_matching_pursuit(A, y, K=None, error=1e-6):
    """
    Orthogonal Matching Pursuit algorithm.
    
    Parameters:
    A (np.array): Dictionary matrix.
    y (np.array): Signal vector.
    K (int, optional): Target sparsity. If not provided, the algorithm uses the target error.
    error (float, optional): Target error. Ignored if K is provided.
    
    Returns:
    gamma (np.array): Sparse representation of y.
    """
    # Initialization
    idx = []
    r = y.copy()
   
    # Main loop
    while True:
        # Compute inner products
        # Select atom with maximum inner product
        k_hat = np.argmax(np.abs(A.T @ r))
        # Update support
        idx.append(k_hat)
        # Update gamma
        gamma, _, _, _ = linalg.lstsq(A[:, idx], y)
        # Update residual
        r = y - np.dot(A[:, idx], gamma)
        
        # Check stopping criterion
        if K is not None:
            if len(idx) >= K:
                break
        else:
            if np.linalg.norm(r) <= error:
                break
    
    return gamma, idx


def orthogonal_matching_pursuit_cholensky(D, x, K=None, error_tol=1e-6):
    """
    Orthogonal Matching Pursuit algorithm.
    
    Parameters:
    A (np.array): Dictionary matrix.
    x (np.array): Signal vector.
    K (int, optional): Target sparsity. If not provided, the algorithm uses the target error.
    error (float, optional): Target error. Ignored if K is provided.
    
    Returns:
    gamma (np.array): Sparse representation of y.
    """
    # Initialization
    idx = []
    L = np.array([[1]])
    r = x
    gamma = np.zeros(D.shape[1])
    alpha = D.T @ x
    n = 1

    while True:
        # Stopping criterion
        if K is not None and n > K:
            break
        if error_tol is not None and np.linalg.norm(r) <= error_tol:
            break

        # Select atom with maximum correlation with residual
        # select column with maximum correlation with residual
        k_hat = np.argmax(np.abs(D.T @ r))
        # k_hat = np.argmax(np.abs(D.T @ r))
        print(f"{k_hat=}; {n=}; {(D.T@r).shape=}")
        if n > 1:
            w = sp.linalg.solve_triangular(L, np.dot(D[:, idx].T, D[:, k_hat]),
                                    lower=True, unit_diagonal=True)
            # should the diagonal be unit in theory? It crashes without it
           
            L = np.r_[np.c_[L, np.zeros(len(L))],
                    np.atleast_2d(np.append(w.T, np.sqrt(1 - np.dot(w.T, w))))]
        
        # Update support
        idx.append(k_hat)

        # Update sparse representation
        Ltc = sp.linalg.solve_triangular(L, alpha[idx], lower=True)

        gamma = sp.linalg.solve_triangular(L, Ltc, trans=1, lower=True)
        # Update residual
        r = x - D[:, idx] @ gamma
        n += 1
    
    return gamma, idx


def orthogonal_matching_pursuit_cholensky_batches(D, x, K=None, error_tol=1e-6):
    """
    Orthogonal Matching Pursuit algorithm.
    
    Parameters:
    A (np.array): Dictionary matrix.
    x (np.array): Signal vector.
    K (int, optional): Target sparsity. If not provided, the algorithm uses the target error.
    error (float, optional): Target error. Ignored if K is provided.
    
    Returns:
    gamma (np.array): Sparse representation of y.
    """
    # Initialization
    idx = []
    L = np.array([[1]])
    r = x
    gamma = np.zeros(D.shape[1])
    alpha = D.T @ x
    n = 1

    while True:
        # Stopping criterion
        if K is not None and n > K:
            break
        if error_tol is not None and np.linalg.norm(r) <= error_tol:
            break

        # Select atom with maximum correlation with residual
        # select column with maximum correlation with residual
        k_hat = np.argmax(np.linalg.norm(D.T @ r, axis=1))
        # k_hat = np.argmax(np.abs(D.T @ r))
        print(f"{k_hat=}; {n=}; {(D.T@r).shape=}")
        if n > 1:
            w = sp.linalg.solve_triangular(L, np.dot(D[:, idx].T, D[:, k_hat]),
                                    lower=True, unit_diagonal=True)
            # should the diagonal be unit in theory? It crashes without it
           
            L = np.r_[np.c_[L, np.zeros(len(L))],
                    np.atleast_2d(np.append(w.T, np.sqrt(1 - np.dot(w.T, w))))]
        
        # Update support
        idx.append(k_hat)

        # Update sparse representation
        Ltc = sp.linalg.solve_triangular(L, alpha[idx], lower=True)

        gamma = sp.linalg.solve_triangular(L, Ltc, trans=1, lower=True)
        # Update residual
        r = x - D[:, idx] @ gamma
        n += 1
    
    return gamma, idx


def orthogonal_matching_pursuit_batch(D, x, K=None, error_tol=1e-6):
    """
    Orthogonal Matching Pursuit algorithm.
    
    Parameters:
    A (np.array): Dictionary matrix.
    x (np.array): Signal vector.
    K (int, optional): Target sparsity. If not provided, the algorithm uses the target error.
    error (float, optional): Target error. Ignored if K is provided.
    
    Returns:
    gamma (np.array): Sparse representation of y.
    """
    # Initialization
    idx = []
    L = np.array([[1]])
    gamma = np.zeros(D.shape[1])
    alpha_0 = D.T @ x
    G = D.T @ D
    alpha = alpha_0
    n = 1
    while True:
        print(f"N is {n}; K is {K}")
        # Stopping criterion
        if K is not None and n > K:
            break
        # if eps < error_tol:
        #     break

        # Select atom with maximum correlation with residual
        k_hat = np.argmax(np.abs(alpha))
        if n > 1:
            w = sp.linalg.solve(L, G[:, idx].T @ G[:, k_hat])
            # should the diagonal be unit in theory? It crashes without it
           
            L = np.r_[np.c_[L, np.zeros(len(L))],
                    np.atleast_2d(np.append(w.T, np.sqrt(1 - np.dot(w.T, w))))]
            # check if L contains NaNs
            if np.isnan(L).any():
                print("L contains NaNs")
                break
        # Update support
        idx.append(k_hat)

        # Update sparse representation

        gamma = sp.linalg.solve(L @ L.T, alpha_0[idx])
        beta = G[:,idx] @ gamma
        alpha = alpha_0 - beta
        n += 1
    
    return gamma, idx


def batch_omp(D, x, K, eps_0=None, eps=None):
    """
    Orthogonal Matching Pursuit (OMP) algorithm for sparse signal recovery.

    Parameters:
    - D: numpy array
        Dictionary matrix of shape (m, n), where m is the number of atoms and n is the signal dimension.
    - x: numpy array
        Input signal of shape (n,).
    - K: int
        Maximum number of atoms to select.
    - eps_0: float, optional
        Initial error tolerance. Default is None.
    - eps: float, optional
        Final error tolerance. Default is None.

    Returns:
    - gamma: numpy array
        Sparse coefficient vector of shape (K,).
    - idx: list
        List of indices corresponding to the selected atoms.

    """
    # initialisation
    idx = []
    alpha_0 = np.dot(D.T, x)
    G = np.dot(D.T, D)
    L = np.ones((1, 1))
    alpha = alpha_0
    eps_curr = eps_0
    delta = 0
    it = 0
    if eps == None:
        stopping_condition = lambda: it == K
    else:
        stopping_condition = lambda: eps_curr <= eps
    while not stopping_condition():
        print(it)
        lam = np.abs(alpha).argmax()
        if len(idx) > 0:
            w = linalg.solve_triangular(L, G[idx, lam],
                                        lower=True,  unit_diagonal=True)
            L = np.r_[np.c_[L, np.zeros(len(L))],
                        np.atleast_2d(np.append(w, np.sqrt(1 - np.inner(w, w))))]
        idx.append(lam)
        it += 1
        Ltc = linalg.solve_triangular(L, alpha_0[idx], lower=True)
        gamma = linalg.solve_triangular(L, Ltc, trans=1, lower=True) 
        beta = np.dot(G[:, idx], gamma)        
        alpha = alpha_0 - beta
        if eps != None:
            eps_curr += delta
            delta = np.inner(gamma, beta[idx])
            eps_curr -= delta
    return gamma, idx                 