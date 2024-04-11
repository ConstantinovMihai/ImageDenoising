"""
This file contains various tests for the implemenations of the OMP algorithms
"""

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit as OMP
import scipy as sp
import matplotlib.pyplot as plt
from time import time
from scipy import linalg
from omp import *

np.random.seed(42)

def generate_dict(n_features, n_components):
    # generate random dictionary
    D = np.random.randn(n_components, n_features)
    D /= np.apply_along_axis(lambda x: np.sqrt(np.dot(x.T, x)), 0, D)
    return D
 

def generate_data(D, sparsity):
    n_features = D.shape[1]
    # generate sparse signal
    x = np.zeros(n_features)
    indices = np.random.randint(0, n_features, sparsity)
    x[indices] = np.random.normal(0, 5, sparsity)

    return (indices, x), np.dot(D, x)


def test_OMP_implementation(cholenski=True):
    """ Simple unit test to check OMP implementation
    """
    n_features, n_components = 64, 128
    D = generate_dict(n_features, n_components)
    sparsity_level = 20
  
    Y = np.zeros((n_components,1))
    X = np.zeros((n_features,1))   

    (_, X) , Y = generate_data(D, sparsity_level)

    if cholenski:
        x_ours, idx = orthogonal_matching_pursuit_cholensky(D, Y, sparsity_level)
    else:
        x_ours, idx = orthogonal_matching_pursuit(D, Y, sparsity_level)
    # Apply Orthogonal Matching Pursuit algorithm
    omp = OMP(n_nonzero_coefs=sparsity_level)
    omp.fit(D, Y)
    
    print(f"{D.shape=}' {Y.shape=}")
    
    x_recovered = omp.coef_
    print(f"{x_recovered.shape=}")

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((X - unsparse(x_ours, idx, n_features)) ** 2)
    print("MSE ours:", mse)

    plt.scatter(np.arange(len(X)), X, label="orig", alpha=0.8)
    # plt.scatter(np.arange(len(X)), unsparse(x_ours, idx, n_features), label="reconstructed", alpha=0.8)
    plt.scatter(np.arange(len(X)), X - unsparse(x_ours, idx, n_features), label="diff")
    plt.legend()
    plt.show()
    

    mse_ours = np.mean((X - x_recovered) ** 2)
    print("MSE sklearn:", mse_ours)

def cholesky_omp2(D, x, m, eps=None):
    if eps == None:
        stopping_condition = lambda: it == m  # len(idx) == m
    else:
        stopping_condition = lambda: np.inner(residual, residual) <= eps

    alpha = np.dot(x, D)
    
    #first step:        
    it = 1
    lam = np.abs(np.dot(x, D)).argmax()
    idx = [lam]
    L = np.ones((1,1))
    gamma = linalg.lstsq(D[:, idx], x)[0]
    residual = x - np.dot(D[:, idx], gamma)
    
    while not stopping_condition():
        lam = np.abs(np.dot(residual, D)).argmax()
        w = linalg.solve_triangular(L, np.dot(D[:, idx].T, D[:, lam]),
                                    lower=True, unit_diagonal=True)
        # should the diagonal be unit in theory? It crashes without it
        L = np.r_[np.c_[L, np.zeros(len(L))],
                  np.atleast_2d(np.append(w, np.sqrt(1 - np.dot(w.T, w))))]
        idx.append(lam)
        it += 1
        #gamma = linalg.solve(np.dot(L, L.T), alpha[idx], sym_pos=True)
        # what am I, stupid??
        Ltc = linalg.solve_triangular(L, alpha[idx], lower=True)
        gamma = linalg.solve_triangular(L, Ltc, trans=1, lower=True)
        residual = x - np.dot(D[:, idx], gamma)


def test_OMP_cholensky_implementation():
    """ Simple unit test to check OMP implementation
    """
    # # Parameters
    # N = 100  # Signal length
    # M = 50   # Number of measurements
    # sparsity_level = 15  # Number of non-zero elements in signal

    # # Generate random sparse signal
    # x = np.zeros(N)
    # indices = np.random.choice(N, sparsity_level, replace=False)
    # x[indices] = np.random.randn(sparsity_level)

    # # Generate random measurement matrix
    # A = np.random.randn(M, N)

    # # Generate noisy measurements
    # noise_level = 0.1
    # y = np.dot(A, x) + noise_level * np.random.randn(M)

    n_features, n_components = 512, 1024
    D = generate_dict(n_features, n_components)
    sparsities = 12
  
    Y = np.zeros(n_components)
    X = np.zeros(n_features)    
    
    (_, X), Y = generate_data(D, sparsities)

    # Apply Orthogonal Matching Pursuit algorithm
    omp = OMP(n_nonzero_coefs=sparsities)
    omp.fit(D, Y)
  
    x_recovered = omp.coef_
    print(f"{D.shape=};, {Y.shape=};, {x_recovered.shape=}")
    # Calculate Mean Squared Error (MSE)
    mse = np.mean((X - x_recovered) ** 2)

    print("MSE sklearn:", mse)

    # Apply our OMP implementation
    gamma_ours, idx = orthogonal_matching_pursuit_cholensky(D, Y, sparsities)
    mse_ours = np.mean((X - unsparse(gamma_ours, idx, n_features)) ** 2)
    print("MSE ours:", mse_ours)



def bench_plot():    
    n_features, n_components = 512, 1024
    D = generate_dict(n_features, n_components)
    sparsities = np.arange(50, 200, 15)
    # sparsities = np.array([50])
  
    Y = np.zeros((n_components, len(sparsities)))
    X = np.zeros((n_features, len(sparsities)))    
    for i, sp in enumerate(sparsities):
        (_, X[:, i]), Y[:, i] = generate_data(D, sp)
   

    naive, cholesky, batch = [], [], []
    naive_err, cholesky_err, batch_err = [], [], []
    for i in range(len(sparsities)):
        #print "sparsity: ", sparsities[i]
        t0 = time()
        x, idx = orthogonal_matching_pursuit(D, Y[:, i], sparsities[i])
        naive.append(time() - t0)
        naive_err.append(linalg.norm(X[:, i] - unsparse(x, idx, n_features)))
        t0 = time()
        x, idx = orthogonal_matching_pursuit_cholensky(D, Y[:, i], sparsities[i])
        cholesky.append(time() - t0)        
        cholesky_err.append(linalg.norm(X[:, i] - unsparse(x, idx, n_features)))
        t0 = time()
        # x = orthogonal_matching_pursuit_batch(D, Y[:, i], sparsities[i])
        # batch_err.append(linalg.norm(X[:, i] - x))
        x, idx = batch_omp(D, Y[:, i], sparsities[i])
        batch.append(time() - t0)
        batch_err.append(linalg.norm(X[:, i] - unsparse(x, idx, n_features)))
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.xlabel('Sparsity level')
    plt.ylabel('Time')
    plt.plot(sparsities, naive, 'o-', label="Vanilla OMP")
    plt.plot(sparsities, cholesky, 'o-', label="Cholesky OMP")
    plt.plot(sparsities, batch, 'o-', label="Batch OMP")
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Sparsity level')
    plt.ylabel('Error')   
    plt.plot(sparsities, naive_err, 'o-', label="Vanilla OMP")
    plt.plot(sparsities, cholesky_err, 'o-', label="Cholesky update OMP")
    plt.plot(sparsities, batch_err, 'o-', label="Batch OMP")
    plt.grid()
    plt.legend()


    plt.show()
    
def plot_reconstruction():
    # init 
    D = generate_dict(n_features=512, n_components=100)
    sparsity = 17
    (indices, x), y = generate_data(D, sparsity)
    plt.subplot(3, 1, 1)
    plt.title("Sparse signal")
    plt.stem(indices, x[indices], use_line_collection=True)
    y_noise = y + np.random.normal(0, 0.15, y.shape)
    
    x_r, i_r = orthogonal_matching_pursuit_cholensky(D, y, sparsity)
    plt.subplot(3, 1, 2)
    plt.title("Recovered signal from noise-free measurements")    
    plt.stem(i_r, x_r, use_line_collection=True)
    
    x_r, i_r = orthogonal_matching_pursuit_cholensky(D, y_noise, sparsity)
    plt.subplot(3, 1, 3)
    plt.title("Recovered signal from noisy measurements")    
    plt.stem(i_r, x_r, use_line_collection=True)
    plt.show()
    
if __name__ == '__main__':    
    #test_OMP_implementation(cholenski=False)
    test_OMP_cholensky_implementation()
    bench_plot()
    plot_reconstruction()