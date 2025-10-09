"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import time


def EnKF(Y, X0, A, h, t, Noise, SIGMA=1e-6):
    """
    Ensemble Kalman Filter (EnKF) function according to Algorithm 1 in
    [Al-Jarrah, M., Jin, N., Hosseini, B. and Taghvaei, A., 2024, July. 
     Nonlinear Filtering with Brenier Optimal Transport Maps. 
     In International Conference on Machine Learning (pp. 813-839). PMLR.]
    
    where:
        
    Y     : True observations with shape (NUM_SIM x T x dy),
    X0    : Initial particles with shape (NUM_SIM x L x N),
    A     : Deterministic dynamic model (without noise),
    h     : Deterministic observation model (without noise),
    t     : Time vector (e.g., t = 0.0, dt, 2*dt, ..., tf),
    Noise : A list [sigma, gamma] defining the noise levels in the dynamics (sigma) and observations (gamma),
    SIGMA : A small positive constant (nugget) added to maintain the invertibility of C_t^{yy}.
    """

    # Determine the dimensions from X0:
    # X0 has shape (NUM_SIM x L x N) where NUM_SIM is the number of simulations,
    # L is the state dimension, and N is the number of particles.
    NUM_SIM = X0.shape[0]
    L = X0.shape[1]
    N = X0.shape[2]
    
    # Extract the dimensions from Y:
    # Y has shape (NUM_SIM x T x dy) where T is the number of time steps,
    # and dy is the observation dimension.
    T = Y.shape[1]
    dy = Y.shape[2]

    # Unpack the noise levels:
    # sigma corresponds to noise in the hidden state,
    # gamma corresponds to noise in the observation.
    sigma = Noise[0]
    gamma = Noise[1]
    
    # Calculate the time step based on the time vector.
    tau = t[1] - t[0]
    
    start_time = time.time()
    # Initialize the output array for the EnKF estimations.
    # Its shape is (NUM_SIM x T x N x L) to store results for each simulation, time step, particle, and state dimension.
    X_EnKF = np.zeros((NUM_SIM, T, N, L))
    
    # Loop over each simulation.
    for k in range(NUM_SIM):
        # Retrieve the observations for the current simulation.
        y = Y[k,]
        
        # Set the initial condition for the current simulation (transposed to match dimensions).
        X_EnKF[k, 0] = X0[k,].T 
        
        # Time stepping for the filter (excluding the final time step).
        for i in range(T - 1):
            # Generate process noise for all particles at the current time step (shape: N x L).
            x_noise = np.random.multivariate_normal(np.zeros(L), sigma * sigma * np.eye(L), N)
            # Propagate particles using the model A with added noise.
            x_hatEnKF = X_EnKF[k, i,] + tau * A(X_EnKF[k, i,].T, t[i]).T + x_noise
            
            # Generate observation noise for each particle (shape: N x dy).
            y_noise = np.random.multivariate_normal(np.zeros(dy), gamma * gamma * np.eye(dy), N)
            # Compute the predicted observation for each particle with added noise.
            y_hatEnKF = h(x_hatEnKF.T).T + y_noise
            
            # Calculate the ensemble means of the propagated state and predicted observation.
            X_hat = x_hatEnKF.mean(axis=0, keepdims=True)
            Y_hat = y_hatEnKF.mean(axis=0, keepdims=True)

            # Compute deviations from the mean for state and observation.
            a = x_hatEnKF - X_hat
            b = y_hatEnKF - Y_hat

            # Calculate the C_t^{xy}, C_t^{yy} matrices.
            C_xy = 1 / N * a.T @ b
            C_yy = 1 / N * b.T @ b 
            
            # Compute the Kalman gain matrix using the covariance matrices.
            K = C_xy @ np.linalg.inv(C_yy + np.eye(dy) * SIGMA)
            
            # Update the state estimates using the Kalman gain and the innovation (observation error).
            X_EnKF[k, i + 1, :, :] = x_hatEnKF + (K @ (y[i + 1, :] - y_hatEnKF.T)).T 
        
    print("--- EnKF running time : %s seconds ---" % (time.time() - start_time))
    # Rearrange the dimensions of the output to (NUM_SIM x T x L x N) for convenient plotting.
    return X_EnKF.transpose(0, 1, 3, 2)
