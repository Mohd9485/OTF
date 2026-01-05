
"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
import torch, time
import sys
import matplotlib
from EnKF import EnKF
from SIR import SIR
from OTF import OTF

# Configure matplotlib to embed fonts in PDF/PS outputs and set default font sizes.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=16)

# Close any open figure windows to start fresh.
plt.close('all')

# Set a fixed random seed for reproducibility.
# np.random.seed(0)


# Choose h(x) here, the observation rule
def h(x):
    # return x[0].reshape(1,-1)*x[0].reshape(1,-1)
    return x*x

def A(x,t=0):
    try:
        return F @ x
    except:
        return torch.from_numpy(F).to(torch.float32) @ x

def Gen_True_Data(L, dy, T, sigma0, sigma, gamma, tau):
    """
    Generates true state and observation data using the Linear-Quadratic model.
    
    For a given number of time steps, the true state is evolved by the
    Linear model. The observations are generated using an observation rule
    (function h) with added Gaussian noise.
    
    Parameters
    ----------
    L : int
        Dimension of the state space.
    dy : int
        Dimension of the observation space.
    T : int
        Number of time steps.
    sigma0 : float
        Standard deviation for the initial state distribution.
    sigma : float
        Standard deviation for the process noise.
    gamma : float
        Standard deviation for the observation noise.
    tau : float
        Time step size.
    
    Returns
    -------
    x : ndarray
        True state evolution with shape (T x L x 1).
    y : ndarray
        Observations with shape (T x dy x 1).
    """
    # Initialize arrays for state and observation data.
    x = np.zeros((T, L, 1))
    y = np.zeros((T, dy, 1))
    
    # Set initial state with added noise from a multivariate normal distribution.
    x[0,] = np.random.multivariate_normal(np.zeros(L), sigma0 * sigma0 * np.eye(L), 1).T

    for i in range(T - 1):
        # Propagate the state using the Linear dynamics.
        x[i + 1, :] =  A(x[i, :]) + np.random.multivariate_normal(np.zeros(L),sigma*sigma * np.eye(L),1).T
        # Generate the observation by applying the observation function h and adding noise.
        y[i + 1, :] = h(x[i + 1, :]) + np.random.multivariate_normal(np.zeros(dy), gamma * gamma * np.eye(dy), 1).T
    
    return x, y

#%%
# Simulation parameters.
n = 2 
L = n*2 # number of states
tau = 1e-1 # timpe step 
T = int(5/tau) # number of time steps T = 5 s
dy = L # number of states observed
t = np.arange(0.0, tau * T, tau)  # Time vector.

# dynmaical system
# H = np.array([[1,0]]) 
H = np.eye(1,dy)
alpha = 0.9
a = alpha 
b = np.sqrt(1-alpha**2)
# c = alpha

F = np.array([[a, -b],[b,a]]) 

F = np.kron(np.eye(int(n)), F)
# F = np.eye(2*n)*0.9


noise = np.sqrt(1e-2) # noise level std
sigma = noise # Noise in the hidden state
sigma0 = 1#5*noise # Noise in the initial state distribution
gamma = noise # Noise in the observation
x0_amp = 1#/noise # Amplifiying the initial state 
Noise = [sigma, gamma]





N = int(1e4)  # Number of ensemble particles.
NUM_SIM = 1        # Number of independent simulations.

# Define hyperparameters for the optimal transport networks.
parameters = {
    'normalization': 'None',   # Options: 'None', 'Standard', 'MinMax'
    'INPUT_DIM': [L, dy],
    'NUM_NEURON': int(64 / 1),
    'BATCH_SIZE': int(64 / 1),
    'LearningRate': [1e-5 , 1e-5],  # Learning rates for the mapping networks.
    'ITERATION': int(1024*4),
    'Final_Number_ITERATION': int(64 * 16),
    'K_in': 10,
    'num_resblocks': [2, 3]  # Number of residual blocks for the two networks (f,T).
}

# Containers for true states, observations, and initial particles.
X_True = np.zeros((NUM_SIM, T, L, 1))
Y_True = np.zeros((NUM_SIM, T, dy, 1))
X0 = np.zeros((NUM_SIM, L, N))

# Generate true state trajectories, observations, and initial particles for each simulation.
for k in range(NUM_SIM):
    X_True[k,], Y_True[k,] = Gen_True_Data(L, dy, T, sigma0, sigma, gamma, tau)
    X0[k,] = np.random.multivariate_normal(np.zeros(L), sigma0 * sigma0 * np.eye(L), N).T

# Apply filtering methods using the true observations and initial particles.
# The expected data structure is: (NUM_SIM x T x L x N)
X_EnKF = EnKF(Y_True, X0, A, h, t, Noise, SIGMA=1e-6)
X_SIR = SIR(Y_True, X0, A, h, t, Noise)
X_OTF = OTF(Y_True, X0, A, h, t, Noise, parameters)

#%%
# Plot the results for each filtering method alongside the true state.
plot_particle = 500
plt.figure(figsize=(20, 12))
for l in range(L):
    # Plot EnKF results.
    plt.subplot(L, 3, 3 * l + 1)
    plt.plot(t, X_EnKF[k, :, l, :plot_particle], 'g', alpha=0.1)
    plt.plot(t, X_True[k, :, l], 'k--', label='True state')
    plt.xlabel('Time')
    if l == 0:
        plt.title('EnKF')
        plt.legend()
    plt.ylabel(f'X({l + 1})')
    if l <= 1:
        plt.gca().get_xaxis().set_visible(False)

for l in range(L):
    # Plot SIR results.
    plt.subplot(L, 3, 3 * l + 2)
    plt.plot(t, X_SIR[k, :, l, :plot_particle], 'b', alpha=0.1)
    plt.plot(t, X_True[k, :, l], 'k--', label='True state')
    plt.xlabel('Time')
    if l == 0:
        plt.title('SIR')
    if l <= 1:
        plt.gca().get_xaxis().set_visible(False)

for l in range(L):
    # Plot OTF results.
    plt.subplot(L, 3, 3 * l + 3)
    plt.plot(t, X_OTF[k, :, l, :plot_particle], 'r', alpha=0.1)
    plt.plot(t, X_True[k, :, l], 'k--', label='True state')
    plt.xlabel('Time')
    if l == 0:
        plt.title('OTF')
    if l <= 1:
        plt.gca().get_xaxis().set_visible(False)

# Optionally, you can display the plots by calling plt.show() here.
plt.show()

