"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def OTF(Y,X0,A,h,t,Noise,parameters):
    """
    Optimal transport filter (OTF) function according to Algorithm 3 in
    [Al-Jarrah, M., Jin, N., Hosseini, B. and Taghvaei, A., 2024, July. 
     Nonlinear Filtering with Brenier Optimal Transport Maps. 
     In International Conference on Machine Learning (pp. 813-839). PMLR.]
    
     --Side note: In the paper, the algorithm was called optimal transport 
     particle filter (OTPF) which is not an approrate name for the algorithm 
     since we aren't working with weighted particles--
     
    where:
        
    Y     : True observations with shape (NUM_SIM x T x dy),
    X0    : Initial particles with shape (NUM_SIM x L x N),
    A     : Deterministic dynamic model (without noise),
    h     : Deterministic observation model (without noise),
    t     : Time vector (e.g., t = 0.0, dt, 2*dt, ..., tf),
    Noise : A list [sigma, gamma] defining the noise levels in the dynamics (sigma) and observations (gamma),
    param : Dictionary of hyperparameters for the neural network components.
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
    
    # Retrieve hyperparameters for the neural networks from the parameters dictionary.
    normalization = parameters['normalization']
    NUM_NEURON = parameters['NUM_NEURON']
    INPUT_DIM = parameters['INPUT_DIM']
    BATCH_SIZE =  parameters['BATCH_SIZE']
    LearningRate = parameters['LearningRate']
    ITERATION = parameters['ITERATION']
    Final_Number_ITERATION = parameters['Final_Number_ITERATION']
    K_in = parameters['K_in'] 
    num_resblocks = parameters['num_resblocks']
    
    # Choose computation device; default is CPU but alternatives (e.g., CUDA, MPS) can be enabled.
    device = torch.device('cpu') # Recommended for problems with low dimensional problems and M chip
    # device = torch.device('mps' if torch.backends.mps.is_built() else 'cpu') # Apple M Chip
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Cuda GPU
      
    class ResidualBlock(nn.Module):
        def __init__(self, hidden_dim, activation):
            super(ResidualBlock, self).__init__()
            # Each block consists of two linear layers
            self.linear1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.activation = activation

        def forward(self, x):
            identity = x  # save input for skip connection
            out = self.linear1(x)
            out = self.activation(out)
            out = self.linear2(out)
            # Add skip connection and apply activation
            out = self.activation(out + identity)
            return out

    # Define the neural network of f funtion.
    class f_NeuralNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_resblocks=2):
            """
            Parameters:
                input_dim (tuple): A tuple where input_dim[0] is the output dimension and 
                                   input_dim[1] is the second part of the input dimension.
                hidden_dim (int): The number of neurons in the hidden layers.
                num_resblocks (int): Number of residual blocks to use.
            """
            super(f_NeuralNet, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.activation = nn.ReLU()  # you can change to nn.SiLU(), nn.ELU(), nn.Sigmoid(), etc.
            
            # Input layer: transforms concatenated inputs to the hidden dimension
            self.layer_input = nn.Linear(self.input_dim[0] + self.input_dim[1], self.hidden_dim, bias=False)
            
            # Create a sequence of residual blocks stored in a ModuleList
            self.resblocks = nn.ModuleList([
                ResidualBlock(self.hidden_dim, self.activation) for _ in range(num_resblocks)
            ])
            
            # Output layer: maps from the hidden dimension to the desired output dimension
            self.layer_out = nn.Linear(self.hidden_dim, 1, bias=False)
            
        def forward(self, x, y):
            # Concatenate x and y along the feature dimension
            inp = torch.concat((x, y), dim=1)
            # Transform the input data to the hidden space
            inp = self.layer_input(inp)
            
            out = inp
            
            # Pass through the stack of residual blocks
            for block in self.resblocks:
                out = block(out)
            
            # Optionally, apply an activation before the output layer
            out = self.activation(out)
            # Map to the output dimension
            out = self.layer_out(out)
            return out
    
    # Define the neural network of the map T.
    class T_NeuralNet(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_resblocks=2):
            """
            Parameters:
                input_dim (tuple): A tuple where input_dim[0] is the output dimension and 
                                   input_dim[1] is the second part of the input dimension.
                hidden_dim (int): The number of neurons in the hidden layers.
                num_resblocks (int): Number of residual blocks to use.
            """
            super(T_NeuralNet, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.activation = nn.ReLU()  # you can change to nn.SiLU(), nn.ELU(), nn.Sigmoid(), etc.
            
            # Input layer: transforms concatenated inputs to the hidden dimension
            self.layer_input = nn.Linear(self.input_dim[0] + self.input_dim[1], self.hidden_dim, bias=False)
            
            # Create a sequence of residual blocks stored in a ModuleList
            self.resblocks = nn.ModuleList([
                ResidualBlock(self.hidden_dim, self.activation) for _ in range(num_resblocks)
            ])
            
            # Output layer: maps from the hidden dimension to the desired output dimension
            self.layer_out = nn.Linear(self.hidden_dim, self.input_dim[0], bias=False)
            
        def forward(self, x, y):
            # Concatenate x and y along the feature dimension
            inp = torch.concat((x, y), dim=1)
            # Transform the input data to the hidden space
            inp = self.layer_input(inp)
            
            out = inp
            
            # Pass through the stack of residual blocks
            for block in self.resblocks:
                out = block(out)
            
            # Optionally, apply an activation before the output layer
            out = self.activation(out)
            # Map to the output dimension
            out = self.layer_out(out)
            return out
    
    # Define a helper function to initialize the weights of linear layers.    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            # torch.nn.init.xavier_normal_(m.weight)
            # torch.nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
            # torch.nn.init.kaiming_uniform_(m.weight,mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                m.bias.data.fill_(0.1) #0.001
                    
    # Define the training routine for the neural networks within the OT filter.
    def train(f, T_net, X_Train, Y_Train, iterations, learning_rate, ts, Ts, batch_size, k, K, K_in):
        # Set networks to training mode.
        f.train()
        T_net.train()

        # Initialize separate Adam optimizers for both networks.
        optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate[0])
        optimizer_T = torch.optim.Adam(T_net.parameters(), lr=learning_rate[1])

        # Configure learning rate schedulers using an exponential decay.
        scheduler_f = ExponentialLR(optimizer_f, gamma=0.999)
        scheduler_T = ExponentialLR(optimizer_T, gamma=0.999)

        inner_iterations = K_in
        # Shuffle the observation training data to improve learning.
        Y_Train_shuffled = Y_Train[torch.randperm(Y_Train.shape[0])].view(Y_Train.shape)
        for i in range(iterations):
            idx = torch.randperm(X1.shape[0])[:batch_size]
            X_train = X_Train[idx].clone().detach()
            Y_train = Y_Train[idx].clone().detach()

            # Randomly shuffle Y_train for training the transport mapping.
            Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)
            for j in range(inner_iterations):
                map_T = T_net.forward(X_train, Y_shuffled)
                f_of_map_T = f.forward(map_T, Y_shuffled)
                loss_T = - f_of_map_T.mean() + 0.5 * ((X_train - map_T) ** 2).sum(axis=1).mean()

                optimizer_T.zero_grad()
                loss_T.backward()
                optimizer_T.step()

            # Update the mapping network f based on current samples.
            f_of_xy = f.forward(X_train, Y_train)
            map_T = T_net.forward(X_train, Y_shuffled)
            f_of_map_T = f.forward(map_T, Y_shuffled)
            loss_f = - f_of_xy.mean() + f_of_map_T.mean()

            optimizer_f.zero_grad()
            loss_f.backward()
            optimizer_f.step()

            scheduler_f.step()
            scheduler_T.step()

            if (i + 1) == iterations:
                f.eval()
                T_net.eval()
                with torch.no_grad():
                    f_of_xy = f.forward(X_Train, Y_Train)
                    map_T = T_net.forward(X_Train, Y_Train_shuffled)
                    f_of_map_T = f.forward(map_T, Y_Train_shuffled)
                    loss_f = f_of_xy.mean() - f_of_map_T.mean()
                    loss = loss_f + 0.5 * ((X_Train - map_T) ** 2).sum(axis=1).mean()
                    print("Simu#%d/%d, Time Step:%d/%d, Iteration: %d/%d, loss = %.4f" %
                          (k + 1, K, ts, Ts - 1, i + 1, iterations, loss.item()))

    start_time = time.time()
    # Initialize the output array for the EnKF estimations.
    # Its shape is (NUM_SIM x T x N x L) to store results for each simulation, time step, particle, and state dimension.
    X_OTF = torch.zeros((NUM_SIM, T, N, L),device=device,dtype=torch.float32)
    
    # Loop over each simulation.
    for k in range(NUM_SIM):
        
        # Retrieve the observations for the current simulation.
        y = Y[k,]
         

        # Set the initial condition for the current simulation (transposed to match dimensions).
        X_OTF[k,0,] = torch.from_numpy(X0[k,].T).to(torch.float32).to(device)
        
        ITERS = ITERATION
        LR = LearningRate
        
        # Instantiate the two neural network models.
        f = f_NeuralNet(INPUT_DIM, NUM_NEURON,num_resblocks[0])
        MAP_T = T_NeuralNet(INPUT_DIM, NUM_NEURON,num_resblocks[1])
        
        f.to(device)
        MAP_T.to(device) 
        
        # Apply custom weight initialization.
        f.apply(init_weights)
        MAP_T.apply(init_weights)     
        for i in range(T-1):
            # Generate process noise for all particles at the current time step (shape: N x L).
            x_noise = torch.distributions.MultivariateNormal(torch.zeros(L), covariance_matrix=sigma * sigma * torch.eye(L))
            # Propagate particles using the model A with added noise.
            X1 = X_OTF[k,i,] + tau * A(X_OTF[k,i,].T,t[i]).T  + x_noise.sample((N,)).to(device)
            
            # Generate observation noise for each particle (shape: N x dy).
            y_noise = torch.distributions.MultivariateNormal(torch.zeros(dy), covariance_matrix=gamma * gamma * torch.eye(dy))
            # Compute the predicted observation for each particle with added noise.
            Y1 = h(X1.T).T + y_noise.sample((N,)).to(device)
            
            # Optionally normalize the particles and predicted observations using the specified scaling method.
            if normalization == 'Standard':
                scaler_X = StandardScaler()
                scaler_Y = StandardScaler()
                
                X1 = torch.tensor(scaler_X.fit_transform(X1.cpu()),device=device,dtype=torch.float32)
                Y1 = torch.tensor(scaler_Y.fit_transform(Y1.cpu()),device=device,dtype=torch.float32)
                
            elif  normalization == 'MinMax':
                scaler_X = MinMaxScaler()
                scaler_Y = MinMaxScaler()
                
                X1 = torch.tensor(scaler_X.fit_transform(X1.cpu()),device=device,dtype=torch.float32)
                Y1 = torch.tensor(scaler_Y.fit_transform(Y1.cpu()),device=device,dtype=torch.float32)
            
            # Train the mapping networks using the current particle and observation data.
            train(f,MAP_T,X1,Y1,ITERS,LR,i+1,T,BATCH_SIZE,k,NUM_SIM,K_in)
            
            # Dynamically reduce the number of training iterations if conditions are met.
            if ITERS > Final_Number_ITERATION and i%1 == 0 and i>=5:
                ITERS = int(ITERS/2)
            
            # Prepare the true observation at the next time step for input to the transport network.
            Y1_true = y[i+1,:]
            if normalization == 'Standard' or normalization == 'MinMax':
                Y1_true = scaler_Y.transform(Y1_true)
            Y1_true = torch.from_numpy(Y1_true)
            Y1_true = Y1_true.to(torch.float32)
            
            # Apply the transport network to the propagated particles.
            X_mapped = MAP_T.forward(X1, (Y1_true*torch.ones((N,dy))).to(device))
            
            # If normalization was used, reverse the scaling on the transported particle states.
            if normalization == 'Standard' or normalization == 'MinMax':  
                X_mapped = torch.tensor(scaler_X.inverse_transform(X_mapped.cpu().detach().numpy()))
            # Update the particle state at the next time step with the newly mapped values.
            X_OTF[k,i+1,] = X_mapped.detach()
            
            

    print("--- OT time : %s seconds ---" % (time.time() - start_time))
    # Rearrange the dimensions of the output to (NUM_SIM x T x L x N) for convenient plotting.
    return X_OTF.cpu().numpy().transpose(0,1,3,2)