"""NeuralUQ for 1-D Poisson equation (forward), from B-PINN paper."""


# See also this paper for reference: 
# B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data

import neuraluq as neuq
from neuraluq.config import tf

from models1 import Model
from process1 import Process

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import norm


def load_data():
    data = sio.loadmat("/Users/dhulls/projects/neuraluq/dataset/Poisson_forward.mat")
    x_test, u_test, f_test = data["x_test"], data["u_test"], data["f_test"]
    x_u_train, u_train = data["x_u_train"], data["u_train"]
    x_f_train, f_train = data["x_f_train"], data["f_train"]
    return x_u_train, u_train, x_f_train, f_train, x_test, u_test, f_test


def pde_fn(x, u):
    D = 0.01
    k = 0.7
    u_x = tf.gradients(u, x)[0]
    u_xx = tf.gradients(u_x, x)[0]
    return D * u_xx + k * tf.tanh(u)


# if __name__ == "__main__":
################## Load data and specify some hyperparameters ####################
x_u_train, u_train, x_f_train, f_train, x_test, u_test, f_test = load_data()
layers = [1, 10, 1] # [1, 50, 50, 1]

####################### Build model and perform inference ########################
# All models share the same general procedure:
# Step 1: build surrogate, e.g. a fully-connected neural network, using [surrogates]
# Step 2: build prior and/or posterior using [variables]
# Step 3: build process, based the surrogate, prior and/or posterior, using [Process]
# Step 4: build likelihood, given noisy measurements, using [likelihoods]
# Step 5: build model using [models]
# Step 6: create an inference method and assign it to the model using [inferences]
# Step 7: perform posterior sampling using [model.run]

process = neuq.process.Process(
    surrogate=neuq.surrogates.FNN(layers=layers),
    prior=neuq.variables.fnn.Samplable(layers=layers, mean=0, sigma=1),
)

process1 = Process(
    surrogate=neuq.surrogates.FNN(layers=layers),
    prior=neuq.variables.fnn.Samplable(layers=layers, mean=0, sigma=1),
)

likelihood_u = neuq.likelihoods.Normal(
    inputs=x_u_train,
    targets=u_train,
    processes=[process],
    pde=None,
    sigma=0.1,
)

likelihood_f = neuq.likelihoods.Normal(
    inputs=x_f_train,
    targets=f_train,
    processes=[process],
    pde=pde_fn,
    sigma=0.1,
)

m1 = Model(
    processes=[process],
    likelihoods=[likelihood_u, likelihood_f],
)

model = neuq.models.Model(
    processes=[process],
    likelihoods=[likelihood_u, likelihood_f],
)

# method = neuq.inferences.HMC(
#     num_samples=1000,
#     num_burnin=1000,
#     init_time_step=0.01,
#     leapfrog_step=50,
#     seed=6666,
# )

# method = neuq.inferences.NUTS(
#     num_samples=10000,
#     num_burnin=1000,
#     time_step=0.00001,
#     seed=6666,
# )

rv1 = norm()
_initial_values = []
for i in range(len(layers) - 1):
    shape = [layers[i], layers[i + 1]]
    siz1 = layers[i]*layers[i+1]
    # add one axis before axis 0, for MCMC sampler
    _initial_values += [tf.constant(norm().rvs(siz1).reshape((layers[i],layers[i + 1])).astype('float32'))]
for i in range(len(layers) - 1):
    shape = [1, layers[i + 1]]   
    siz1 = 1*layers[i+1]
    _initial_values += [tf.constant(norm().rvs(siz1).reshape((1,layers[i + 1])).astype('float32'))]


# _initial_values = []
# _initial_values += [tf.constant(np.array(0.1).reshape((1,1)).astype('float32'))]
# _initial_values += [tf.constant(np.array(0.15).reshape((1,1)).astype('float32'))]
# _initial_values += [tf.constant(np.array(-0.15).reshape((1,1)).astype('float32'))]
# _initial_values += [tf.constant(np.array(-0.1).reshape((1,1)).astype('float32'))]

grads1 = m1.getgrads(_initial_values)

method = neuq.inferences.MALA(
    num_samples=1000000,
    num_burnin=100,
    time_step=0.00001,
)

# tf.Session().run(grads1)

    
model.compile(method)
samples, results = model.run()
print("Acceptance rate: %.3f \n"%(np.mean(results)))  # if HMC is used

################################# Predictions ####################################
(u_pred,) = model.predict(x_test, samples, processes=[process])
(f_pred,) = model.predict(x_test, samples, processes=[process], pde_fn=pde_fn)
############################### Postprocessing ###################################
neuq.utils.plot1d(x_u_train, u_train, x_test, u_test, u_pred[..., 0])
neuq.utils.plot1d(x_f_train, f_train, x_test, f_test, f_pred[..., 0])



### autograd test

def func1(coords):
    
    # #******** 2D Gaussian Four Mixtures #********
    q1, q2, p1, p2 = np.split(coords,4)
    sigma_inv = np.array([[1.,0.],[0.,1.]])
    term1 = 0.
    
    mu = np.array([3.,0.])
    y = np.array([q1-mu[0],q2-mu[1]])
    tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    
    mu = np.array([-3.,0.])
    y = np.array([q1-mu[0],q2-mu[1]])
    tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    
    mu = np.array([0.,3.])
    y = np.array([q1-mu[0],q2-mu[1]])
    tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    
    mu = np.array([0.,-3.])
    y = np.array([q1-mu[0],q2-mu[1]])
    tmp1 = np.array([sigma_inv[0,0]*y[0]+sigma_inv[0,1]*y[1],sigma_inv[1,0]*y[0]+sigma_inv[1,1]*y[1]]).reshape(2)
    term1 = term1 + 0.25*np.exp(-y[0]*tmp1[0] - y[1]*tmp1[1])
    
    term1 = -np.log(term1)
    term2 = p1**2/2+p2**2/2
    H = term1 + term2

    return H

def arr_to_bnntensor(array1):

    counter1 = 0
    _initial_values = []

    for i in range(len(layers) - 1):
        # shape = [layers[i], layers[i + 1]]
        siz1 = layers[i]*layers[i+1]
        _initial_values += [tf.constant(array1[counter1:(counter1+siz1)].reshape((layers[i],layers[i + 1])).astype('float32'))]
        counter1 = counter1 + siz1
    
    for i in range(len(layers) - 1):
        # shape = [1, layers[i + 1]]   
        siz1 = 1*layers[i+1]
        _initial_values += [tf.constant(array1[counter1:(counter1+siz1)].reshape((1,layers[i + 1])).astype('float32'))]
        counter1 = counter1 + siz1

    return _initial_values

def bnntensor_to_arr(tensor_list):
    req_arr = []
    counter1 = 0
    for i in range(len(layers) - 1):
        # shape = [layers[i], layers[i + 1]]
        siz1 = layers[i]*layers[i+1]
        req_arr = np.concatenate((req_arr, tensor_list[counter1].reshape(siz1)))
        counter1 = counter1 + 1
    for i in range(len(layers) - 1):
        # shape = [layers[i], layers[i + 1]]
        siz1 = 1*layers[i+1]
        req_arr = np.concatenate((req_arr, tensor_list[counter1].reshape(siz1)))
        counter1 = counter1 + 1
    return req_arr


