import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
import scipy
from pyDOE import lhs
import optax
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import scipy.io as sio
import math


np.random.seed(1234)


MODEL_FILE_NAME = "./checkpoints/RBC_1e4.eqx"
DATA_FILE_NAME = "./data/data_1e4.mat"
SAVE_FILE_NAME = "./outputs/theta_grid_1e4.npy"


class NeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, key, units=100):
        key_list = jax.random.split(key, 10)
        # These contain trainable parameters.
        self.layers = [eqx.nn.Linear(2, units, key=key_list[0]),
                       eqx.nn.Linear(units, units, key=key_list[1]),
                       eqx.nn.Linear(units, units, key=key_list[2]),
                       eqx.nn.Linear(units, units, key=key_list[4]),
                       eqx.nn.Linear(units, 1, key=key_list[9])
                       ]


    def __call__(self, x, y):
        xt = jnp.hstack((x, y))
        for layer in self.layers[:-1]:
            xt = jax.nn.tanh(layer(xt))
        return self.layers[-1](xt).reshape(())


if __name__ == "__main__":
    key = jr.PRNGKey(339)
    key, init_key = jr.split(key)
    pinn = NeuralNetwork(init_key)
    # pinn = init_linear_weight(pinn, trunc_init, init_key)
    pinn = eqx.tree_deserialise_leaves(MODEL_FILE_NAME, pinn)
    
    N_u = 1000  # Number of Initial and Boundary data points
    N_f = 1000

    data = sio.loadmat(DATA_FILE_NAME)
    x = data['x'].flatten()[:, None]
    y = data['y'].flatten()[:, None]
    u = data['u'].flatten()[:, None]
    v = data['v'].flatten()[:, None]
    theta = data['theta'].flatten()[:, None]

    # Residual condition 1
    idx = np.random.choice(theta.shape[0], N_f, replace=False)
    x_f = x[idx, :].flatten()[:, None]
    y_f = y[idx, :].flatten()[:, None]

    idx = np.random.choice(theta.shape[0], theta.shape[0], replace=False)[:100000]
    x_star = x[idx, :].flatten()[:, None]
    y_star = y[idx, :].flatten()[:, None]
    theta_star = theta[idx, :].flatten()[:, None]
    # x_star = x.flatten()[:, None]
    # y_star = y.flatten()[:, None]
    # theta_star = theta.flatten()[:, None]

    # Some data
    N_d = 100
    idx = np.random.choice(theta.shape[0], N_d, replace=False)
    x_d = x[idx, :].flatten()[:, None]
    y_d = y[idx, :].flatten()[:, None]
    xy_d = jnp.hstack([x_d, y_d])
    theta_d = theta[idx, :].flatten()[:, None]

    # Test data
    x_star = jax.device_put(x_star, device=jax.devices("cpu")[0])
    y_star = jax.device_put(y_star, device=jax.devices("cpu")[0])
    theta_star = jax.device_put(theta_star, device=jax.devices("cpu")[0])

    theta_pred = jax.vmap(pinn, in_axes=(0, 0))(x_star, y_star)
    theta_pred = theta_pred.reshape(-1, 1)
    theta_star = theta_star.reshape(-1, 1)

    err_l2 = jnp.linalg.norm((theta_pred - theta_star), 2) / np.linalg.norm(theta_star, 2)
    print(err_l2)

    plt.figure()
    plt.plot(x_f, y_f, "x")
    plt.show()

    x_plot = np.linspace(0, 1, 100)
    y_plot = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(x_plot, y_plot)
    XX = XX.flatten()[:, None]
    YY = YY.flatten()[:, None]
    jax.device_put(XX, device=jax.devices("cpu")[0])
    jax.device_put(YY, device=jax.devices("cpu")[0])
    theta_grid = jax.vmap(pinn, in_axes=(0, 0))(XX, YY)
    np.save(SAVE_FILE_NAME, theta_grid, allow_pickle=True)


