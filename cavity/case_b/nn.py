import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
import scipy
import optax
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.io as sio
import math


Pr = 0.71
DATA_WEIGHT = 1.0
RESIDUAL_WEIGHT = 1.0
lr = 1e-03
N_u = 1000  # Number of initial and boundary data points
N_f = 5000 # Number of residual points
initializer = jax.nn.initializers.glorot_normal()
units = 100
np.random.seed(1234)


#################### Ra = 1e4 ####################
Ra = 10 ** 4
MODEL_FILE_NAME = "./checkpoints/RBC_1e4_nn.eqx"
SAVE_FILE_NAME = "./outputs/RBC_1e4_nn.mat"
DATA_FILE_NAME = "./data/data_1e4.mat"
N_EPOCHS = 100000
scheduler = optax.piecewise_constant_schedule(
    init_value=lr,
    boundaries_and_scales={
        int(5e4): 0.1,
    }
)
##################################################

#################### Ra = 1e5 ####################
# Ra = 10 ** 5
# MODEL_FILE_NAME = "./checkpoints/RBC_1e5.eqx"
# SAVE_FILE_NAME = "./outputs/RBC_1e5.mat"
# DATA_FILE_NAME = "./data/data_1e5.mat"
# N_EPOCHS = 100000
# scheduler = optax.piecewise_constant_schedule(
#     init_value=lr,
#     boundaries_and_scales={
#         int(5e4): 0.1,
#     }
# )
##################################################

#################### Ra = 1e6 ####################
# Ra = 10 ** 6
# MODEL_FILE_NAME = "./checkpoints/RBC_1e6.eqx"
# SAVE_FILE_NAME = "./outputs/RBC_1e6.mat"
# DATA_FILE_NAME = "./data/data_1e6.mat"
# N_EPOCHS = 300000
# scheduler = optax.piecewise_constant_schedule(
#     init_value=lr,
#     boundaries_and_scales={
#         int(2e5): 0.1,
#     }
# )
##################################################


def trunc_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    out, in_ = weight.shape
    return initializer(key, shape=(out, in_))


def init_linear_weight(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [x.weight
                             for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                             if is_linear(x)]

    get_bias = lambda m: [x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
                          if is_linear(x)]

    weights = get_weights(model)
    biases = get_bias(model)

    new_biases = jax.tree_map(lambda p: 0.0 * jnp.abs(p), biases)

    new_weights = [init_fn(weight, subkey)
                   for weight, subkey in zip(weights, jax.random.split(key, len(weights)))]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_model = eqx.tree_at(get_bias, new_model, new_biases)

    biases = get_bias(new_model)
    print(f"B: {biases}")
    return new_model


class NeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, key, units=100):
        key_list = jax.random.split(key, 10)
        # These contain trainable parameters.
        self.layers = [
            eqx.nn.Linear(2, units, key=key_list[0]),
            eqx.nn.Linear(units, units, key=key_list[1]),
            eqx.nn.Linear(units, units, key=key_list[2]),
            eqx.nn.Linear(units, units, key=key_list[4]),
            eqx.nn.Linear(units, 1, key=key_list[9]),
        ]

    def __call__(self, x, y):
        xt = jnp.hstack((x, y))
        for layer in self.layers[:-1]:
            xt = jax.nn.tanh(layer(xt))
        return self.layers[-1](xt)


# To Do
@jax.jit
def loss_fn(network, xy, theta):
    theta_pred = jax.vmap(network)(xy[:, 0], xy[:, 1])
    return jnp.mean((theta_pred - theta) ** 2)


if __name__ == "__main__":
    key = jr.PRNGKey(42)
    key, init_key = jr.split(key)
    pinn = NeuralNetwork(init_key)
    pinn = init_linear_weight(pinn, trunc_init, init_key)

    data = sio.loadmat(DATA_FILE_NAME)
    x = data['x'].flatten()[:, None]
    y = data['y'].flatten()[:, None]
    theta = data['theta'].flatten()[:, None]

    # Residual condition 1
    idx = np.random.choice(theta.shape[0], N_f, replace=False)
    x_f = x[idx, :].flatten()[:, None]
    y_f = y[idx, :].flatten()[:, None]
    theta_res = theta[idx, :].flatten()[:, None]
    xy_f = jnp.hstack((x_f, y_f))

    x_star = jax.device_put(x, device=jax.devices("cpu")[0])
    y_star = jax.device_put(y, device=jax.devices("cpu")[0])
    theta_star = jax.device_put(theta, device=jax.devices("cpu")[0])

    optimizer = optax.adam(learning_rate=scheduler)
    opt_state = optimizer.init(eqx.filter(pinn, eqx.is_array))


    @eqx.filter_jit
    def train_step_opt(network, state):
        l, grad = eqx.filter_value_and_grad(loss_fn)(network, xy_f, theta_res)
        updates, new_state = optimizer.update(grad, state, network)
        new_network = eqx.apply_updates(network, updates)
        return new_network, new_state, l


    loss_history = []
    error_l2_list = []

    for epoch in range(N_EPOCHS):
        pinn, opt_state, loss = train_step_opt(pinn, opt_state)


        if epoch % 1000 == 0:
            theta_pred = jax.vmap(pinn, in_axes=(0, 0))(x_star, y_star)

            err_l2 = jnp.linalg.norm((theta_pred - theta_star), 2) / np.linalg.norm(theta_star, 2)
            print(epoch, err_l2)
    
    # save model
    eqx.tree_serialise_leaves(MODEL_FILE_NAME, pinn)
