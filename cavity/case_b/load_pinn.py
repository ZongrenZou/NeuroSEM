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


MODEL_FILE_NAME = "./checkpoints/RBC_1e4.eqx"
DATA_FILE_NAME = "./data/data_1e4.mat"
SAVE_FILE_NAME = "./outputs/uv_grid_1e4.mat"


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


if __name__ == "__main__":
    key = jr.PRNGKey(42)
    key, init_key = jr.split(key)
    pinn = NeuralNetwork(init_key)
    pinn = eqx.tree_deserialise_leaves(MODEL_FILE_NAME, pinn)

    data = sio.loadmat(DATA_FILE_NAME)
    x = data['x'].flatten()[:, None]
    y = data['y'].flatten()[:, None]
    u = data['u'].flatten()
    v = data['v'].flatten()
   
    # Prediction
    uvp_pred = jax.vmap(pinn, in_axes=(0, 0))(x, y)
    u_pred = uvp_pred[:, 0]
    v_pred = uvp_pred[:, 1]
    
    # Error
    u_err_l2 = jnp.linalg.norm((u_pred - u), 2) / np.linalg.norm(u, 2)
    v_err_l2 = jnp.linalg.norm((v_pred - v), 2) / np.linalg.norm(v, 2)
    print("L2 relative error of u:", u_err_l2)
    print("L2 relative error of v:", v_err_l2)

    # For plots
    x_plot = np.linspace(0, 1, 100)
    y_plot = np.linspace(0, 1, 100)
    XX, YY = np.meshgrid(x_plot, y_plot)
    XX = XX.flatten()[:, None]
    YY = YY.flatten()[:, None]
    jax.device_put(XX, device=jax.devices("cpu")[0])
    jax.device_put(YY, device=jax.devices("cpu")[0])
    uvp_grid = jax.vmap(pinn, in_axes=(0, 0))(XX, YY)
    u_grid = uvp_grid[:, 0]
    v_grid = uvp_grid[:, 1]
    
    sio.savemat(
        SAVE_FILE_NAME,
        {
            "u_grid": u_grid, "v_grid": v_grid,
        }
    )
    