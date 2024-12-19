import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import griddata
from matplotlib import colormaps


###################################################################
initializer = jax.nn.initializers.glorot_normal()


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
    return new_model


class NeuralNetwork(eqx.Module):
    layers: list
    units: int = 100
    #num_ada_w: int
    #ada_weights: jnp.array
    num_keys: int = 15

    def __init__(self, key_model):
        key_list = jax.random.split(key_model, self.num_keys)
        # These contain trainable parameters.
        self.layers = [
            eqx.nn.Linear(3, self.units, key=key_list[0]),
            eqx.nn.Linear(self.units, self.units, key=key_list[1]),
            eqx.nn.Linear(self.units, self.units, key=key_list[2]),
            eqx.nn.Linear(self.units, self.units, key=key_list[3]),
            eqx.nn.Linear(self.units, self.units, key=key_list[4]),
            eqx.nn.Linear(self.units, 1, key=key_list[9])
        ]

    def __call__(self, x, y, t):
        xt = jnp.hstack((x, y, t))
        for layer in self.layers[:-1]:
            xt = jnp.tanh(layer(xt))
            # xt = jnp.sin(layer(xt))
        return self.layers[-1](xt).reshape(())


if __name__ == "__main__":
    N_T = 81

    #################### Load model ####################
    key = jr.PRNGKey(8791)
    key, init_key = jr.split(key)
    pinn = NeuralNetwork(init_key)
    # pinn = eqx.tree_deserialise_leaves("./checkpoints/RBC_new_100.eqx", pinn)
    # pinn = eqx.tree_deserialise_leaves("./checkpoints/RBC_T_5_5k_100_deep.eqx", pinn)
    # pinn = eqx.tree_deserialise_leaves("./checkpoints/RBC_T_5_5k_100_noisy_1.eqx", pinn)
    # pinn = eqx.tree_deserialise_leaves("./checkpoints/RBC_T_5_new_case.eqx", pinn)
    # pinn = eqx.tree_deserialise_leaves("./checkpoints/RBC_T_5_2k_100.eqx", pinn)
    # pinn = eqx.tree_deserialise_leaves("./checkpoints/RBC_T_5_2k_4x100.eqx", pinn)
    # pinn = eqx.tree_deserialise_leaves("./checkpoints/RBC_T_5_new_case_5kuv_1ktheta.eqx", pinn)
    pinn = eqx.tree_deserialise_leaves("./checkpoints/RBC_T_20_3k_5x100_new_data.eqx", pinn)
    
    #################### Load data ####################
    data = sio.loadmat('./data/data_new.mat')
    # residual = sio.loadmat("./data/train.mat")
    # x_f = residual["x_f"]
    # y_f = residual["y_f"]
    # t_f = residual["t_f"]
    # theta_ref = residual["theta_ref"]

    x = data['x'].flatten()[:, None]
    y = data['y'].flatten()[:, None]
    t = data['Time'].flatten()[:, None]
    u = data['u'].flatten()[:, None]
    v = data['v'].flatten()[:, None]
    theta_data = data['theta'].flatten()[:, None]
    xy_data = jnp.hstack((x, y))
    N_data = xy_data.shape[0]
    xy_data_tile = jnp.tile(xy_data, (N_T, 1))
    t_data_tile = jnp.tile(t.T, (N_data, 1))
    t_data_tile = t_data_tile.swapaxes(0, 1)
    t_data_tile = t_data_tile.reshape(N_data * N_T, 1)
    xyt_data = jnp.hstack((xy_data_tile, t_data_tile))
    N_data = data["x_ic"].flatten().shape[0]
    start_id_test = 0 * N_data
    end_id_test = 81 * N_data

    x_star = jnp.array(xyt_data[start_id_test:end_id_test, 0])
    y_star = jnp.array(xyt_data[start_id_test:end_id_test, 1])
    t_star = jnp.array(xyt_data[start_id_test:end_id_test, 2])
    print(np.max(t_star))
    print("*******")
    print(x_star.shape)
    theta_star = jnp.array(
        theta_data[start_id_test:end_id_test, :]
    )

    #################### Make predictions and test ####################
    theta_pred = jax.vmap(pinn, in_axes=(0, 0, 0))(x_star, y_star, t_star)
    theta_pred = theta_pred.reshape(-1, 1)
    theta_star = theta_star.reshape(-1, 1)
    print(t_star)
    print("L2 relative error: ", np.linalg.norm(theta_star - theta_pred, 2) / np.linalg.norm(theta_star, 2))

    #################### Make plots ####################
    _x = np.linspace(-4, 12, 501)
    _y = np.linspace(-4, 4, 251)
    x_grid, y_grid = np.meshgrid(_x, _y)
    start_id_test = 20 * N_data
    end_id_test = 21 * N_data
    x_star = jnp.array(xyt_data[start_id_test:end_id_test, 0])
    y_star = jnp.array(xyt_data[start_id_test:end_id_test, 1])
    t_star = jnp.array(xyt_data[start_id_test:end_id_test, 2])
    theta_star = jnp.array(theta_data[start_id_test:end_id_test, :])
    theta_pred = jax.vmap(pinn, in_axes=(0, 0, 0))(x_star, y_star, t_star)
    theta_pred = theta_pred.reshape(-1, 1)
    theta_star = theta_star.reshape(-1, 1)

    theta_sem = griddata(
        np.hstack([x, y]), 
        theta_star, 
        (x_grid, y_grid), 
        method='cubic',
    )
    plt.figure()
    plt.pcolormesh(x_grid, y_grid, theta_sem[..., 0], cmap=colormaps["jet"])
    plt.title("SEM")
    plt.savefig("./outputs/sem.png")

    #### PINN ####
    theta_pinn = griddata(
        np.hstack([x, y]), 
        theta_pred, 
        (x_grid, y_grid), 
        method='cubic',
    )
    plt.figure()
    plt.pcolormesh(x_grid, y_grid, theta_pinn[..., 0], cmap=colormaps["jet"])
    plt.title("PINN+SEM")
    plt.savefig("./outputs/pinn.png")

    #### Error ####
    plt.figure()
    plt.pcolormesh(x_grid, y_grid, np.abs(theta_sem-theta_pinn)[..., 0], cmap=colormaps["jet"])
    plt.colorbar()
    plt.title("Absolute error")
    plt.savefig("./outputs/error.png")

    #################### How error develops against time ####################
    t = data['Time'].flatten()
    # print(t)
    theta_fn = jax.vmap(pinn, in_axes=(0, 0, 0))
    # errs = []
    # l2s = []
    # ts = []

    # for i in range(t.shape[0]):
    #     x_star = x_f[:, i]
    #     y_star = y_f[:, i]
    #     t_star = t_f[:, i]
    #     # print(np.max(t_star), np.min(t_star))
    #     theta_star = theta_ref[:, i].reshape([-1, 1])
    #     theta_pred = theta_fn(x_star, y_star, t_star).reshape([-1, 1])
    #     errs += [np.mean((theta_star - theta_pred) ** 2)]
    #     ts += [t[i]]
    #     l2s += [np.linalg.norm(theta_star - theta_pred, 2) / np.linalg.norm(theta_star, 2)]
    #     print(errs[-1])

    # plt.figure()
    # plt.semilogy(ts, errs, "-*")
    # # plt.ylim([5e-7, 2e-5])
    # plt.xlabel("$t$")
    # plt.ylabel("MSE")
    # plt.savefig("./outputs/errs_vs_ts_40.png")

    # plt.figure()
    # plt.semilogy(ts, errs, "-*")
    # # plt.ylim([5e-7, 2e-5])
    # plt.xlabel("$t$")
    # plt.ylabel("MSE")
    # plt.savefig("./outputs/l2s_vs_ts_40.png")

    errs = []
    l2s = []
    ts = []
    for i in range(t.shape[0]):
        start_id_test = i * N_data
        end_id_test = (i + 1) * N_data
        x_star = jnp.array(xyt_data[i * N_data: (i + 1) * N_data, 0])
        y_star = jnp.array(xyt_data[i * N_data: (i + 1) * N_data, 1])
        t_star = jnp.array(xyt_data[i * N_data: (i + 1) * N_data, 2])
        theta_star = jnp.array(theta_data[i * N_data: (i + 1) * N_data, :]).reshape([-1, 1])
        theta_pred = theta_fn(x_star, y_star, t_star).reshape([-1, 1])

        l2s += [np.linalg.norm(theta_star - theta_pred, 2) / np.linalg.norm(theta_star, 2)]
        errs += [np.mean((theta_star - theta_pred) ** 2)]
        ts += [t[i]]
        # print(errs[-1])

    plt.figure()
    plt.semilogy(ts, errs, "-*")
    # plt.ylim([5e-7, 2e-5])
    plt.xlabel("$t$")
    plt.ylabel("MSE")
    plt.savefig("./outputs/mses.png")

    plt.figure()
    plt.semilogy(ts, l2s, "-*")
    # plt.ylim([5e-7, 2e-5])
    plt.xlabel("$t$")
    plt.ylabel("L2")
    plt.savefig("./outputs/l2s.png")
