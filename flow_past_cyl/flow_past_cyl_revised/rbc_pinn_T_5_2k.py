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
import time


np.random.seed(1234)

Pe = 71.0


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
            eqx.nn.Linear(self.units, 1, key=key_list[9])
        ]

    def __call__(self, x, y, t):
        xt = jnp.hstack((x, y, t))
        for layer in self.layers[:-1]:
            xt = jnp.tanh(layer(xt))
        return self.layers[-1](xt).reshape(())

@jax.jit
def pde_residual(network, xx, yy, tt, uu, vv):
    theta_x = jax.grad(network, argnums=0)(xx, yy, tt)
    theta_y = jax.grad(network, argnums=1)(xx, yy, tt)
    theta_t = jax.grad(network, argnums=2)(xx, yy, tt)
    theta_xx = jax.grad((jax.grad(network, argnums=0)), argnums=0)(xx, yy, tt)
    theta_yy = jax.grad((jax.grad(network, argnums=1)), argnums=1)(xx, yy, tt)
    f = theta_t + uu * theta_x + vv * theta_y - (1.0 / Pe) * (theta_xx + theta_yy)
    return f


# @jax.jit
def loss_fn(network, weight_d, weight_f, ds_f, ds_bc, ds_ic, ds_cyl):
    print("Number of data for initial condition: ", ds_ic["xyt_ic"].shape, flush=True)
    print("Number of data for boundary condition: ", ds_bc["xyt_bc"].shape, flush=True)
    print("Number of data for the cylinder: ", ds_cyl["xyt_cyl"].shape, flush=True)
    print("Number of data for residual: ", ds_f["xyt_f"].shape, flush=True)
    print(ds_bc["theta_bc"].shape)
    print(ds_cyl["xyt_cyl"].shape)

    theta_ic_nn = jax.vmap(network, in_axes=(0, 0, 0))(ds_ic["xyt_ic"][:, 0], ds_ic["xyt_ic"][:, 1],
                                                       ds_ic["xyt_ic"][:, 2])

    theta_bc_nn = jax.vmap(network, in_axes=(0, 0, 0))(ds_bc["xyt_bc"][:, 0], ds_bc["xyt_bc"][:, 1],
                                                       ds_bc["xyt_bc"][:, 2])

    theta_cyl_nn = jax.vmap(network, in_axes=(0, 0, 0))(ds_cyl["xyt_cyl"][:, 0], ds_cyl["xyt_cyl"][:, 1],
                                                        ds_cyl["xyt_cyl"][:, 2])

    theta_ic_nn = theta_ic_nn.flatten()[:, None]
    theta_bc_nn = theta_bc_nn.flatten()[:, None]
    theta_cyl_nn = theta_cyl_nn.flatten()[:, None]

    theta_ic_d = ds_ic["theta_ic"]
    theta_bc_d = ds_bc["theta_bc"]
    theta_cyl_d = ds_cyl["theta_cyl"]
    
    loss_ic = jnp.mean(jnp.square(theta_ic_nn - theta_ic_d))
    loss_bc = jnp.mean(jnp.square(theta_bc_nn - theta_bc_d)) + jnp.mean(jnp.square(theta_cyl_nn - theta_cyl_d))

    loss_f = jax.vmap(pde_residual, in_axes=(None, 0, 0, 0, 0, 0))(
        network, 
        ds_f["xyt_f"][:, 0],
        ds_f["xyt_f"][:, 1], 
        ds_f["xyt_f"][:, 2],
        ds_f["u_f"][:, 0], 
        ds_f["v_f"][:, 0],
    )
    
    loss_f = jnp.mean(jnp.square(loss_f))

    total_loss = weight_d * loss_ic + weight_d * loss_bc + loss_f

    return total_loss


if __name__ == "__main__":
    N_T = 21 # 0.25 seconds per snapshot

    #################### Load data for initial condition ####################
    # Load Data for Cylinder
    data = sio.loadmat('./data/data.mat')
    x_ic = data['x_ic'].flatten()[:, None]
    y_ic = data['y_ic'].flatten()[:, None]
    t_ic = np.zeros_like(x_ic)
    theta_ic = data['theta_ic'].flatten()[:, None]
    xyt_ic = jnp.hstack((x_ic, y_ic, t_ic))
    print(theta_ic.shape, xyt_ic.shape)
    data_ic = {"xyt_ic": xyt_ic, "theta_ic": theta_ic}

    #################### Load data for boundary condition ####################
    # Load Data for Cylinder
    x_bc = data['x_bc'].flatten()[:, None]
    y_bc = data['y_bc'].flatten()[:, None]
    t_bc = data["Time"].flatten()[:, None]
    t_bc = t_bc[:N_T, :]

    xy_bc = jnp.hstack((x_bc, y_bc))
    N_bc = xy_bc.shape[0]
    xy_bc_tile = jnp.tile(xy_bc, (N_T, 1))
    t_bc_tile = jnp.tile(t_bc.T, (N_bc, 1))
    t_bc_tile = t_bc_tile.swapaxes(0, 1)
    t_bc_tile = t_bc_tile.reshape(N_bc * N_T, 1)

    xyt_bc = jnp.hstack((xy_bc_tile, t_bc_tile))
    theta_bc = data['theta_bc'].flatten()[:1016*N_T, None]
    data_bc = {"xyt_bc": xyt_bc, "theta_bc": theta_bc}
   
    # cylinder
    xyt_cyl = data["xyt_cyl"]
    theta_cyl = data["theta_cyl"]
    data_cyl = {
        "xyt_cyl": xyt_cyl[:360*N_T, :], 
        "theta_cyl": theta_cyl[:360*N_T, :],
    }

    #################### Load data for residual ####################
    # subsampling residual points
    # the residual points are the same cross different snapshots
    K = 2000
    N = data["x"].flatten().shape[0] # 28748

    x_data = data['x'].flatten()
    y_data = data['y'].flatten()
    t = data['Time'].flatten()
    u_data = data["u"]
    v_data = data["v"]
    theta_data = data["theta"]
    
    u_f = []
    v_f = []
    x_f = []
    y_f = []
    t_f = []
    theta_ref = []
    # Set the random seed is set to 3555
    # np.random.seed(3555)
    np.random.seed(87551)
    # np.random.seed(9999)
    for i in range(t[:N_T].shape[0]):
        idx = np.random.choice(N, N, replace=False)[:K]
        u_f += [u_data[i, idx].reshape([-1, 1])]
        v_f += [v_data[i, idx].reshape([-1, 1])]
        x_f += [x_data[idx].reshape([-1, 1])]
        y_f += [y_data[idx].reshape([-1, 1])]
        t_f += [t[i] * np.ones_like(x_f[-1])]
        theta_ref += [theta_data[i, idx].reshape([-1, 1])]
    # stack them
    u_f = np.concatenate(u_f, axis=-1)
    v_f = np.concatenate(v_f, axis=-1)
    x_f = np.concatenate(x_f, axis=-1)
    y_f = np.concatenate(y_f, axis=-1)
    t_f = np.concatenate(t_f, axis=-1)
    theta_ref = np.concatenate(theta_ref, axis=-1)

    # sio.savemat(
    #     "./data/train.mat",
    #     {
    #         "x_f": x_f, "y_f": y_f, "t_f": t_f,
    #         "u_f": u_f, "v_f": v_f,
    #         "theta_ref": theta_ref,
    #     }
    # )

    u_f = u_f.flatten().reshape([-1, 1])
    v_f = v_f.flatten().reshape([-1, 1])
    x_f = x_f.flatten().reshape([-1, 1])
    y_f = y_f.flatten().reshape([-1, 1])
    t_f = t_f.flatten().reshape([-1, 1])
    xyt_f = np.concatenate([x_f, y_f, t_f], axis=-1)
    print(np.max(np.abs(u_f)), np.max(np.abs(v_f)))
    print(np.mean(np.abs(u_f)), np.mean(np.abs(v_f)))
    noise_u = 0.00
    noise_v = 0.00
    # noise_u = 0.01
    # noise_v = 0.001
    u_f = u_f + noise_u * np.random.normal(u_f.shape)
    v_f = v_f + noise_v * np.random.normal(v_f.shape)
    print("Hey hey")
    print(u_f.shape, v_f.shape, x_f.shape, y_f.shape, t_f.shape, xyt_f.shape)  
    noise_u 
    data_f = {
        "xyt_f": xyt_f, 
        "u_f": u_f, 
        "v_f": v_f,
    }

    #################### Load data for prediction ####################
    N_T = 401
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
    N_data = x_ic.shape[0]
    start_id_test = 20 * N_data
    end_id_test = 21 * N_data
    x_star = jax.device_put(xyt_data[start_id_test:end_id_test, 0], device=jax.devices("cpu")[0])
    y_star = jax.device_put(xyt_data[start_id_test:end_id_test, 1], device=jax.devices("cpu")[0])
    t_star = jax.device_put(xyt_data[start_id_test:end_id_test, 2], device=jax.devices("cpu")[0])
    print(t_star)
    theta_star = jax.device_put(theta_data[start_id_test:end_id_test, :], device=jax.devices("cpu")[0])

    #################### Build the model and the training ####################
    N_EPOCHS = int(1e6)
    DATA_WEIGHT = 1.0
    RESIDUAL_WEIGHT = 1.0
    key = jr.PRNGKey(5566)
    # key = jr.PRNGKey(8791)
    key, init_key = jr.split(key)
    pinn = NeuralNetwork(init_key)
    pinn = init_linear_weight(pinn, trunc_init, init_key)

    schedule = optax.piecewise_constant_schedule(
        init_value=1e-3,
        boundaries_and_scales={
            int(5e5): 0.5,
            int(7e5): 0.2,
            int(9e5): 0.1,
        }
    )
    optimizer = optax.adamw(learning_rate=schedule, )
    opt_state = optimizer.init(eqx.filter(pinn, eqx.is_array))

    @eqx.filter_jit
    def train_step_opt(network, params_state):
        l, grad = eqx.filter_value_and_grad(loss_fn)(network, DATA_WEIGHT, RESIDUAL_WEIGHT,
                                                     data_f, data_bc, data_ic, data_cyl)
        updates, new_state = optimizer.update(grad, params_state, network)
        new_network = eqx.apply_updates(network, updates)
        return new_network, new_state, l


    loss_history = []
    error_l2_list = []

    # counter = tqdm(np.arange(N_EPOCHS))
    t0 = time.time()
    _t0 = time.time()
    counter = range(N_EPOCHS)
    for epoch in counter:
        pinn, opt_state, loss = train_step_opt(pinn, opt_state)
        # print(f"Epoch: {epoch}, Err_l2: {loss}")
        if epoch % 1000 == 0:
            theta_pred = jax.vmap(pinn, in_axes=(0, 0, 0))(x_star, y_star, t_star)
            theta_pred = theta_pred.reshape(-1, 1)
            theta_star = theta_star.reshape(-1, 1)
            
            err_l2 = jnp.linalg.norm((theta_pred - theta_star), 2) / jnp.linalg.norm(theta_star, 2)
            # counter.set_postfix_str(f"Epoch: {epoch}, loss: {loss}, L2-Error: {err_l2}")
            _t1 = time.time()
            print(epoch, loss, err_l2, _t1 - _t0, flush=True)
            _t0 = time.time()
            loss_history.append(loss)
            error_l2_list.append(err_l2)
    t1 = time.time()
    print("Total training time: ", t1 - t0, flush=True)
    # save model
    SAVE_MODEL_NAME = "./checkpoints/RBC_T_5_2k_4x100.eqx"
    eqx.tree_serialise_leaves(SAVE_MODEL_NAME, pinn)
    # np.save("./outputs/loss_"+str(N_snapshots), loss_history)
    # np.save("./outputs/error_"+str(N_snapshots), error_l2_list)
    print("End main.", flush=True)
