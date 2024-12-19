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
N_f = 100 # Number of residual points
initializer = jax.nn.initializers.glorot_normal()
units = 100
np.random.seed(1234)


#################### Ra = 1e4 ####################
Ra = 10 ** 4
MODEL_FILE_NAME = "./checkpoints/RBC_1e4_2.eqx"
SAVE_FILE_NAME = "./outputs/RBC_1e4_2.mat"
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
            eqx.nn.Linear(units, 3, key=key_list[9]),
        ]

    def __call__(self, x, y):
        xt = jnp.hstack((x, y))
        for layer in self.layers[:-1]:
            xt = jax.nn.tanh(layer(xt))
        return self.layers[-1](xt)


def pde_residual(network, xx, yy, theta):
    u_fn = lambda _x, _y: network(_x, _y)[0]
    v_fn = lambda _x, _y: network(_x, _y)[1]
    p_fn = lambda _x, _y: network(_x, _y)[2]

    uv = network(xx, yy)[:2]
    u, v = uv[0], uv[1]
    u_x, u_xx = jax.value_and_grad(jax.grad(u_fn, argnums=0), argnums=0)(xx, yy)
    u_y, u_yy = jax.value_and_grad(jax.grad(u_fn, argnums=1), argnums=1)(xx, yy)
    v_x, v_xx = jax.value_and_grad(jax.grad(v_fn, argnums=0), argnums=0)(xx, yy)
    v_y, v_yy = jax.value_and_grad(jax.grad(v_fn, argnums=1), argnums=1)(xx, yy)
    p_x = jax.grad(p_fn, argnums=0)(xx, yy)
    p_y = jax.grad(p_fn, argnums=1)(xx, yy)
    
    f1 = u_x + v_y
    f2 = u * u_x + v * u_y + p_x - jnp.sqrt(Pr / Ra) * (u_xx + u_yy)
    f3 = u * v_x + v * v_y + p_y - jnp.sqrt(Pr / Ra) * (v_xx + v_yy) - theta

    return f1, f2, f3


# To Do
@jax.jit
def loss_fn(network, weight_d, weight_f, xy_r, theta_r, xy_d, uv_d):
    uv_pred = jax.vmap(network)(xy_d[:, 0], xy_d[:, 1])[:, 0:2]
    dirichlet_loss = jnp.mean(jnp.square(uv_pred - uv_d))

    f1, f2, f3 = jax.vmap(pde_residual, in_axes=(None, 0, 0, 0))(network, xy_r[:, 0], xy_r[:, 1], theta_r[:, 0])
    loss_f = jnp.mean(jnp.square(f1)) + jnp.mean(jnp.square(f2)) + jnp.mean(jnp.square(f3))
    total_loss = weight_d * dirichlet_loss + weight_f * loss_f

    return total_loss


if __name__ == "__main__":
    key = jr.PRNGKey(42)
    key, init_key = jr.split(key)
    pinn = NeuralNetwork(init_key)
    pinn = init_linear_weight(pinn, trunc_init, init_key)

    data = sio.loadmat(DATA_FILE_NAME)
    x = data['x'].flatten()[:, None]
    y = data['y'].flatten()[:, None]
    u = data['u'].flatten()[:, None]
    v = data['v'].flatten()[:, None]
    theta = data['theta'].flatten()[:, None]
    x_domain = jnp.linspace(0, 1, N_u).flatten()[:, None]
    y_domain = jnp.linspace(0, 1, N_u).flatten()[:, None]

    # Left Boundary of the Domain
    xy_lb = jnp.hstack((jnp.zeros_like(y_domain), y_domain))
    # Right Boundary of the Domain
    xy_rb = jnp.hstack((jnp.ones_like(y_domain), y_domain))
    # Top Boundary of the Domain
    xy_tb = jnp.hstack((x_domain, jnp.ones_like(x_domain)))
    # Bottom Boundary of the Domain
    xy_bb = jnp.hstack((x_domain, jnp.zeros_like(x_domain)))
    xy_data = jnp.concatenate(
        [xy_lb, xy_rb, xy_tb, xy_bb], axis=0,
    )
    uv_data = jnp.zeros([xy_data.shape[0], 2])

    # Residual condition 1
    idx = np.random.choice(theta.shape[0], N_f, replace=False)
    x_f = x[idx, :].flatten()[:, None]
    y_f = y[idx, :].flatten()[:, None]
    u_res = u[idx, :].flatten()[:, None]
    v_res = v[idx, :].flatten()[:, None]
    theta_res = theta[idx, :].flatten()[:, None]
    xy_f = jnp.hstack((x_f, y_f))

    x_star = jax.device_put(x_f, device=jax.devices("cpu")[0])
    y_star = jax.device_put(y_f, device=jax.devices("cpu")[0])
    u_star = jax.device_put(u_res, device=jax.devices("cpu")[0])
    v_star = jax.device_put(v_res, device=jax.devices("cpu")[0])

    optimizer = optax.adam(learning_rate=scheduler)
    opt_state = optimizer.init(eqx.filter(pinn, eqx.is_array))


    @eqx.filter_jit
    def train_step_opt(network, state):
        l, grad = eqx.filter_value_and_grad(loss_fn)(network, DATA_WEIGHT, RESIDUAL_WEIGHT,
                                                     xy_f, theta_res, xy_data, uv_data)
        updates, new_state = optimizer.update(grad, state, network)
        new_network = eqx.apply_updates(network, updates)
        return new_network, new_state, l


    loss_history = []
    error_l2_list = []

    # counter = tqdm(np.arange(N_EPOCHS))
    # for epoch in counter:
    for epoch in range(N_EPOCHS):
        pinn, opt_state, loss = train_step_opt(pinn, opt_state)

        # loss_history.append(loss)
        # error_l2_list.append(err_l2)

        if epoch % 1000 == 0:
            uvp_pred = jax.vmap(pinn, in_axes=(0, 0))(x_star, y_star)
            u_pred = uvp_pred[:, 0:1]
            v_pred = uvp_pred[:, 1:2]
            u_star = u_star.reshape([-1, 1])
            v_star = v_star.reshape([-1, 1])

            u_err_l2 = jnp.linalg.norm((u_pred - u_star), 2) / np.linalg.norm(u_star, 2)
            v_err_l2 = jnp.linalg.norm((v_pred - v_star), 2) / np.linalg.norm(v_star, 2)
            print(epoch, loss, u_err_l2, v_err_l2)
            # u_err_l2 = np.round(u_err_l2, 4)
            # v_err_l2 = np.round(v_err_l2, 5)
            # counter.set_postfix_str(f"Epoch: {epoch}, loss: {loss}, L2-Error of u, v: {u_err_l2}, {v_err_l2}")
    
    # save model
    eqx.tree_serialise_leaves(MODEL_FILE_NAME, pinn)
