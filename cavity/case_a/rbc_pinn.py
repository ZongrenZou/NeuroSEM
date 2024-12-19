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


Pr = 0.71
DATA_WEIGHT = 1.0
RESIDUAL_WEIGHT = 1.0
lr = 1e-03
key = jr.PRNGKey(9831)
N_u = 1000  # Number of initial and boundary data points
N_f = 10000 # Number of residual points
initializer = jax.nn.initializers.glorot_normal()
units = 100
np.random.seed(5631)


#################### Ra = 1e4 ####################
Ra = 10 ** 4
MODEL_FILE_NAME = "./checkpoints/RBC_1e4.eqx"
SAVE_FILE_NAME = "./outputs/RBC_1e4.mat"
DATA_FILE_NAME = "./case_b/data/data_1e4.mat"
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
# DATA_FILE_NAME = "../from_theta_to_uv/data/data_1e5.mat"
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
# DATA_FILE_NAME = "../from_theta_to_uv/data/data_1e6.mat"
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

    def __init__(self, key, units=50):
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


def pde_residual(network, xx, yy, uu, vv):
    theta_x = jax.grad(network, argnums=0)(xx, yy)
    theta_y = jax.grad(network, argnums=1)(xx, yy)
    theta_xx = jax.grad((jax.grad(network, argnums=0)), argnums=0)(xx, yy)
    theta_yy = jax.grad((jax.grad(network, argnums=1)), argnums=1)(xx, yy)
    f = uu * theta_x + vv * theta_y - (1 / jnp.sqrt((Ra * Pr))) * (theta_xx + theta_yy)
    return f


def neuman_bc_theta(network, x_nb, y_nb):
    theta_x = jax.grad(network, argnums=0)(x_nb, y_nb)
    return theta_x


# To Do
@jax.jit
def loss_fn(network, weight_d, weight_f, xy_r, u_r, v_r, x_d, theta_d):
    theta_x_l = jax.vmap(neuman_bc_theta, in_axes=(None, 0, 0))(network, x_d[0][:, 0], x_d[0][:, 1])
    theta_x_l = theta_x_l.flatten()[:, None]
    neuman_loss_1 = jnp.mean(jnp.square(theta_x_l))

    theta_x_r = jax.vmap(neuman_bc_theta, in_axes=(None, 0, 0))(network, x_d[1][:, 0], x_d[1][:, 1])
    theta_x_r = theta_x_r.flatten()[:, None]
    neuman_loss_2 = jnp.mean(jnp.square(theta_x_r))

    neuman_loss = neuman_loss_1 + neuman_loss_2

    theta_t = jax.vmap(network, in_axes=(0, 0))(x_d[2][:, 0], x_d[2][:, 1])
    theta_b = jax.vmap(network, in_axes=(0, 0))(x_d[3][:, 0], x_d[3][:, 1])

    theta_t = theta_t.flatten()[:, None]
    theta_b = theta_b.flatten()[:, None]

    dirichlet_loss = jnp.mean(jnp.square(theta_t - theta_d[2]))
    dirichlet_loss = dirichlet_loss + jnp.mean(jnp.square(theta_b - theta_d[3]))

    f = jax.vmap(pde_residual, in_axes=(None, 0, 0, 0, 0))(network, xy_r[:, 0], xy_r[:, 1], u_r, v_r)
    loss_f = jnp.mean(jnp.square(f))
    total_loss = weight_d * neuman_loss + weight_d * dirichlet_loss + weight_f * loss_f

    return total_loss


if __name__ == "__main__":
    key, init_key = jr.split(key)
    pinn = NeuralNetwork(init_key, units=units)
    pinn = init_linear_weight(pinn, trunc_init, init_key)
    # pinn = eqx.tree_deserialise_leaves(MODEL_FILE_NAME, pinn)

    data = sio.loadmat(DATA_FILE_NAME)
    x = data['x'].flatten()[:, None]
    y = data['y'].flatten()[:, None]
    u = data['u'].flatten()[:, None]
    v = data['v'].flatten()[:, None]
    theta = data['theta'].flatten()[:, None]
    x_domain = jnp.linspace(0, 1, N_u).flatten()[:, None]
    y_domain = jnp.linspace(0, 1, N_u).flatten()[:, None]

    # Left Boundary of the Domain
    print(f"{'#' * 10} Data on the left boundary {'#' * 10} ")
    xy_lb = jnp.hstack((jnp.zeros_like(y_domain), y_domain))
    theta_lb_x = jnp.zeros_like(x_domain)
    print(f"{'#' * 10} Left Boundary Done {'#' * 10} ")

    # Right Boundary of the Domain
    print(f"{'#' * 10} Data on the right boundary {'#' * 10} ")
    xy_rb = jnp.hstack((jnp.ones_like(y_domain), y_domain))
    theta_rb_x = jnp.zeros_like(x_domain)
    print(f"{'#' * 10} Left Boundary Done {'#' * 10} ")

    # Top Boundary of the Domain
    print(f"{'#' * 10} Data on the top boundary {'#' * 10} ")
    xy_tb = jnp.hstack((x_domain, jnp.ones_like(x_domain)))
    theta_tb = -0.5 * jnp.ones_like(x_domain)
    print(f"{'#' * 10} Top Boundary Done {'#' * 10} ")

    # Bottom Boundary of the Domain
    print(f"{'#' * 10} Data on the top boundary {'#' * 10} ")
    xy_bb = jnp.hstack((x_domain, jnp.zeros_like(x_domain)))
    theta_bb = 0.5 * jnp.ones_like(x_domain)
    print(f"{'#' * 10} Bottom Boundary Done {'#' * 10} ")

    # Residual condition 1
    idx = np.random.choice(theta.shape[0], N_f, replace=False)
    x_f = x[idx, :].flatten()[:, None]
    y_f = y[idx, :].flatten()[:, None]
    u_res = u[idx, :].flatten()[:, None]
    v_res = v[idx, :].flatten()[:, None]
    theta_res = theta[idx, :].flatten()[:, None]

    sio.savemat(
        SAVE_FILE_NAME,
        {
            "x": x_f, "y": y_f, "u": u_res, "v": v_res, 
        }
    )
    
    xy_f = jnp.hstack((x_f, y_f))
    x_data = [xy_lb, xy_rb, xy_tb, xy_bb]
    theta_data = [theta_lb_x, theta_rb_x, theta_tb, theta_bb]

    x_star = jax.device_put(x_f, device=jax.devices("cpu")[0])
    y_star = jax.device_put(y_f, device=jax.devices("cpu")[0])
    theta_star = jax.device_put(theta_res, device=jax.devices("cpu")[0])
    
    optimizer = optax.adam(learning_rate=scheduler)
    opt_state = optimizer.init(eqx.filter(pinn, eqx.is_array))


    @eqx.filter_jit
    def train_step_opt(network, state):
        l, grad = eqx.filter_value_and_grad(loss_fn)(network, DATA_WEIGHT, RESIDUAL_WEIGHT,
                                                     xy_f, u_res, v_res, x_data, theta_data)
        updates, new_state = optimizer.update(grad, state, network)
        new_network = eqx.apply_updates(network, updates)
        return new_network, new_state, l
    
    
    @eqx.filter_jit
    def compute_loss(network):
        return loss_fn(network, DATA_WEIGHT, RESIDUAL_WEIGHT, xy_f, u_res, v_res, x_data, theta_data)
        

    loss_history = []
    error_l2_list = []

    # counter = tqdm(np.arange(N_EPOCHS))
    # min_loss = 1000
    # for epoch in counter:
    for epoch in range(N_EPOCHS):
        pinn, opt_state, loss = train_step_opt(pinn, opt_state)

        # loss_history.append(loss)
        # error_l2_list.append(err_l2)

        if epoch % 1000 == 0:
            theta_pred = jax.vmap(pinn, in_axes=(0, 0))(x_star, y_star)
            theta_pred = theta_pred.reshape(-1, 1)
            theta_star = theta_star.reshape(-1, 1)

            err_l2 = jnp.linalg.norm((theta_pred - theta_star), 2) / np.linalg.norm(theta_star, 2)
            print(epoch, loss, err_l2)
            # counter.set_postfix_str(f"Epoch: {epoch}, loss: {loss}, L2-Error: {err_l2}")

            # current_loss = compute_loss(pinn)
            # if current_loss < min_loss:
                # min_loss = current_loss
                # eqx.tree_serialise_leaves(MODEL_FILE_NAME, pinn)
                # save(MODEL_FILE_NAME, pinn)

    # save model
    eqx.tree_serialise_leaves(MODEL_FILE_NAME, pinn)

    print("End main.")
