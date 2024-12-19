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


Ra = 10 ** 4
Pr = 0.71

N_EPOCHS = int(1e5) 
WEIGHT_F = 1
WEIGHT_UV = 1
WEIGHT_THETA = 1
lr = 1e-3
noise_uv = 0.0
noise_theta = 0.0

MODEL_FILE_NAME_theta = "./checkpoints/RBC_theta_1e4.eqx"
MODEL_FILE_NAME_uvp = "./checkpoints/RBC_uvp_1e4.eqx"
DATA_FILE_NAME = "../case_b/data/data_1e4.mat"

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
    print(f"B: {biases}")
    return new_model


class NeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, key, output_dim=1, units=50):
        key_list = jax.random.split(key, 10)
        # These contain trainable parameters.
        self.layers = [eqx.nn.Linear(2, units, key=key_list[0]),
                       eqx.nn.Linear(units, units, key=key_list[1]),
                       eqx.nn.Linear(units, units, key=key_list[2]),
                       eqx.nn.Linear(units, units, key=key_list[4]),
                       eqx.nn.Linear(units, output_dim, key=key_list[9])
                       ]


    def __call__(self, x, y):
        xt = jnp.hstack((x, y))
        for layer in self.layers[:-1]:
            xt = jax.nn.tanh(layer(xt))
        return self.layers[-1](xt)


def pde_residual(network1, network2, xx, yy):
    u_fn = lambda _x, _y: network1(_x, _y)[0]
    v_fn = lambda _x, _y: network1(_x, _y)[1]
    p_fn = lambda _x, _y: network1(_x, _y)[2]
    theta_fn = lambda _x, _y: network2(_x, _y)[0]


    u = u_fn(xx, yy)
    v = v_fn(xx, yy)
    theta = theta_fn(xx, yy)
    u_x, u_xx = jax.value_and_grad(jax.grad(u_fn, argnums=0), argnums=0)(xx, yy)
    u_y, u_yy = jax.value_and_grad(jax.grad(u_fn, argnums=1), argnums=1)(xx, yy)
    v_x, v_xx = jax.value_and_grad(jax.grad(v_fn, argnums=0), argnums=0)(xx, yy)
    v_y, v_yy = jax.value_and_grad(jax.grad(v_fn, argnums=1), argnums=1)(xx, yy)
    p_x = jax.grad(p_fn, argnums=0)(xx, yy)
    p_y = jax.grad(p_fn, argnums=1)(xx, yy)
    theta_x, theta_xx = jax.value_and_grad(jax.grad(theta_fn, argnums=0), argnums=0)(xx, yy)
    theta_y, theta_yy = jax.value_and_grad(jax.grad(theta_fn, argnums=1), argnums=1)(xx, yy)
    
    f1 = u_x + v_y
    f2 = u * u_x + v * u_y + p_x - jnp.sqrt(Pr / Ra) * (u_xx + u_yy)
    f3 = u * v_x + v * v_y + p_y - jnp.sqrt(Pr / Ra) * (v_xx + v_yy) - theta
    f4 = u * theta_x + v * theta_y - (1 / jnp.sqrt(Ra * Pr)) * (theta_xx + theta_yy)
    
    return f1, f2, f3, f4


def neuman_bc_theta(network, x_nb, y_nb):
    theta_fn = lambda _x, _y: network(_x, _y)[0]
    theta_x = jax.grad(theta_fn, argnums=0)(x_nb, y_nb)
    return theta_x


# To Do
@jax.jit
def loss_fn(nns, weight_f, weight_uv, weight_theta, xy_f, xy_uv, uv, xy_theta, theta):
    network1 = nns[0]
    network2 = nns[1]
    # PDE residual
    f1, f2, f3, f4 = jax.vmap(pde_residual, in_axes=(None, None, 0, 0))(network1, network2, xy_f[:, 0], xy_f[:, 1])
    loss_f = jnp.mean(f1 ** 2) + jnp.mean(f2 ** 2) + \
             jnp.mean(f3 ** 2) + jnp.mean(f4 ** 2)

    # data
    uv_pred = jax.vmap(network1)(xy_uv[:, 0], xy_uv[:, 1])[:, 0:2]
    loss_uv = jnp.mean((uv_pred - uv) ** 2)
    theta_pred = jax.vmap(network2)(xy_theta[:, 0], xy_theta[:, 1])
    loss_theta = jnp.mean((theta_pred - theta) ** 2)
    
    total_loss = weight_f * loss_f + weight_uv * loss_uv + weight_theta * loss_theta

    return total_loss


if __name__ == "__main__":
    key = jr.PRNGKey(666)
    _, init_key = jr.split(key)
    uvp_nn = NeuralNetwork(init_key, output_dim=3, units=50)
    uvp_nn = init_linear_weight(uvp_nn, trunc_init, init_key)

    key = jr.PRNGKey(99123)
    _, init_key = jr.split(key)
    theta_nn = NeuralNetwork(init_key, output_dim=1, units=50)
    theta_nn = init_linear_weight(theta_nn, trunc_init, init_key)

    data = sio.loadmat(DATA_FILE_NAME)
    x = data['x'].flatten()[:, None]
    y = data['y'].flatten()[:, None]
    u = data['u'].flatten()[:, None]
    v = data['v'].flatten()[:, None]
    theta = data['theta'].flatten()[:, None]

    ###################### Make data ######################
    np.random.seed(7611)
    # Data of uv
    N_uv = 50
    idx = (x >= 0.4) * (x <= 0.6) * (y >= 0.4) * (y <= 0.6)
    idx_uv = np.random.choice(np.sum(idx), N_uv, replace=False)
    xy_uv = np.concatenate(
        [x[idx][idx_uv][:, None], y[idx][idx_uv][:, None]], axis=-1,
    )
    uv_d = np.concatenate(
        [u[idx][idx_uv][:, None], v[idx][idx_uv][:, None]], axis=-1,
    )
    uv_d = uv_d + noise_uv * np.random.normal(size=uv_d.shape)

    # Data of theta
    N_theta = 50
    idx = (x >= 0.4) * (x <= 0.6) * (y >= 0.4) * (y <= 0.6)
    idx_theta = np.random.choice(np.sum(idx), N_theta, replace=False)
    xy_theta = np.concatenate(
        [x[idx][idx_theta][:, None], y[idx][idx_theta][:, None]], axis=-1,
    )
    theta_d = theta[idx][idx_theta][:, None]
    theta_d = theta_d + noise_theta * np.random.normal(size=theta_d.shape)

    sio.savemat(
        "./outputs/data.mat",
        {
            "xy_uv": xy_uv,
            "uv": uv_d,
            "xy_theta": xy_theta,
            "theta": theta_d,
        }
    )

    # Residual condition
    N_f = 1000
    idx = (x >= 0.4) * (x <= 0.6) * (y >= 0.4) * (y <= 0.6)
    idx_f = np.random.choice(np.sum(idx), N_f, replace=False)
    xy_f = np.concatenate(
        [x[idx][idx_f][:, None], y[idx][idx_f][:, None]], axis=-1,
    )

    # Test data
    x_star = jax.device_put(x[idx][idx_f][:, None], device=jax.devices("cpu")[0])
    y_star = jax.device_put(y[idx][idx_f][:, None], device=jax.devices("cpu")[0])
    u_star = jax.device_put(u[idx][idx_f][:, None], device=jax.devices("cpu")[0])
    v_star = jax.device_put(v[idx][idx_f][:, None], device=jax.devices("cpu")[0])
    theta_star = jax.device_put(theta[idx][idx_f][:, None], device=jax.devices("cpu")[0])

    scheduler = optax.piecewise_constant_schedule(
        init_value=lr,
        boundaries_and_scales={
            int(5e4): 0.1,
        }
    )
    optimizer = optax.adam(learning_rate=scheduler)
    opt_state = optimizer.init(
        eqx.filter([uvp_nn, theta_nn], eqx.is_array),
    )

    @eqx.filter_jit
    def train_step_opt(nns, state):
        l, grad = eqx.filter_value_and_grad(loss_fn)(
            nns,
            WEIGHT_F,
            WEIGHT_UV,
            WEIGHT_THETA, 
            xy_f,
            xy_uv,
            uv_d,
            xy_theta,
            theta_d,
        )
        updates, new_state = optimizer.update(grad, state, nns)
        new_nns = eqx.apply_updates(nns, updates)
        return new_nns, new_state, l


    # counter = tqdm(np.arange(N_EPOCHS))

    nns = [uvp_nn, theta_nn]
    # for epoch in counter:
    for epoch in range(N_EPOCHS):
        nns, opt_state, loss = train_step_opt(nns, opt_state)
        uvp_nn, theta_nn = nns
        uvp_nn = nns[0]
        theta_nn = nns[1]

        if epoch % 1000 == 0:
            uvp_pred = jax.vmap(uvp_nn, in_axes=(0, 0))(x_star, y_star)
            theta_pred = jax.vmap(theta_nn, in_axes=(0, 0))(x_star, y_star)
            u_pred = uvp_pred[:, 0:1]
            v_pred = uvp_pred[:, 1:2]

            u_star = u_star.reshape([-1, 1])
            v_star = v_star.reshape([-1, 1])
            theta_star = theta_star.reshape([-1, 1])

            u_err_l2 = jnp.linalg.norm((u_pred - u_star), 2) / np.linalg.norm(u_star, 2)
            v_err_l2 = jnp.linalg.norm((v_pred - v_star), 2) / np.linalg.norm(v_star, 2)
            theta_err_l2 = jnp.linalg.norm((theta_pred - theta_star), 2) / np.linalg.norm(theta_star, 2)
            # counter.set_postfix_str(
            #     f"Epoch: {epoch}, loss: {loss:.4f}, L2-Errors: {u_err_l2:.4f}, {v_err_l2:.4f}, {theta_err_l2:.4f}",
            # )
            print(epoch, u_err_l2, v_err_l2, theta_err_l2)
            
    # save models
    eqx.tree_serialise_leaves(MODEL_FILE_NAME_uvp, uvp_nn)
    eqx.tree_serialise_leaves(MODEL_FILE_NAME_theta, theta_nn)
    