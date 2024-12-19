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


np.random.seed(77772)

Ra = 10 ** 4
Pr = 0.71
N_EPOCHS = 100000
DATA_WEIGHT = 1.0
RESIDUAL_WEIGHT = 1.0
lr = 1e-03
key = jr.PRNGKey(9831)
scheduler = optax.piecewise_constant_schedule(
    init_value=lr,
    boundaries_and_scales={
        int(5e4): 0.1,
    }
)
noise_uv = 0.01
noise_theta = 0.01


MODEL_FILE_NAME = "./checkpoints/RBC_1e4.eqx"
DATA_FILE_NAME = "./data/data_1e4.mat"


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
def loss_fn(network, weight_d, weight_f, xy_r, u_r, v_r, xy_d, theta_d):
    # neuman bc
    theta_x_l = jax.vmap(neuman_bc_theta, in_axes=(None, 0, 0))(network, xy_d[0][:, 0], xy_d[0][:, 1])
    theta_x_l = theta_x_l.flatten()[:, None]
    neuman_loss_1 = jnp.mean(jnp.square(theta_x_l))

    theta_x_r = jax.vmap(neuman_bc_theta, in_axes=(None, 0, 0))(network, xy_d[1][:, 0], xy_d[1][:, 1])
    theta_x_r = theta_x_r.flatten()[:, None]
    neuman_loss_2 = jnp.mean(jnp.square(theta_x_r))

    neuman_loss = neuman_loss_1 + neuman_loss_2

    # data
    theta_p_nn = jax.vmap(network, in_axes=(0, 0))(xy_d[2][:, 0], xy_d[2][:, 1])

    theta_p_nn = theta_p_nn.flatten()[:, None]
    theta_p = theta_d[2].flatten()[:, None]

    loss_p = jnp.mean(jnp.square(theta_p_nn - theta_p))
    
    # residual 
    f = jax.vmap(pde_residual, in_axes=(None, 0, 0, 0, 0))(network, xy_r[:, 0], xy_r[:, 1], u_r, v_r)
    loss_f = jnp.mean(jnp.square(f))

    # total
    total_loss = weight_d * neuman_loss + \
                 weight_f * loss_f + \
                 weight_d * loss_p

    return total_loss


if __name__ == "__main__":
    
    key, init_key = jr.split(key)
    pinn = NeuralNetwork(init_key)
    pinn = init_linear_weight(pinn, trunc_init, init_key)
    # pinn = load(MODEL_FILE_NAME, pinn)

    N_u = 1000  # Number of Initial and Boundary data points
    N_f = 1000

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
    theta_lb_x = jnp.zeros_like(x_domain)

    # Right Boundary of the Domain
    xy_rb = jnp.hstack((jnp.ones_like(y_domain), y_domain))
    theta_rb_x = jnp.zeros_like(x_domain)

    # Residual points
    idx = np.random.choice(theta.shape[0], N_f, replace=False)
    x_f = x[idx, :].flatten()[:, None]
    y_f = y[idx, :].flatten()[:, None]
    u_res = u[idx, :].flatten()[:, None]
    v_res = v[idx, :].flatten()[:, None]
    theta_res = theta[idx, :].flatten()[:, None]
    xy_f = jnp.hstack((x_f, y_f))
    # x_data = [xy_lb, xy_rb, xy_tb, xy_bb]
    # theta_data = [theta_lb_x, theta_rb_x, theta_tb, theta_bb]

    # Some data
    N_d = 100
    idx = np.random.choice(theta.shape[0], N_d, replace=False)
    x_d = x[idx, :].flatten()[:, None]
    y_d = y[idx, :].flatten()[:, None]
    xy_d = jnp.hstack([x_d, y_d])
    theta_d = theta[idx, :].flatten()[:, None]

    u_res = u_res + noise_uv * np.random.normal(size=u_res.shape)
    v_res = v_res + noise_uv * np.random.normal(size=v_res.shape)
    theta_d = theta_d + noise_theta * np.random.normal(size=theta_d.shape)

    sio.savemat(
        "./outputs/case_c.mat",
        {
            "x": np.array(x_d),
            "y": np.array(y_d),
            "theta": np.array(theta_d),
            "x_f": x_f,
            "y_f": y_f,
            "u": u_res,
            "v": v_res,
        }
    )

    xy_data = [xy_lb, xy_rb, xy_d]
    theta_data = [theta_lb_x, theta_rb_x, theta_d]

    # Test data
    x_star = jax.device_put(x_f, device=jax.devices("cpu")[0])
    y_star = jax.device_put(y_f, device=jax.devices("cpu")[0])
    theta_star = jax.device_put(theta_res, device=jax.devices("cpu")[0])
    
    optimizer = optax.adam(learning_rate=scheduler)
    opt_state = optimizer.init(eqx.filter(pinn, eqx.is_array))


    @eqx.filter_jit
    def train_step_opt(network, state):
        l, grad = eqx.filter_value_and_grad(loss_fn)(
            network, 
            DATA_WEIGHT, 
            RESIDUAL_WEIGHT,
            xy_f, 
            u_res, 
            v_res, 
            xy_data, 
            theta_data,
        )
        updates, new_state = optimizer.update(grad, state, network)
        new_network = eqx.apply_updates(network, updates)
        return new_network, new_state, l
    
    
    @eqx.filter_jit
    def compute_loss(network):
        return loss_fn(
            network, 
            DATA_WEIGHT, 
            RESIDUAL_WEIGHT, 
            xy_f, 
            u_res, 
            v_res, 
            xy_data, 
            theta_data,
        )
        

    loss_history = []
    error_l2_list = []
    t0 = time.time()
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
            t1 = time.time()
            print("Epoch:", epoch, ", loss: ", loss, ", l2: ", err_l2, ", time: ", t1-t0)
            t0 = time.time()
            # counter.set_postfix_str(f"Epoch: {epoch}, loss: {loss}, L2-Error: {err_l2}")

    # save model
    eqx.tree_serialise_leaves(MODEL_FILE_NAME, pinn)
