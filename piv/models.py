import os

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax



# ====================================================================
# ========== Random sampling for residuals (Shengze's code) ==========
# ====================================================================
def LHSample(D, bounds, N):
    """
        D: Number of parameters
        bounds:  [[min_1, max_1],[min_2, max_2],[min_3, max_3]](list)
        N: Number of samples
        return: Samples
    """
    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N
    for i in range(D):
        for j in range(N):
            temp[j] = np.random.uniform(low=j * d, high=(j + 1) * d, size=1)[0]
        np.random.shuffle(temp)
        for j in range(N):
            result[j, i] = temp[j]
    # Stretching the sampling
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]
    if np.any(lower_bounds > upper_bounds):
        print('Wrong value bound')
        return None
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(result, (upper_bounds - lower_bounds), out=result),
           lower_bounds, out=result)
    return result


# ====================================================================
# ====================== Neural networks =============================
# ====================================================================
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
            eqx.nn.Linear(self.units, 3, key=key_list[-1])
        ]

    def __call__(self, x, y, t):
        out = jnp.hstack((x, y, t))
        for layer in self.layers[:-1]:
            out = jnp.tanh(layer(out))
        out = self.layers[-1](out)
        return out


# ====================================================================
# ========================= PDE residual =============================
# ====================================================================
@jax.jit
def pde_residual(network, xx, yy, tt, Re):
    u_fn = lambda _x, _y, _t: network(_x, _y, _t)[0]
    v_fn = lambda _x, _y, _t: network(_x, _y, _t)[1]
    p_fn = lambda _x, _y, _t: network(_x, _y, _t)[2]

    uv = network(xx, yy, tt)[:2]
    u, v = uv[0], uv[1]
    u_t = jax.grad(u_fn, argnums=2)(xx, yy, tt)
    v_t = jax.grad(v_fn, argnums=2)(xx, yy, tt)
    u_x, u_xx = jax.value_and_grad(jax.grad(u_fn, argnums=0), argnums=0)(xx, yy, tt)
    u_y, u_yy = jax.value_and_grad(jax.grad(u_fn, argnums=1), argnums=1)(xx, yy, tt)
    v_x, v_xx = jax.value_and_grad(jax.grad(v_fn, argnums=0), argnums=0)(xx, yy, tt)
    v_y, v_yy = jax.value_and_grad(jax.grad(v_fn, argnums=1), argnums=1)(xx, yy, tt)
    p_x = jax.grad(p_fn, argnums=0)(xx, yy, tt)
    p_y = jax.grad(p_fn, argnums=1)(xx, yy, tt)

    f1 = u_x + v_y
    f2 = u_t + (u*u_x + v*u_y) + p_x - 1/Re * (u_xx + u_yy)
    f3 = v_t + (u*v_x + v*v_y) + p_y - 1/Re * (v_xx + v_yy)

    print(f1.shape, f2.shape, Re.shape)
    return f1, f2, f3


# ====================================================================
# ======================== Loss functions =============================
# ====================================================================
@jax.jit
def loss_fn(network, weight_d, weight_f, xyt_r, xyt_d, uv_d, Re):
    uv_pred = jax.vmap(network)(xyt_d[:, 0], xyt_d[:, 1], xyt_d[:, 2])[:, 0:2]
    data_loss = jnp.mean((uv_pred[:, 0:1] - uv_d[:, 0:1]) ** 2) + jnp.mean((uv_pred[:, 1:2] - uv_d[:, 1:2]) ** 2)

    f1, f2, f3 = jax.vmap(pde_residual, in_axes=(None, 0, 0, 0, None))(network, xyt_r[:, 0], xyt_r[:, 1], xyt_r[:, 2], Re)
    pde_loss = jnp.mean(f1 ** 2) + jnp.mean(f2 ** 2) + jnp.mean(f3 ** 2)
    total_loss = weight_d * data_loss + weight_f * pde_loss
    return total_loss


@jax.jit
def loss_fn_1(network, weight_d, weight_f, xyt_d, uv_d, Re):
    uv_pred = jax.vmap(network)(xyt_d[:, 0], xyt_d[:, 1], xyt_d[:, 2])[:, 0:2]
    data_loss = jnp.mean((uv_pred[:, 0:1] - uv_d[:, 0:1]) ** 2) + jnp.mean((uv_pred[:, 1:2] - uv_d[:, 1:2]) ** 2)

    # f1, f2, f3 = jax.vmap(pde_residual, in_axes=(None, 0, 0, 0, None))(network, xyt_r[:, 0], xyt_r[:, 1], xyt_r[:, 2], Re)
    # pde_loss = jnp.mean(f1 ** 2) + jnp.mean(f2 ** 2) + jnp.mean(f3 ** 2)
    # total_loss = weight_d * data_loss + weight_f * pde_loss
    return data_loss


@jax.jit
def loss_fn_2(network, weight_d, weight_f, xyt_r, xyt_d, uv_d, Re):
    uv_pred = jax.vmap(network)(xyt_d[:, 0], xyt_d[:, 1], xyt_d[:, 2])[:, 0:2]
    data_loss = jnp.mean((uv_pred[:, 0:1] - uv_d[:, 0:1]) ** 2) + jnp.mean((uv_pred[:, 1:2] - uv_d[:, 1:2]) ** 2)

    f1, f2, f3 = jax.vmap(pde_residual, in_axes=(None, 0, 0, 0, None))(network, xyt_r[:, 0], xyt_r[:, 1], xyt_r[:, 2], Re)
    pde_loss = jnp.mean(f1 ** 2) + jnp.mean(f2 ** 2) + jnp.mean(f3 ** 2)

    total_loss = weight_d * data_loss + weight_f * pde_loss
    return total_loss, data_loss, pde_loss
