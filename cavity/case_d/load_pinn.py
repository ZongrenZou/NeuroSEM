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
import scipy.io as sio
import math
from scipy.interpolate import griddata
from matplotlib import cm, colors


np.random.seed(1234)


MODEL_FILE_NAME = "./checkpoints/RBC_1e4.eqx"
DATA_FILE_NAME = "../case_b/data/data_1e4.mat"
SAVE_FILE_NAME = "./outputs/theta_grid_1e4.npy"


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


if __name__ == "__main__":
    units = 50
    key = jr.PRNGKey(666)
    _, init_key = jr.split(key)
    uvp_nn = NeuralNetwork(init_key, output_dim=3, units=units)
    uvp_nn = eqx.tree_deserialise_leaves("./checkpoints/RBC_uvp_1e4.eqx", uvp_nn)
    theta_nn = NeuralNetwork(init_key, output_dim=1, units=units)
    theta_nn = eqx.tree_deserialise_leaves("./checkpoints/RBC_theta_1e4.eqx", theta_nn)
    
    data = sio.loadmat(DATA_FILE_NAME)
    x = data['x'].reshape([-1, 1])
    y = data['y'].reshape([-1, 1])
    u = data['u'].reshape([-1, 1])
    v = data['v'].reshape([-1, 1])
    theta = data['theta'].reshape([-1, 1])
    idx = (x >= 0.4) * (x <= 0.6) * (y >= 0.4) * (y <= 0.6)
    x = x[idx][:, None]
    y = y[idx][:, None]
    u = u[idx][:, None]
    v = v[idx][:, None]
    theta = theta[idx][:, None]

    # Test data
    x_star = jax.device_put(x, device=jax.devices("cpu")[0])
    y_star = jax.device_put(y, device=jax.devices("cpu")[0])
    u_star = jax.device_put(u, device=jax.devices("cpu")[0])
    v_star = jax.device_put(v, device=jax.devices("cpu")[0])
    theta_star = jax.device_put(theta, device=jax.devices("cpu")[0])


    theta_pred = jax.vmap(theta_nn, in_axes=(0, 0))(x_star, y_star)
    uv_pred = jax.vmap(uvp_nn, in_axes=(0, 0))(x_star, y_star)[:, 0:2]

    err_theta = jnp.linalg.norm((theta_pred - theta_star), 2) / jnp.linalg.norm(theta_star, 2)
    err_u = jnp.linalg.norm((uv_pred[:, 0:1] - u_star), 2) / jnp.linalg.norm(u_star, 2)
    err_v = jnp.linalg.norm((uv_pred[:, 1:2] - v_star), 2) / jnp.linalg.norm(v_star, 2)

    print("L2 error of theta:", err_theta)
    print("L2 error of u:", err_u)
    print("L2 error of v:", err_v)


    ############### For plot ###############
    N = 100
    xi = np.linspace(0.4, 0.6, N)
    yi = np.linspace(0.4, 0.6, N)
    xx, yy = np.meshgrid(xi, yi)
    xx = jnp.array(xx.reshape([-1, 1]))
    yy = jnp.array(yy.reshape([-1, 1]))
    theta_pred = jax.vmap(theta_nn, in_axes=(0, 0))(xx, yy)
    uv_pred = jax.vmap(uvp_nn, in_axes=(0, 0))(xx, yy)[:, 0:2]
    theta_pred = np.array(theta_pred).reshape([N, N])
    u_pred = np.array(uv_pred[:, 0:1]).reshape([N, N])
    v_pred = np.array(uv_pred[:, 1:2]).reshape([N, N])
    
    ## interpolation
    x = data['x'].reshape([-1, 1])
    y = data['y'].reshape([-1, 1])
    u = data['u'].reshape([-1, 1])
    v = data['v'].reshape([-1, 1])
    theta = data['theta'].reshape([-1, 1])

    xx, yy = np.meshgrid(xi, yi)
    u_ref = griddata(
        np.hstack([x, y]),
        u,
        (xx, yy), 
        method='cubic',
    )[..., 0]
    v_ref = griddata(
        np.hstack([x, y]),
        v,
        (xx, yy), 
        method='cubic',
    )[..., 0]
    theta_ref = griddata(
        np.hstack([x, y]),
        theta,
        (xx, yy), 
        method='cubic',
    )[..., 0]
    
    fig = plt.figure(figsize=(15, 10), dpi=200)
    ax1 = fig.add_subplot(2, 3, 1)
    # nm1 = colors.Normalize(vmin=float(np.nanmin(theta)), vmax=float(np.nanmax(theta)))
    cp1 = ax1.contourf(xi, yi, theta_pred, cmap=cm.hsv, levels=50)
    fig.colorbar(cp1, ax=ax1, shrink=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('$\\theta$ from PINN')
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    ax1 = fig.add_subplot(2, 3, 2)
    # nm1 = colors.Normalize(vmin=float(np.nanmin(theta)), vmax=float(np.nanmax(theta)))
    cp1 = ax1.contourf(xi, yi, u_pred, cmap=cm.hsv, levels=50)
    fig.colorbar(cp1, ax=ax1, shrink=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('$u$ from PINN')
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    ax1 = fig.add_subplot(2, 3, 3)
    # nm1 = colors.Normalize(vmin=float(np.nanmin(theta)), vmax=float(np.nanmax(theta)))
    cp1 = ax1.contourf(xi, yi, v_pred, cmap=cm.hsv, levels=50)
    fig.colorbar(cp1, ax=ax1, shrink=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('$v$ from PINN')
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    ax1 = fig.add_subplot(2, 3, 4)
    # nm1 = colors.Normalize(vmin=float(np.nanmin(theta)), vmax=float(np.nanmax(theta)))
    cp1 = ax1.contourf(xi, yi, theta_ref, cmap=cm.hsv, levels=50)
    fig.colorbar(cp1, ax=ax1, shrink=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Reference of $\\theta$')
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    ax1 = fig.add_subplot(2, 3, 5)
    # nm1 = colors.Normalize(vmin=float(np.nanmin(theta)), vmax=float(np.nanmax(theta)))
    cp1 = ax1.contourf(xi, yi, u_ref, cmap=cm.hsv, levels=50)
    fig.colorbar(cp1, ax=ax1, shrink=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Reference of $u$')
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    ax1 = fig.add_subplot(2, 3, 6)
    # nm1 = colors.Normalize(vmin=float(np.nanmin(theta)), vmax=float(np.nanmax(theta)))
    cp1 = ax1.contourf(xi, yi, v_ref, cmap=cm.hsv, levels=50)
    fig.colorbar(cp1, ax=ax1, shrink=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Reference of $v$')
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    fig.savefig("./outputs/domain.png")


    ### Boundary ###
    fig = plt.figure(figsize=(20, 20), dpi=200)

    ax1 = fig.add_subplot(4, 3, 1)
    ax1.plot(xx[0, :], theta_ref[0, :], "k-")
    ax1.plot(xx[0, :], theta_pred[0, :], "r--")
    ax1.legend(["Reference", "PINN"])
    ax1.set_title("$y=0.4$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$\\theta$")
    print(yy[0, :])

    ax1 = fig.add_subplot(4, 3, 2)
    ax1.plot(xx[0, :], u_ref[0, :], "k-")
    ax1.plot(xx[0, :], u_pred[0, :], "r--")
    ax1.set_title("$y=0.4$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$u$")

    ax1 = fig.add_subplot(4, 3, 3)
    ax1.plot(xx[0, :], v_ref[0, :], "k-")
    ax1.plot(xx[0, :], v_pred[0, :], "r--")
    ax1.set_title("$y=0.4$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$v$")

    ax1 = fig.add_subplot(4, 3, 4)
    ax1.plot(xx[-1, :], theta_ref[-1, :], "k-")
    ax1.plot(xx[-1, :], theta_pred[-1, :], "r--")
    ax1.set_title("$y=0.6$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$\\theta$")
    print(yy[-1, :])

    ax1 = fig.add_subplot(4, 3, 5)
    ax1.plot(xx[-1, :], u_ref[-1, :], "k-")
    ax1.plot(xx[-1, :], u_pred[-1, :], "r--")
    ax1.set_title("$y=0.6$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$u$")

    ax1 = fig.add_subplot(4, 3, 6)
    ax1.plot(xx[-1, :], v_ref[-1, :], "k-")
    ax1.plot(xx[-1, :], v_pred[-1, :], "r--")
    ax1.set_title("$y=0.6$")
    ax1.set_xlabel("$x$")
    ax1.set_ylabel("$v$")

    ax1 = fig.add_subplot(4, 3, 7)
    ax1.plot(yy[:, 0], theta_ref[:, 0], "k-")
    ax1.plot(yy[:, 0], theta_pred[:, 0], "r--")
    ax1.set_title("$x=0.4$")
    ax1.set_xlabel("$y$")
    ax1.set_ylabel("$\\theta$")
    print(xx[:, 0])

    ax1 = fig.add_subplot(4, 3, 8)
    ax1.plot(yy[:, 0], u_ref[:, 0], "k-")
    ax1.plot(yy[:, 0], u_pred[:, 0], "r--")
    ax1.set_title("$x=0.4$")
    ax1.set_xlabel("$y$")
    ax1.set_ylabel("$u$")

    ax1 = fig.add_subplot(4, 3, 9)
    ax1.plot(yy[:, 0], v_ref[:, 0], "k-")
    ax1.plot(yy[:, 0], v_pred[:, 0], "r--")
    ax1.set_title("$x=0.4$")
    ax1.set_xlabel("$y$")
    ax1.set_ylabel("$v$")

    ax1 = fig.add_subplot(4, 3, 10)
    ax1.plot(yy[:, -1], theta_ref[:, -1], "k-")
    ax1.plot(yy[:, -1], theta_pred[:, -1], "r--")
    ax1.set_title("$x=0.6$")
    ax1.set_xlabel("$y$")
    ax1.set_ylabel("$\\theta$")
    print(xx[:, -1])

    ax1 = fig.add_subplot(4, 3, 11)
    ax1.plot(yy[:, -1], u_ref[:, -1], "k-")
    ax1.plot(yy[:, -1], u_pred[:, -1], "r--")
    ax1.set_title("$x=0.6$")
    ax1.set_xlabel("$y$")
    ax1.set_ylabel("$u$")

    ax1 = fig.add_subplot(4, 3, 12)
    ax1.plot(yy[:, -1], v_ref[:, -1], "k-")
    ax1.plot(yy[:, -1], v_pred[:, -1], "r--")
    ax1.set_title("$x=0.6$")
    ax1.set_xlabel("$y$")
    ax1.set_ylabel("$v$")

    fig.savefig("./outputs/boundaries.png")

