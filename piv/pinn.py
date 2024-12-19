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


import models


if __name__ == "__main__": 

    # =========================================================
    # ============= define the domain  ========================
    # =========================================================
    # They can also be loaded from the data
    x_min = 0
    x_max = 3.5179
    y_min = -2.1504
    y_max =  0
    t_min = 0
    t_max = 0.500


    # =========================================================
    # ============= Specify the hyperparameters ===============
    # =========================================================
    DATA_WEIGHT = 10 # weight for the data loss
    PDE_WEIGHT = 1 # weight for the PDE loss
    N_f = 1000000 # number of residual points
    N_EPOCHS = int(1000)
    batch_size = 50000
    SAVE_MODEL_NAME = "./checkpoints/PINN.eqx"

    # ====================================================================
    # ========================= Load data ================================
    # ====================================================================
    data = sio.loadmat("./data/PINNdata_dSpace1_dTime1.mat")
    Re = data['ReyNum'].astype(np.float32).reshape([])
    T_star = data['T_star'].astype(np.float32).flatten()[:, None]
    X_star = data['X_star'].astype(np.float32).flatten()[:, None]
    Y_star = data['Y_star'].astype(np.float32).flatten()[:, None] 
    U_star = data['U_star'].astype(np.float32).flatten()[:, None]
    V_star = data['V_star'].astype(np.float32).flatten()[:, None]
    N_data = T_star.shape[0]
    print(np.unique(T_star))
    print("*****************************")
    print(N_data, N_f)
    print(T_star.shape, X_star.shape, Y_star.shape, U_star.shape, V_star.shape)
    xyt_d = np.concatenate([X_star, Y_star, T_star], axis=-1)
    uv_d = np.concatenate([U_star, V_star], axis=-1)

    xyt_r = models.LHSample(3, [[t_min, t_max], [x_min, x_max], [y_min, y_max]], N_f)


    # ====================================================================
    # ==================+ Build model and train step =====================
    # ====================================================================
    key = jr.PRNGKey(6677)
    key, init_key = jr.split(key)
    pinn = models.NeuralNetwork(init_key)
    pinn = models.init_linear_weight(pinn, models.trunc_init, init_key)

    schedule = optax.piecewise_constant_schedule(
        init_value=1e-3,
        boundaries_and_scales={
            int(2e5): 0.5,
            int(4e5): 0.2,
            int(6e5): 0.1,
        }
    )

    optimizer = optax.adamw(learning_rate=schedule)
    # optimizer = optax.adamw(learning_rate=1e-3)
    opt_state = optimizer.init(eqx.filter(pinn, eqx.is_array))

    @eqx.filter_jit
    def train_step_opt(network, params_state, xyt_r, xyt_d, uv_d):

        l, grad = eqx.filter_value_and_grad(models.loss_fn)(
            network, DATA_WEIGHT, PDE_WEIGHT, xyt_r, xyt_d, uv_d, Re,
        )
        updates, new_state = optimizer.update(grad, params_state, network)
        new_network = eqx.apply_updates(network, updates)
        return new_network, new_state, l
    

    for epoch in range(N_EPOCHS):
        idx_d = np.random.choice(N_data, N_data, replace=False)

        for i in range(N_data // batch_size + 1):
            batch_id = idx_d[i*batch_size: (i+1)*batch_size]
            xyt_d_batch = xyt_d[batch_id, :]
            uv_d_batch = uv_d[batch_id, :]

            idx_r = np.random.choice(N_f, N_f, replace=False)
            for j in range(N_f // batch_size):
                batch_id = idx_r[j*batch_size: (j+1)*batch_size]
                xyt_r_batch = xyt_r[batch_id]

                pinn, opt_state, loss = train_step_opt(
                    pinn, opt_state, xyt_r_batch, xyt_d_batch, uv_d_batch,
                )

        total_loss, data_loss, pde_loss = models.loss_fn_2(pinn, DATA_WEIGHT, PDE_WEIGHT, xyt_r_batch, xyt_d, uv_d, Re)
        print("Epoch: ", epoch, ", total loss:", total_loss, ", data loss: ", data_loss, ", pde loss: ", pde_loss)
        
        eqx.tree_serialise_leaves(SAVE_MODEL_NAME, pinn)


    print("End main.")    