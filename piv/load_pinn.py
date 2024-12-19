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
    # ============= Specify the hyperparameters ===============
    # =========================================================
    LOAD_MODEL_NAME = "./checkpoints/PINN1.eqx"

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
    xyt_d = np.concatenate([X_star, Y_star, T_star], axis=-1)
    uv_d = np.concatenate([U_star, V_star], axis=-1)

    # ====================================================================
    # =========================== Load model =============================
    # ====================================================================
    key = jr.PRNGKey(6677)
    key, init_key = jr.split(key)
    pinn = models.NeuralNetwork(init_key)
    pinn = eqx.tree_deserialise_leaves(LOAD_MODEL_NAME, pinn)


    # ====================================================================
    # ======================== Make prediction ===========================
    # ====================================================================
    ## evaluate on the original grid points
    uvp_pred = jax.vmap(pinn, in_axes=(0, 0, 0))(X_star, Y_star, T_star)
    u_pred, v_pred, p_pred = np.split(uvp_pred, 3, axis=-1)

    print(uvp_pred.shape, u_pred.shape, v_pred.shape, U_star.shape, V_star.shape)

    error_u = np.linalg.norm(U_star - u_pred, 2) / np.linalg.norm(U_star, 2)
    error_v = np.linalg.norm(V_star - v_pred, 2) / np.linalg.norm(V_star, 2)
    print(error_u, error_v)

    mse_u = np.mean((U_star - u_pred) ** 2)
    mse_v = np.mean((V_star - v_pred) ** 2)
    print(mse_u, mse_v, mse_u+mse_v)

    sio.savemat(
        "./outputs/results_ori_1.mat",
        {
            "U_pred": u_pred,
            "V_pred": v_pred,
            "P_pred": p_pred,
            "X_star": X_star,
            "Y_star": Y_star,
            "T_star": T_star,
            "U_star": U_star,
            "V_star": V_star,
        }
    )

    ## evaluate on the dense grid points

    # ====================================================================
    # ===========  evaluation on the original grid points ================
    # ====================================================================
    data = sio.loadmat("./data/PINNdata_grids.mat")
    t_test_all = data["T_grid"].astype(np.float32).flatten()[:, None]
    x_test = data["X_grid"].astype(np.float32).flatten()[:, None]
    y_test = data["Y_grid"].astype(np.float32).flatten()[:, None]

    for t_idx in range(0, 51):
    
        t_test = t_test_all[t_idx,:].flatten()[:, None]
        T = t_test.shape[0]
        N = x_test.shape[0] 
        # Rearrange Data 
        T_test = np.tile(t_test, (1, N)).T               # N x T
        X_test = np.tile(x_test.reshape(N, 1), (1, T))    # N x T
        Y_test = np.tile(y_test.reshape(N, 1), (1, T))    # N x T
        T_test = T_test.flatten()[:, None] # NT x 1
        X_test = X_test.flatten()[:, None] # NT x 1
        Y_test = Y_test.flatten()[:, None] # NT x 1

        uvp_pred = jax.vmap(pinn, in_axes=(0, 0, 0))(X_test, Y_test, T_test)
        u_pred, v_pred, p_pred = np.split(uvp_pred, 3, axis=-1)

        u_pred = u_pred.reshape(759, 1241)
        v_pred = v_pred.reshape(759, 1241)
        p_pred = p_pred.reshape(759, 1241)

        sio.savemat(
            "./outputs/results_grids_t{}.mat".format(str(t_idx+1)),
            {
                "U_pred": u_pred, "V_pred": v_pred, "P_pred": p_pred,
                # "X_grid": X_grid, "Y_grid": Y_grid, "T_grid": t_test_all,
            }
        )
        
    print("End main.")    