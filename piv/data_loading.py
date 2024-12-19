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
    N_EPOCHS = int(1e6)
    batch_size = 50000


    # ====================================================================
    # ========================= Load data ================================
    # ====================================================================
    data = sio.loadmat("./data/PINNdata_dSpace1_dTime1.mat")
    Re = data['ReyNum'].astype(np.float32) 
    T_star = data['T_star'].astype(np.float32).flatten()[:, None]
    X_star = data['X_star'].astype(np.float32).flatten()[:, None]
    Y_star = data['Y_star'].astype(np.float32).flatten()[:, None] 
    U_star = data['U_star'].astype(np.float32).flatten()[:, None]
    V_star = data['V_star'].astype(np.float32).flatten()[:, None]
    N_data = T_star.shape[0]
    print(np.unique(T_star))
    print("*****************************")
    print(N_data, N_f)
    xyt_d = np.concatenate([X_star, Y_star, T_star], axis=-1)
    uv_d = np.concatenate([U_star, V_star], axis=-1)

    xyt_f = models.LHSample(3, [[t_min, t_max], [x_min, x_max], [y_min, y_max]], N_f)

    print(xyt_f.shape, xyt_d.shape, uv_d.shape)