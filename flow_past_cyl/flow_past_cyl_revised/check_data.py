import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt


data = sio.loadmat('./data/data.mat')
x_ic = data['x_ic'].flatten()[:, None]
y_ic = data['y_ic'].flatten()[:, None]
t_ic = np.zeros_like(x_ic)
print(x_ic.shape, y_ic.shape, t_ic.shape)

x_bc = data['x_bc'].flatten()[:, None]
y_bc = data['y_bc'].flatten()[:, None]
t_bc = data["Time"].flatten()[:, None]
print("######################")
print(x_bc.shape, y_bc.shape, t_bc.shape)
theta_bc = data["theta_bc"].flatten()[:, None]
print(theta_bc[0:1016] - theta_bc[0:1016])
# print(theta_bc[0:100] - theta_bc[100:200])

print("**********************")


xyt_cyl = data["xyt_cyl"]
theta_cyl = data["theta_cyl"]
print(xyt_cyl[:360*11, -1])


N_T_f = 100
K = 1000
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
print(t[:21])
for i in range(t[:21].shape[0]):
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

print(u_f.shape, theta_ref.shape)