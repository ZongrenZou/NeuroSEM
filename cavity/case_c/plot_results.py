import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import sys


DATA_FILE_NAME = "./data/data_1e4.mat"
THETA_FILE_NAME = "./outputs/theta_grid_1e4.npy"
FIG_FILE_NAME = "./outputs/theta_1e4.png"


### Additional data
data = sio.loadmat("./outputs/case_c.mat")
x_f = data["x_f"]
y_f = data["y_f"]
x_d = data["x"]
y_d = data["y"]
theta_d = data["theta"]

### PINN DATA
number = 100
dens = 1
xi = np.linspace(0, 1, number)
yi = np.linspace(0, 1, number)
x_grid, y_grid = np.meshgrid(xi, yi)
theta = np.load(THETA_FILE_NAME, allow_pickle=True)
theta = theta.reshape((100, 100))
print(f"{np.shape(theta)}")

fig = plt.figure(figsize=(20, 8))
fig.suptitle("PINN without top and bottom Dirichlet boundary condition", fontsize=30)
ax1 = fig.add_subplot(141)
nm1 = colors.Normalize(vmin=float(np.nanmin(theta)), vmax=float(np.nanmax(theta)))
cp1 = ax1.contourf(xi, yi, theta, cmap=cm.hsv, norm=nm1, levels=50)
fig.colorbar(cp1, ax=ax1, shrink=0.5)
# ax1.plot(x_f.flatten(), y_f.flatten(), "wo")
# ax1.plot(x_d.flatten(), y_d.flatten(), "kx")
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('PINN')
ax1.set_aspect('equal', adjustable='box')

## Actual Data
data_num = sio.loadmat(DATA_FILE_NAME)
x_num = data_num["x"].reshape(-1, 1)
y_num = data_num["y"].reshape(-1, 1)
theta_num = data_num["theta"].reshape(-1, 1)
xyp = np.hstack([x_num, y_num])
theta_num_grid = griddata(xyp, theta_num, (x_grid, y_grid), method='cubic')
theta_num_grid = theta_num_grid[:, :, 0]

ax2 = fig.add_subplot(142)
nm2 = colors.Normalize(vmin=float(np.nanmin(theta_num_grid)), vmax=float(np.nanmax(theta_num_grid)))
cp2 = ax2.contourf(xi, yi, theta_num_grid, cmap=cm.hsv, norm=nm2, levels=50)
fig.colorbar(cp2, ax=ax2, shrink=0.5)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Reference')
ax2.set_aspect('equal', adjustable='box')

err = np.sum((theta - theta_num_grid) ** 2) / np.sum(theta_num_grid ** 2)
print("Error of theta: ", np.sqrt(err))


### Error
theta_err = theta_num_grid - theta

ax3 = fig.add_subplot(143)
nm3 = colors.Normalize(vmin=float(np.nanmin(theta_err)), vmax=float(np.nanmax(theta_err)))
cp3 = ax3.contourf(xi, yi, np.abs(theta_err), cmap=cm.hsv, norm=nm3, levels=50)
fig.colorbar(cp3, ax=ax3, shrink=0.5)
# ax3.plot(x_d.flatten(), y_d.flatten(), "kx")
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Pointwise error')
ax3.set_aspect('equal', adjustable='box')


ax4 = fig.add_subplot(144)
ax4.plot(x_f.flatten(), y_f.flatten(), "m.")
ax4.plot(x_d.flatten(), y_d.flatten(), "kx")
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('Data (x for $\\theta$; $\cdot$ for $u, v$)')
ax4.set_aspect('equal', adjustable='box')

plt.savefig(FIG_FILE_NAME, dpi=300)
