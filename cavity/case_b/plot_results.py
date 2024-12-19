import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import sys



DATA_FILE_NAME = "./data/data_1e6.mat"
UV_FILE_NAME = "./outputs/uv_grid_1e6.mat"
FIG_FILE_NAME = "./outputs/uv_1e6.png"


### PINN DATA
number = 100
dens = 1
xi = np.linspace(0, 1, number)
yi = np.linspace(0, 1, number)
x_grid, y_grid = np.meshgrid(xi, yi)
uv_grid_data = sio.loadmat(UV_FILE_NAME)
u = uv_grid_data["u_grid"].reshape([100, 100])
v = uv_grid_data["v_grid"].reshape([100, 100])


## reference from SEM
data_num = sio.loadmat(DATA_FILE_NAME)
x_num = data_num["x"].reshape(-1, 1)
y_num = data_num["y"].reshape(-1, 1)
u_num = data_num["u"].reshape(-1, 1)
v_num = data_num["v"].reshape(-1, 1)
xyp = np.hstack([x_num, y_num])
u_num_grid = griddata(xyp, u_num, (x_grid, y_grid), method='cubic')
u_num_grid = u_num_grid[:, :, 0]
v_num_grid = griddata(xyp, v_num, (x_grid, y_grid), method='cubic')
v_num_grid = v_num_grid[:, :, 0]
u_ref = u_num_grid
v_ref = v_num_grid

## figure
fig = plt.figure(figsize=(20, 5))
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)
ax4 = fig.add_subplot(144)

## subfigure
nm1 = colors.Normalize(vmin=float(np.nanmin(u)), vmax=float(np.nanmax(u)))
cp1 = ax1.contourf(xi, yi, u, cmap=cm.hsv, norm=nm1, levels=15)
fig.colorbar(cp1, ax=ax1, shrink=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('u from PINN')
ax1.set_aspect('equal', adjustable='box')

nm2 = colors.Normalize(vmin=float(np.nanmin(u_ref)), vmax=float(np.nanmax(u_ref)))
cp2 = ax2.contourf(xi, yi, u_ref, cmap=cm.hsv, norm=nm2, levels=15)
fig.colorbar(cp2, ax=ax2, shrink=0.5)
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('u from SEM')
ax2.set_aspect('equal', adjustable='box')

nm3 = colors.Normalize(vmin=float(np.nanmin(v)), vmax=float(np.nanmax(v)))
cp3 = ax3.contourf(xi, yi, v, cmap=cm.hsv, norm=nm3, levels=15)
fig.colorbar(cp3, ax=ax3, shrink=0.5)
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('v from PINN')
ax3.set_aspect('equal', adjustable='box')

nm4 = colors.Normalize(vmin=float(np.nanmin(v_ref)), vmax=float(np.nanmax(v_ref)))
cp4 = ax4.contourf(xi, yi, v_ref, cmap=cm.hsv, norm=nm4, levels=15)
fig.colorbar(cp4, ax=ax4, shrink=0.5)
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_title('v from SEM')
ax4.set_aspect('equal', adjustable='box')

plt.savefig(FIG_FILE_NAME, dpi=300)