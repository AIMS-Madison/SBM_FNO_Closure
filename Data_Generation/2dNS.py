import sys
sys.path.append('C:\\UWMadisonResearch\\SBM_FNO_Closure')

import torch
import numpy as np
import math
import h5py
from timeit import default_timer
from Data_Generation.generator_sns import navier_stokes_2d, navier_stokes_2d_model
from Data_Generation.random_forcing import GaussianRF

filename = "./Data_Generation/"
device = torch.device('cuda')

# Viscosity parameter
nu = 1e-3

# Spatial Resolution
s = 64
sub = 1

# Temporal Resolution
T = 40
delta_t = 1e-3

# Number of solutions to generate
N = 1

# Set up 2d GRF with covariance parameters
GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

# Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s + 1, device=device)
t = t[0:-1]

X, Y = torch.meshgrid(t, t)
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

# Stochastic forcing function: sigma*dW/dt
stochastic_forcing = {'alpha': 0.005, 'kappa': 10, 'sigma': 0.00005}

# Number of snapshots from solution
record_steps = 20000

# Solve equations in batches (order of magnitude speed-up)
# Batch size
bsize = 1

c = 0
t0 = default_timer()

sol_col = torch.zeros(N, s, s, record_steps+1).to(device)
nonlinear_col = torch.zeros(N, s, s, record_steps+1).to(device)
diffusion_col = torch.zeros(N, s, s, record_steps+1).to(device)

for j in range(N // bsize):
    w0 = GRF.sample(bsize)

    sol, sol_t, diffusion, nonlinear = navier_stokes_2d([1, 1], w0, f, nu, T, delta_t, record_steps,
                                                                thres=20000, stochastic_forcing = None)

    c += bsize
    t1 = default_timer()
    print(j, c, t1 - t0)

    sol_col[j * bsize:(j + 1) * bsize] = sol
    nonlinear_col[j * bsize:(j + 1) * bsize] = nonlinear
    diffusion_col[j * bsize:(j + 1) * bsize] = diffusion


sol_nocor, sol_t, execution_time = navier_stokes_2d_model(sol[..., 0], f, nu, nonlinear[..., 0:], None, False,
                                                          delta_t, 20000)

sol_t_np = sol_t.cpu().numpy()
sol_col_np = sol_col.cpu().numpy()
nonlinear_col_np = nonlinear_col.cpu().numpy()
diffusion_col_np = diffusion_col.cpu().numpy()

nu_np = np.array(nu)

# raw data
filename = '2040_64_traj.h5'

with h5py.File(filename, 'w') as file:
    file.create_dataset('t', data=sol_t_np)
    file.create_dataset('sol', data=sol_col_np)
    file.create_dataset('nonlinear', data=nonlinear_col_np)
    file.create_dataset('diffusion', data=diffusion_col_np)
    file.create_dataset('nu', data=nu_np)

print(f'Data saved to {filename}')


import seaborn as sns
import matplotlib.pyplot as plt


fig = plt.figure(figsize=(8, 10))
axs = fig.subplots(5, 5)
for n in range(25):
    sns.heatmap(sol_col_np[0, :, :, n*800], ax=axs.flatten()[n], cbar=False, cmap="rocket")
    axs.flatten()[n].axis("off")
fig.tight_layout()
plt.show()
