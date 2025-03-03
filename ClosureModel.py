import sys
sys.path.append('C:\\UWMadisonResearch\\SBM_FNO_Closure\\DiffusionTerm_Generation')

import numpy as np
import h5py
import math
import torch
import torch.nn.functional as F
from functools import partial
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rc("text", usetex=True)
mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
from mpl_toolkits.axes_grid1 import make_axes_locatable

from Data_Generation.generator_sns import navier_stokes_2d_closure
from utility import (set_seed, get_sigmas_karras, sampler, fro_err, mse_err, interpolate2d, energy_spectrum)
from Model_Designs import (marginal_prob_std, diffusion_coeff, FNO2d)

import warnings
warnings.filterwarnings("ignore")

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')


sigma = 26
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 8
width = 20

model_interp_G = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
model_interp_G.load_state_dict(torch.load('C:\\UWMadisonResearch\\SBM_FNO_Closure\\Training_Sampling'
                  '\\Trained_Models\\SparseDiffusionModel_Interp_v2.pth', map_location=device))

model_conv_G = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
model_conv_G.load_state_dict(torch.load('C:\\UWMadisonResearch\\SBM_FNO_Closure\\Training_Sampling'
                  '\\Trained_Models\\SparseDiffusionModel_Conv_v2.pth', map_location=device))

model_interp_H = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
model_interp_H.load_state_dict(torch.load('C:\\UWMadisonResearch\\SBM_FNO_Closure\\Training_Sampling'
                  '\\Trained_Models\\SparseNonlinearModel_Interp.pth', map_location=device))

model_conv_H = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
model_conv_H.load_state_dict(torch.load('C:\\UWMadisonResearch\\SBM_FNO_Closure\\Training_Sampling'
                  '\\Trained_Models\\SparseNonlinearModel_Conv.pth', map_location=device))

filename = 'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Data_Generation\\2d_ns_3050_sto_64_closure_traj.h5'

# Open the HDF5 file
with h5py.File(filename, 'r') as file:
    sol_t = torch.tensor(file['t'][()], device='cuda')
    sol = torch.tensor(file['sol'][()], device='cuda')
    diffusion = torch.tensor(file['diffusion'][()], device='cuda')
    nonlinear = torch.tensor(file['nonlinear'][()], device='cuda')

delta_t = 1e-3
nu = 1e-3
sample_size = 1
sample_steps = 10
total_steps = 20000
spatial_dim = 64

vorticity_init = sol[0:1, :, :, 0]

vorticity_condition = sol[0:1, :, :, 0: total_steps]

t = torch.linspace(0, 1, spatial_dim + 1, device=device)
t = t[0:-1]

X, Y = torch.meshgrid(t, t)
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

# Extract sparse information for convoluted model
diffusion_target = diffusion[0:0+sample_size, :, :, 0: total_steps]
nonlinear_target = nonlinear[0:0+sample_size, :, :, 0: total_steps]
# Define the size of the convolutional kernel
kernel_size = 7
kernel64 = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
kernel64 = kernel64.to(device)


mask = torch.zeros_like(diffusion_target)
mask[:, ::4, ::4, :] = 1
diffusion_target_sparse = diffusion_target * mask
diffusion_target_sparse_GF = torch.empty_like(diffusion_target_sparse)

for t in range(total_steps):
    slice_squeezed = diffusion_target_sparse[:, :, :, t].unsqueeze(1)
    slice_convolved  = F.conv2d(slice_squeezed, kernel64, padding='same')
    diffusion_target_sparse_GF[:, :, :, t] = slice_convolved.squeeze(1)

diffusion_target_sparse_normalized = torch.empty_like(diffusion_target_sparse_GF)
for i in range(diffusion_target_sparse_GF.shape[0]):
    for t in range(diffusion_target_sparse_GF.shape[3]):
        batch_sparse = diffusion_target_sparse[i, :, :, t][diffusion_target_sparse[i, :, :, t] != 0]
        batch_smoothed = diffusion_target_sparse_GF[i, :, :, t][diffusion_target_sparse_GF[i, :, :, t] != 0]
        sparse_min, sparse_max = torch.min(batch_sparse), torch.max(batch_sparse)
        smoothed_min, smoothed_max = torch.min(batch_smoothed), torch.max(batch_smoothed)
        batch_normalized = (diffusion_target_sparse_GF[i, :, :, t] - smoothed_min) / (smoothed_max - smoothed_min)
        batch_normalized = batch_normalized * (sparse_max - sparse_min) + sparse_min
        diffusion_target_sparse_normalized[i, :, :, t] = batch_normalized

diffusion_target_sparse = diffusion_target.reshape(-1, 64, 64)
diffusion_target_interp = interpolate2d(diffusion_target_sparse, 16).reshape(1, 64, 64, 20000)

nonlinear_target_sparse = nonlinear_target * mask
nonlinear_target_sparse_GF = torch.empty_like(nonlinear_target_sparse)

for t in range(total_steps):
    slice_squeezed = nonlinear_target_sparse[:, :, :, t].unsqueeze(1)
    slice_convolved  = F.conv2d(slice_squeezed, kernel64, padding='same')
    nonlinear_target_sparse_GF[:, :, :, t] = slice_convolved.squeeze(1)

nonlinear_target_sparse_normalized = torch.empty_like(nonlinear_target_sparse_GF)
for i in range(nonlinear_target_sparse_GF.shape[0]):
    for t in range(nonlinear_target_sparse_GF.shape[3]):
        batch_sparse = nonlinear_target_sparse[i, :, :, t][nonlinear_target_sparse[i, :, :, t] != 0]
        batch_smoothed = nonlinear_target_sparse_GF[i, :, :, t][nonlinear_target_sparse_GF[i, :, :, t] != 0]
        sparse_min, sparse_max = torch.min(batch_sparse), torch.max(batch_sparse)
        smoothed_min, smoothed_max = torch.min(batch_smoothed), torch.max(batch_smoothed)
        batch_normalized = (nonlinear_target_sparse_GF[i, :, :, t] - smoothed_min) / (smoothed_max - smoothed_min)
        batch_normalized = batch_normalized * (sparse_max - sparse_min) + sparse_min
        nonlinear_target_sparse_normalized[i, :, :, t] = batch_normalized

nonlinear_target_sparse = nonlinear_target.reshape(-1, 64, 64)
nonlinear_target_interp = interpolate2d(nonlinear_target_sparse, 16).reshape(1, 64, 64, 20000)


sde_time_min = 1e-3
sde_time_max = 0.1

time_noises = get_sigmas_karras(sample_steps, sde_time_min, sde_time_max, device=device)
sample_steps = 10
sampler_conv_G = partial(sampler,
                  score_model = model_conv_G,
                    spatial_dim = spatial_dim,
                    marginal_prob_std = marginal_prob_std_fn,
                    diffusion_coeff = diffusion_coeff_fn,
                    batch_size = sample_size,
                    num_steps = sample_steps,
                    time_noises = time_noises,
                    device = device,
                    sparse = True)

sampler_interp_G = partial(sampler,
                    score_model = model_interp_G,
                        spatial_dim = spatial_dim,
                        marginal_prob_std = marginal_prob_std_fn,
                        diffusion_coeff = diffusion_coeff_fn,
                        batch_size = sample_size,
                        num_steps = sample_steps,
                        time_noises = time_noises,
                        device = device,
                        sparse = True)

sampler_conv_H = partial(sampler,
                    score_model = model_conv_H,
                        spatial_dim = spatial_dim,
                        marginal_prob_std = marginal_prob_std_fn,
                        diffusion_coeff = diffusion_coeff_fn,
                        batch_size = sample_size,
                        num_steps = sample_steps,
                        time_noises = time_noises,
                        device = device,
                        sparse = True)

sampler_interp_H = partial(sampler,
                    score_model = model_interp_H,
                        spatial_dim = spatial_dim,
                        marginal_prob_std = marginal_prob_std_fn,
                        diffusion_coeff = diffusion_coeff_fn,
                        batch_size = sample_size,
                        num_steps = sample_steps,
                        time_noises = time_noises,
                        device = device,
                        sparse = True)

set_seed(13)
vorticity_withoutG, _, t_withoutG = navier_stokes_2d_closure([1, 1], vorticity_init, f, nu, diffusion_target_interp,
                                            sampler, missing_term='diffusion', closure = False, delta_t = delta_t,
                                            record_steps = total_steps, eval_steps=5)
set_seed(13)
vorticity_withG_conv, _, t_withG = navier_stokes_2d_closure([1, 1], vorticity_init, f, nu, diffusion_target_sparse_normalized,
                                            sampler = sampler_conv_G, missing_term='diffusion', closure = True, delta_t = delta_t,
                                            record_steps = total_steps, eval_steps=5)
set_seed(13)
vorticity_withG_interp, _, t_withG = navier_stokes_2d_closure([1, 1], vorticity_init, f, nu, diffusion_target_interp,
                                            sampler = sampler_interp_G, missing_term='diffusion', closure = True, delta_t = delta_t,
                                            record_steps = total_steps, eval_steps=5)

set_seed(13)
vorticity_withoutH, _, t_withoutH = navier_stokes_2d_closure([1, 1], vorticity_init, f, nu, nonlinear_target_interp,
                                            sampler, missing_term='nonlinear', closure = False, delta_t = delta_t,
                                            record_steps = total_steps, eval_steps=5)
set_seed(13)
vorticity_withH_conv, _, t_withH = navier_stokes_2d_closure([1, 1], vorticity_init, f, nu, nonlinear_target_sparse_normalized,
                                            sampler = sampler_conv_H, missing_term='nonlinear', closure = True, delta_t = delta_t,
                                            record_steps = total_steps, eval_steps=5)
set_seed(13)
vorticity_withH_interp, _, t_withH = navier_stokes_2d_closure([1, 1], vorticity_init, f, nu, nonlinear_target_interp,
                                            sampler = sampler_interp_H, missing_term='nonlinear', closure = True, delta_t = delta_t,
                                            record_steps = total_steps, eval_steps=5)


k = 0
fs = 37
# Create a figure and a grid of subplots
fig, axs = plt.subplots(5, 5, figsize=(25, 27), gridspec_kw={'width_ratios': [1]*4 + [1.073]})

# Plot each row using seaborn heatmap
for row in range(5):
    for i in range(5):  # Loop through all ten columns
        ax = axs[row, i]

        j = i * 4999
        generated = vorticity_withH_conv[k, :, :, j].cpu()
        generated_nog = vorticity_withoutH[k, :, :, j].cpu()
        truth = sol[k, :, :, j+1].cpu()
        error_field = abs(generated - truth)
        error_field_nog = abs(generated_nog - truth)

        rmse = fro_err(torch.tensor(generated.unsqueeze(0)), torch.tensor(truth.unsqueeze(0)))
        mse = mse_err(torch.tensor(generated.unsqueeze(0)), torch.tensor(truth.unsqueeze(0)))
        rmse_nog = fro_err(torch.tensor(generated_nog.unsqueeze(0)), torch.tensor(truth.unsqueeze(0)))
        mse_nog = mse_err(torch.tensor(generated_nog.unsqueeze(0)), torch.tensor(truth.unsqueeze(0)))

        if row == 0:
            print(f"Time: {sol_t[j] + 30:.2f}s")
            print(f"RMSE: {rmse:.4f}")
            print(f"MSE: {mse:.8f}")
            print(f"RMSE NoG: {rmse_nog:.4f}")
            print(f"MSE NoG: {mse_nog:.8f}")

        # Set individual vmin and vmax based on the row
        if row == 0:
            data = truth
            vmin, vmax = -2.5, 3.0  # Limits for Truth and Generated rows
            ax.set_title(f'$t$ = {sol_t[j]+30:.2f}', fontsize=fs)

            sns.heatmap(data, ax=ax, cmap="rocket", vmin=vmin, vmax=vmax, square=True, cbar=False)
        elif row == 1:
            data = generated
            vmin, vmax = -2.5, 3.0  # Limits for Truth and Generated rows

            sns.heatmap(data, ax=ax, cmap="rocket", vmin=vmin, vmax=vmax, square=True, cbar=False)
        elif row == 2:
            data = generated_nog
            vmin, vmax = -2.5, 3.0

            sns.heatmap(data, ax=ax, cmap="rocket", vmin=vmin, vmax=vmax, square=True, cbar=False)
        elif row == 3:
            data = error_field
            vmin, vmax = 0, 3.0
            cbar_ticks_contour = np.linspace(vmin, vmax, 6)
            S = data.shape[0]
            x = np.arange(S)
            y = np.arange(S)
            X, Y = np.meshgrid(x, y)
            ax_contour = ax.contourf(X, Y, data, levels=cbar_ticks_contour,
                                     cmap="rocket", vmin=vmin, vmax=vmax)
            ax.set_aspect('equal', adjustable='box')
        else:
            data = error_field_nog
            vmin, vmax = 0, 3.0
            cbar_ticks_contour = np.linspace(vmin, vmax, 6)
            S = data.shape[0]
            x = np.arange(S)
            y = np.arange(S)
            X, Y = np.meshgrid(x, y)
            ax_contour = ax.contourf(X, Y, data, levels=cbar_ticks_contour,
                                     cmap="rocket", vmin=vmin, vmax=vmax)
            ax.set_aspect('equal', adjustable='box')

        ax.axis('off')

        if i == 4:
            # Create a new axis for the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(ax.collections[0], cax=cax, ticks=np.linspace(vmin, vmax, 6))
            cax.tick_params(labelsize=fs)

            # Format tick labels based on the row
            if row < 3:  # For the first two rows
                cb.ax.set_yticklabels(['{:.1f}'.format(tick) for tick in np.linspace(vmin, vmax, 6)])

# Add row titles on the side
row_titles = [f'Truth', f'Simulation with $H$', f'Simulation w/o $H$', f'Error with $H$', f'Error w/o $H$']
for ax, row_title in zip(axs[:, 0], row_titles):
    ax.annotate(row_title, xy=(0.1, 0.5), xytext=(-50, 0),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='center', rotation=90, fontsize=fs)

plt.tight_layout()  # Adjust the subplots to fit into the figure area
plt.savefig(
    'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Plots\\Closure_H.png',
    dpi=300,
    bbox_inches='tight'
)




vorticity_withH_conv[..., 0] = sol[..., 0]
vorticity_withG_conv[..., 0] = sol[..., 0]

fig, axs = plt.subplots(1, 3, figsize=(42, 12), gridspec_kw={'width_ratios': [1, 1, 1]})
fs=62
# Flatten the axs array and only use the first 5 subplots
axs = axs.flatten()
for i, ax in enumerate(axs):  # Use only the first 5 axes
    index = i * 9999

    # Compute energy spectra
    _, k32, E32 = energy_spectrum(sol[0:1, :, :, index].cpu(), 1, 1, smooth=False)
    _, k32_conv_H, E32_conv_H = energy_spectrum(vorticity_withH_conv[0:1, :, :, index].cpu(), 1, 1, smooth=False)
    _, k32_conv_G, E32_conv_G = energy_spectrum(vorticity_withG_conv[0:1, :, :, index].cpu(), 1, 1, smooth=False)

    # Plot energy spectra
    ax.loglog(k32[1:], E32[1:], label=f'Truth', linewidth=6, linestyle=":")
    ax.loglog(k32_conv_H[1:], E32_conv_H[1:], label=f'Closure with $G$', linewidth=6, linestyle="--")
    ax.loglog(k32_conv_G[1:], E32_conv_G[1:], label=f'Closure with $H$', linewidth=6, linestyle="-.")
    ax.tick_params(axis='both', which='major', length=14, width=2)
    ax.tick_params(axis='both', which='minor', length=7, width=2)
    ax.tick_params(axis='x', which='major', pad=12)

    # Add reference line for k^(-3)
    k_ref = k32[2]   # Reference k point in the middle
    E_ref = E32[2]
    k_line = np.linspace(7, 50, 100)
    E_line = E_ref * (k_line / k_ref) ** (-3)
    ax.loglog(k_line, E_line, linestyle='solid', label=r'$k^{-3}$', color='red', linewidth=6)

    # Set plot details
    ax.set_title(f'$t$ = {sol_t[index]+30:.2f}', fontsize=fs)
    ax.set_xlim(k32[1], 10**3)  # Ensure a minimum of 1 for k
    ax.tick_params(axis='both', labelsize=fs)

    ax.set_ylim(10**-21, 10**0)
    ax.set_xlim(None, 1e3)
    ticks = [10 ** -21, 10 ** -14, 10 ** -7, 10 ** 0]

    # Define corresponding labels (the first label is for the lowest tick, etc.)
    labels = ['$10^{-21}$', '$10^{-14}$', '$10^{-7}$', '$10^{0}$']

    # Apply the tick positions and labels
    ax.set_yticks(ticks)
    # ax.set_yticklabels(labels)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    if i == 0:
        ax.set_ylabel(f'Energy ($E$)', fontsize=fs)
    if i == 0 or i == 1 or i == 2:
        ax.set_xlabel(f'Wave number ($k$)', fontsize=fs)
    if i ==1 or i == 2:
        ax.get_yaxis().set_ticks([])

handles, labels = axs[0].get_legend_handles_labels()
lege = fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=fs,bbox_to_anchor=(0.5, 1.05),
                  fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(2)
plt.subplots_adjust(top=0.85)
plt.tight_layout(rect=[0, 0, 1, 0.88])
# plt.show()
plt.savefig(
    'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Plots\\TKE_Closure_3050.png',
    dpi=300,
    bbox_inches='tight'
)
