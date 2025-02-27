import sys
sys.path.append('C:\\UWMadisonResearch\\SBM_FNO_Closure\\DiffusionTerm_Generation')
import time
import numpy as np
import h5py
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Qt5Agg")
plt.rcParams["agg.path.chunksize"] = 10000
plt.rc("text", usetex=True)
mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from functools import partial
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from utility import (set_seed, get_sigmas_karras, sampler, fro_err, mse_err, interpolate2d)
from Model_Designs import (marginal_prob_std, diffusion_coeff, FNO2d)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')

def navier_stokes_2d_diffusion(a, w0, f, visc, sparse_condition, sampler,
                           closure = False, delta_t=1e-4, record_steps=1, eval_steps=10):
    # Grid size - must be power of 2
    N1, N2 = w0.size()[-2], w0.size()[-1]

    # Maximum frequency
    k_max = math.floor(N1 / 2.0)

    # Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)
    # Forcing to Fourier space
    f_h = torch.fft.rfft2(f)
    # If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                     torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N1, 1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0, 1)

    # Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    # Physical wavenumbers
    kx_2d = 2.0 * torch.pi * k_x / a[0]
    ky_2d = 2.0 * torch.pi * k_y / a[1]

    # Negative Laplacian in Fourier space
    lap = kx_2d ** 2 + ky_2d ** 2
    lap[0, 0] = 1.0

    # Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max,
                                                torch.abs(k_x) <= (2.0 / 3.0) * k_max).float(), 0)

    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    t = 0.0

    start_time = time.time()
    for i in tqdm(range(record_steps)):
        psi_h= w_h / lap

        # Velocity field in x-direction = psi_y
        q = ky_2d * 1j * psi_h
        q = torch.fft.irfft2(q, s=(N1, N2))

        # Velocity field in y-direction = -psi_x
        v = -kx_2d * 1j * psi_h
        v = torch.fft.irfft2(v, s=(N1, N2))

        # Partial x of vorticity
        w_x = kx_2d * 1j * w_h
        w_x = torch.fft.irfft2(w_x, s=(N1, N2))

        # Partial y of vorticity
        w_y = ky_2d * 1j * w_h
        w_y = torch.fft.irfft2(w_y, s=(N1, N2))

        # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
        F_h = torch.fft.rfft2(q * w_x + v * w_y)

        # Dealias
        F_h = dealias * F_h

        w = torch.fft.irfft2(w_h, s=(N1, N2))

        if closure == True:
            if i % eval_steps == 0:
                diffusion_sample = sampler(w, sparse_condition[:, :, :, i])
            else:
                diffusion_sample = diffusion_sample + torch.randn_like(diffusion_sample) * 0.00005

            # laplacian term
            diffusion_h = torch.fft.rfft2(diffusion_sample)

            w_h = ((w_h - delta_t * F_h + delta_t * f_h + 0.5 * delta_t * diffusion_h)
                           / (1.0 + 0.5 * delta_t * visc * lap))

        if closure == False:
            w_h = ((w_h - delta_t * F_h + delta_t * f_h)
                           / (1.0 + 0.5 * delta_t * visc * lap))

        sol[..., i] = w
        sol_t[i] = t
        t += delta_t



    end_time = time.time()

    execution_time = end_time - start_time
    return sol, sol_t, execution_time

def navier_stokes_2d_nonlinear(a, w0, f, visc, sparse_condition, sampler,
                           closure = False, delta_t=1e-4, record_steps=1, eval_steps=10):
    # Grid size - must be power of 2
    N1, N2 = w0.size()[-2], w0.size()[-1]

    # Maximum frequency
    k_max = math.floor(N1 / 2.0)

    # Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)
    # Forcing to Fourier space
    f_h = torch.fft.rfft2(f)
    # If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                     torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N1, 1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0, 1)

    # Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    # Physical wavenumbers
    kx_2d = 2.0 * torch.pi * k_x / a[0]
    ky_2d = 2.0 * torch.pi * k_y / a[1]

    # Negative Laplacian in Fourier space
    lap = kx_2d ** 2 + ky_2d ** 2
    lap[0, 0] = 1.0

    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    t = 0.0

    start_time = time.time()
    for i in tqdm(range(record_steps)):
        w = torch.fft.irfft2(w_h, s=(N1, N2))

        if closure == True:
            if i % eval_steps == 0:
                nonlinear_sample = sampler(w, sparse_condition[:, :, :, i])
            else:
                nonlinear_sample = nonlinear_sample + torch.randn_like(nonlinear_sample) * 0.00005

            # convection term
            nonlinear_h = torch.fft.rfft2(nonlinear_sample)

            w_h = ((w_h  + delta_t * f_h + delta_t * nonlinear_h
                            - 0.5 * delta_t * visc * lap * w_h)
                           / (1.0 + 0.5 * delta_t * visc * lap))

        if closure == False:
            w_h = ((w_h  + delta_t * f_h
                            - 0.5 * delta_t * visc * lap * w_h)
                           / (1.0 + 0.5 * delta_t * visc * lap))

        sol[..., i] = w
        sol_t[i] = t
        t += delta_t



    end_time = time.time()

    execution_time = end_time - start_time
    return sol, sol_t, execution_time


def navier_stokes_2d_model(a, w0, f, visc, sparse_condition, sampler,
                           closure = False, delta_t=1e-4, record_steps=1, eval_steps=10):
    N1, N2 = w0.size()[-2], w0.size()[-1]
    k_max1 = math.floor(N1 / 2.0)
    k_max2 = math.floor(N1 / 2.0)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max2, step=1, device=w0.device),
                     torch.arange(start=-k_max2, end=0, step=1, device=w0.device)), 0).repeat(N1, 1).transpose(0, 1)
    # Wavenumbers in x-direction
    k_x = torch.cat((torch.arange(start=0, end=k_max1, step=1, device=w0.device),
                     torch.arange(start=-k_max1, end=0, step=1, device=w0.device)), 0).repeat(N2, 1)
    # Negative Laplacian in Fourier space
    lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
    lap[0, 0] = 1.0

    # Dealiasing mask
    dealias = torch.unsqueeze(
        torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max2,
                          torch.abs(k_x) <= (2.0 / 3.0) * k_max1).float(), 0)

    # Initial vorticity to Fourier space
    w_h = torch.fft.fftn(w0, dim=[1, 2])
    w_h = torch.stack([w_h.real, w_h.imag], dim=-1)

    # Forcing to Fourier space
    if f is not None:
        f_h = torch.fft.fftn(f, dim=[-2, -1])
        f_h = torch.stack([f_h.real, f_h.imag], dim=-1)
        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
    else:
        f_h = torch.zeros_like(w_h)

    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)

    t = 0.0

    start_time = time.time()
    for i in tqdm(range(record_steps)):
        psi_h = w_h.clone()
        psi_h[..., 0] = psi_h[..., 0] / lap
        psi_h[..., 1] = psi_h[..., 1] / lap

        # Velocity field in x-direction = psi_y
        q = psi_h.clone()
        temp = q[..., 0].clone()
        q[..., 0] = -2 * math.pi * k_y * q[..., 1]
        q[..., 1] = 2 * math.pi * k_y * temp
        q = torch.fft.ifftn(torch.view_as_complex(q), dim=[1, 2], s=(N1, N2)).real

        # Velocity field in y-direction = -psi_x
        v = psi_h.clone()
        temp = v[..., 0].clone()
        v[..., 0] = 2 * math.pi * k_x * v[..., 1]
        v[..., 1] = -2 * math.pi * k_x * temp
        v = torch.fft.ifftn(torch.view_as_complex(v), dim=[1, 2], s=(N1, N2)).real

        # Partial x of vorticity
        w_x = w_h.clone()
        temp = w_x[..., 0].clone()
        w_x[..., 0] = -2 * math.pi * k_x * w_x[..., 1]
        w_x[..., 1] = 2 * math.pi * k_x * temp
        w_x = torch.fft.ifftn(torch.view_as_complex(w_x), dim=[1, 2], s=(N1, N2)).real

        # Partial y of vorticity
        w_y = w_h.clone()
        temp = w_y[..., 0].clone()
        w_y[..., 0] = -2 * math.pi * k_y * w_y[..., 1]
        w_y[..., 1] = 2 * math.pi * k_y * temp
        w_y = torch.fft.ifftn(torch.view_as_complex(w_y), dim=[1, 2], s=(N1, N2)).real

        F_h = torch.fft.fftn(q * w_x + v * w_y, dim=[1, 2])
        F_h = torch.stack([F_h.real, F_h.imag], dim=-1)

        # Dealias
        F_h[..., 0] = dealias * F_h[..., 0]
        F_h[..., 1] = dealias * F_h[..., 1]

        w = torch.fft.ifftn(torch.view_as_complex(w_h), dim=[1, 2], s=(N1, N2)).real

        if closure == True:
            if i % eval_steps == 0:
                diffusion_sample = sampler(w, sparse_condition[:, :, :, i])
            else:
                diffusion_sample = diffusion_sample + torch.randn_like(diffusion_sample) * 0.00005

            # laplacian term
            diffusion_h = torch.fft.fftn(diffusion_sample, dim=[1, 2])
            diffusion_h = torch.stack([diffusion_h.real, diffusion_h.imag], dim=-1)

            w_h[..., 0] = ((w_h[..., 0] - delta_t * F_h[..., 0] + delta_t * f_h[..., 0] + 0.5 * delta_t * diffusion_h[..., 0])
                           / (1.0 + 0.5 * delta_t * visc * lap))
            w_h[..., 1] = ((w_h[..., 1] - delta_t * F_h[..., 1] + delta_t * f_h[..., 1] + 0.5 * delta_t * diffusion_h[..., 1])
                           / (1.0 + 0.5 * delta_t * visc * lap))

        if closure == False:
            w_h[..., 0] = ((w_h[..., 0] - delta_t * F_h[..., 0] + delta_t * f_h[..., 0])
                           / (1.0 + 0.5 * delta_t * visc * lap))
            w_h[..., 1] = ((w_h[..., 1] - delta_t * F_h[..., 1] + delta_t * f_h[..., 1])
                           / (1.0 + 0.5 * delta_t * visc * lap))

        sol[..., i] = w
        sol_t[i] = t
        t += delta_t



    end_time = time.time()

    execution_time = end_time - start_time
    return sol, sol_t, execution_time


sigma = 26
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 8
width = 20

model_interp_G = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
model_interp_G.load_state_dict(torch.load('C:\\UWMadisonResearch\\SBM_FNO_Closure\\DiffusionTerm_Generation'
                  '\\Trained_Models\\SparseDiffusionModel_Interp_v2.pth', map_location=device))

model_conv_G = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
model_conv_G.load_state_dict(torch.load('C:\\UWMadisonResearch\\SBM_FNO_Closure\\DiffusionTerm_Generation'
                  '\\Trained_Models\\SparseDiffusionModel_Conv_v2.pth', map_location=device))

model_interp_H = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
model_interp_H.load_state_dict(torch.load('C:\\UWMadisonResearch\\SBM_FNO_Closure\\DiffusionTerm_Generation'
                  '\\Trained_Models\\SparseNonlinearModel_Interp.pth', map_location=device))

model_conv_H = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
model_conv_H.load_state_dict(torch.load('C:\\UWMadisonResearch\\SBM_FNO_Closure\\DiffusionTerm_Generation'
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
vorticity_withoutG, _, t_withoutG = navier_stokes_2d_diffusion([1, 1], vorticity_init, f, nu, diffusion_target_interp,
                                            sampler, closure = False, delta_t = delta_t,
                                            record_steps = total_steps, eval_steps=5)
set_seed(13)
vorticity_withG_conv, _, t_withG = navier_stokes_2d_model([1, 1], vorticity_init, f, nu, diffusion_target_sparse_normalized,
                                            sampler = sampler_conv_G, closure = True, delta_t = delta_t,
                                            record_steps = total_steps, eval_steps=5)
set_seed(13)
vorticity_withG_interp, _, t_withG = navier_stokes_2d_diffusion([1, 1], vorticity_init, f, nu, diffusion_target_interp,
                                            sampler = sampler_interp_G, closure = True, delta_t = delta_t,
                                            record_steps = total_steps, eval_steps=5)

set_seed(13)
vorticity_withoutH, _, t_withoutH = navier_stokes_2d_nonlinear([1, 1], vorticity_init, f, nu, nonlinear_target_interp,
                                            sampler, closure = False, delta_t = delta_t,
                                            record_steps = total_steps, eval_steps=5)
set_seed(13)
vorticity_withH_conv, _, t_withH = navier_stokes_2d_nonlinear([1, 1], vorticity_init, f, nu, nonlinear_target_sparse_normalized,
                                            sampler = sampler_conv_H, closure = True, delta_t = delta_t,
                                            record_steps = total_steps, eval_steps=5)
set_seed(13)
vorticity_withH_interp, _, t_withH = navier_stokes_2d_nonlinear([1, 1], vorticity_init, f, nu, nonlinear_target_interp,
                                            sampler = sampler_interp_H, closure = True, delta_t = delta_t,
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
        generated = vorticity_withG_conv[k, :, :, j].cpu()
        generated_nog = vorticity_withoutG[k, :, :, j].cpu()
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
row_titles = [f'Truth', f'Simulation with $G$', f'Simulation w/o $G$', f'Error with $G$', f'Error w/o $G$']
for ax, row_title in zip(axs[:, 0], row_titles):
    ax.annotate(row_title, xy=(0.1, 0.5), xytext=(-50, 0),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='center', rotation=90, fontsize=fs)

plt.tight_layout()  # Adjust the subplots to fit into the figure area
plt.savefig(
    'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Plots\\Closure_G.pdf',
    dpi=600,
    bbox_inches='tight'
)




vorticity_withH_conv[..., 0] = sol[..., 0]
vorticity_withG_conv[..., 0] = sol[..., 0]

def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def energy_spectrum(phi, lx=1, ly=1, smooth=True):
    # Assuming phi is of shape (time_steps, nx, ny)
    nx, ny = phi.shape[1], phi.shape[2]
    nt = nx * ny

    phi_h = np.fft.fftn(phi, axes=(1, 2)) / nt  # Fourier transform along spatial dimensions

    energy_h = 0.5 * (phi_h * np.conj(phi_h)).real  # Spectral energy density

    k0x = 2.0 * np.pi / lx
    k0y = 2.0 * np.pi / ly
    knorm = (k0x + k0y)

    kxmax = nx // 2
    kymax = ny // 2

    wave_numbers = knorm * np.arange(0, nx)
    energy_spectrum = np.zeros(len(wave_numbers))

    for kx in range(nx):
        rkx = kx if kx <= kxmax else kx - nx
        for ky in range(ny):
            rky = ky if ky <= kymax else ky - ny
            rk = np.sqrt(rkx ** 2 + rky ** 2)
            k = int(np.round(rk))
            if k < len(wave_numbers):
                energy_spectrum[k] += np.sum(energy_h[:, kx, ky])

    energy_spectrum /= knorm

    if smooth:
        smoothed_spectrum = moving_average(energy_spectrum, 3)  # Smooth the spectrum
        # smoothed_spectrum = np.append(smoothed_spectrum, np.zeros(4))  # Append zeros to match original length after convolution
        smoothed_spectrum[:4] = np.sum(energy_h[:, :4, :4].real, axis=(0, 1, 2)) / (knorm * phi.shape[0])  # First 4 values corrected
        energy_spectrum = smoothed_spectrum

    knyquist = knorm * min(nx, ny) / 2

    return knyquist, wave_numbers, energy_spectrum

from utility import (set_seed, energy_spectrum)

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
    'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Plots\\TKE_Closure_3050.pdf',
    dpi=600,
    bbox_inches='tight'
)
