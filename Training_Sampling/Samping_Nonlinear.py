import sys
sys.path.append('C:\\UWMadisonResearch\\SBM_FNO_Closure\\Training_Sampling')
import numpy as np
import seaborn as sns
import h5py
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rc("text", usetex=True)
mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

from functools import partial
import warnings
warnings.filterwarnings("ignore")

from utility import (set_seed, energy_spectrum,
                     get_sigmas_karras, sampler,
                     mse_err, max_err, fro_err, spectral_err)
from Model_Designs import (marginal_prob_std, diffusion_coeff,FNO2d, FNO2d_NoSparse)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')

set_seed(13)
torch.set_printoptions(precision=4, sci_mode=True)
np.set_printoptions(formatter={'float_kind': lambda x: f"{x:.4e}"})

test_file = 'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Data_Generation\\test_diffusion_nonlinear.h5'
with h5py.File(test_file, 'r') as file:
    test_vorticity_64 = torch.tensor(file['test_vorticity_64'][:200], device=device)
    test_nonlinear_64 = torch.tensor(file['test_nonlinear_64'][:200], device=device)
    test_vorticity_128 = torch.tensor(file['test_vorticity_128'][:200], device=device)
    test_nonlinear_128 = torch.tensor(file['test_nonlinear_128'][:200], device=device)
    test_vorticity_256 = torch.tensor(file['test_vorticity_256'][:200], device=device)
    test_nonlinear_256 = torch.tensor(file['test_nonlinear_256'][:200], device=device)

    test_nonlinear_64_sparse_interp = torch.tensor(file['test_nonlinear_64_sparse_interp'][:200], device=device)
    test_nonlinear_128_sparse_interp = torch.tensor(file['test_nonlinear_128_sparse_interp'][:200], device=device)
    test_nonlinear_256_sparse_interp = torch.tensor(file['test_nonlinear_256_sparse_interp'][:200], device=device)
    test_nonlinear_64_sparse_normalized = torch.tensor(file['test_nonlinear_64_sparse_normalized'][:200], device=device)
    test_nonlinear_128_sparse_normalized = torch.tensor(file['test_nonlinear_128_sparse_normalized'][:200], device=device)
    test_nonlinear_256_sparse_normalized = torch.tensor(file['test_nonlinear_256_sparse_normalized'][:200], device=device)

################################################################
################# Traditional Upscale Error ####################
################################################################
mse_err_interp = mse_err(test_nonlinear_64[:100], test_nonlinear_64_sparse_interp[:100])
max_err_interp = max_err(test_nonlinear_64[:100], test_nonlinear_64_sparse_interp[:100])
fro_err_interp = fro_err(test_nonlinear_64[:100], test_nonlinear_64_sparse_interp[:100])
spectral_err_interp = spectral_err(test_nonlinear_64[:100], test_nonlinear_64_sparse_interp[:100])

mse_err_conv = mse_err(test_nonlinear_64[:100], test_nonlinear_64_sparse_normalized[:100])
max_err_conv = max_err(test_nonlinear_64[:100], test_nonlinear_64_sparse_normalized[:100])
fro_err_conv = fro_err(test_nonlinear_64[:100], test_nonlinear_64_sparse_normalized[:100])
spectral_err_conv = spectral_err(test_nonlinear_64[:100], test_nonlinear_64_sparse_normalized[:100])



sigma = 26
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 8
width = 20

sde_time_data: float = 0.5
sde_time_min = 1e-3
sde_time_max = 0.1
sample_steps = 10
sample_batch_size = 20

time_noises = get_sigmas_karras(sample_steps, sde_time_min, sde_time_max, device=device)


################################################################
################### No Sparse Generation #######################
################################################################

model_nosparse = FNO2d_NoSparse(marginal_prob_std_fn, modes, modes, width).cuda()
model_nosparse.load_state_dict(torch.load('C:\\UWMadisonResearch\\SBM_FNO_Closure\\Training_Sampling'
                  '\\Trained_Models\\SparseNonlinearModel_NoSparse.pth', map_location=device))



sampler = partial(sampler,
                    marginal_prob_std = marginal_prob_std_fn,
                    diffusion_coeff = diffusion_coeff_fn,
                    batch_size = sample_batch_size,
                    num_steps = sample_steps,
                    time_noises = time_noises,
                    device = device,
                    sparse = False)
set_seed(13)
samples_64_nosparse = sampler(test_vorticity_64[:sample_batch_size, :, :],
                              None, model_nosparse, spatial_dim=64)

### MSE and Relative Error
mse_64_nosparse = mse_err(test_nonlinear_64[:sample_batch_size, :, :], samples_64_nosparse)
max_64_nosparse = max_err(test_nonlinear_64[:sample_batch_size, :, :], samples_64_nosparse)
fro_64_nosparse = fro_err(test_nonlinear_64[:sample_batch_size, :, :], samples_64_nosparse)
spec_64_nosparse = spectral_err(test_nonlinear_64[:sample_batch_size, :, :], samples_64_nosparse)

### Plot and save
set_seed(13)

data1 = test_nonlinear_64[:sample_batch_size, :, :].cpu()
data2 = samples_64_nosparse.cpu()
data3 = np.abs(data1 - data2)

# Initialize the plot with 4 rows and 4 columns
fig, axs = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
fs = 28
plt.rcParams.update({'font.size': fs})

# Define tick positions and labels
def create_ticks_labels(size, step=20):
    ticks = np.arange(0, size, step * size / 64)
    tick_labels = [str(int(tick)) for tick in ticks]
    return ticks, tick_labels

ticks_1, tick_labels_1 = create_ticks_labels(data1.shape[1])
ticks_2, tick_labels_2 = create_ticks_labels(data2.shape[1])
ticks_3, tick_labels_3 = create_ticks_labels(data3.shape[1])

# Randomly sample indices equal to the number of columns (4) for clarity
indices = [torch.randint(0, data1.shape[0], (1,)).item() for _ in range(4)]

# Define color scale parameters
max_val = 0.9
min_val = -0.9
err_max = 0.3
err_min = 0
cbar_ticks = np.linspace(min_val, max_val, 6)
cbar_ticks_err = np.linspace(err_min, err_max, 6)
cbar_ticks_contour = np.linspace(err_min, err_max, 6)

# Plot heatmaps and contour plots
for i, idx in enumerate(indices):
    j = i % 4  # Column index

    # --- Row 1: Truth Heatmap ---
    truth = data1[idx, ...].cpu().numpy()
    sns.heatmap(
        truth,
        ax=axs[0, j],
        cmap='rocket',
        cbar=(j == 3),  # Show colorbar only on the last column
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )
    axs[0, j].set_title(r"\text{Truth }" + str(j + 1))
    axs[0, j].set_xticks(ticks_1)
    axs[0, j].set_yticks(ticks_1)
    axs[0, j].set_xticklabels(tick_labels_1, rotation=0)
    axs[0, j].set_yticklabels(tick_labels_1, rotation=0)
    axs[0, j].invert_yaxis()

    # --- Row 2: Generated Heatmap ---
    generated = data2[idx, ...].cpu().numpy()
    sns.heatmap(
        generated,
        ax=axs[1, j],
        cmap='rocket',
        cbar=(j == 3),
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )
    axs[1, j].set_title(r"\text{Generated }" + str(j + 1))
    axs[1, j].set_xticks(ticks_2)
    axs[1, j].set_yticks(ticks_2)
    axs[1, j].set_xticklabels(tick_labels_2, rotation=0)
    axs[1, j].set_yticklabels(tick_labels_2, rotation=0)
    axs[1, j].invert_yaxis()

    # --- Row 3: Error Heatmap ---
    error = data3[idx, ...].cpu().numpy()
    ax_contour = axs[2, j]
    # Define the grid coordinates
    S = error.shape[0]
    x = np.arange(S)
    y = np.arange(S)
    X, Y = np.meshgrid(x, y)

    # Create filled contour plot using matplotlib
    contour = ax_contour.contourf(
        X, Y, error,
        levels=cbar_ticks_contour,  # Six levels to match cbar_ticks_err
        cmap='rocket',
        vmin=err_min,
        vmax=err_max
    )

    # Add colorbar only on the last column
    if j == 3:
        cbar_contour = fig.colorbar(
            contour,
            ax=ax_contour,
            format='%.2f',
        )

    ax_contour.set_title(r"\text{Error Contour }" + str(j + 1))
    ax_contour.set_xticks(ticks_3)
    ax_contour.set_yticks(ticks_3)
    ax_contour.set_xticklabels(tick_labels_3, rotation=0)
    ax_contour.set_yticklabels(tick_labels_3, rotation=0)

# Adjust tick parameters for all axes
for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=fs)

# Adjust layout and save the plot
plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.5)
plt.savefig(
    'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Plots\\nonlinearModelWithoutSparse.png',
    dpi=300,
    bbox_inches='tight'
)



################################################################
####################### Interpolation Model ####################
################################################################
model_interp = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
model_interp.load_state_dict(torch.load('C:\\UWMadisonResearch\\SBM_FNO_Closure\\Training_Sampling'
                  '\\Trained_Models\\SparseNonlinearModel_Interp.pth', map_location=device))

sampler = partial(sampler,
                    marginal_prob_std = marginal_prob_std_fn,
                    diffusion_coeff = diffusion_coeff_fn,
                    batch_size = sample_batch_size,
                    num_steps = sample_steps,
                    time_noises = time_noises,
                    device = device,
                    sparse = True)
set_seed(13)
samples_64_interp = sampler(test_vorticity_64[:sample_batch_size, :, :],
                            test_nonlinear_64_sparse_interp[:sample_batch_size, :, :], model_interp, spatial_dim=64)
samples_128_interp = sampler(test_vorticity_128[:sample_batch_size, :, :],
                             test_nonlinear_128_sparse_interp[:sample_batch_size, :, :], model_interp, spatial_dim=128)
samples_256_interp = sampler(test_vorticity_256[:sample_batch_size, :, :],
                             test_nonlinear_256_sparse_interp[:sample_batch_size, :, :], model_interp, spatial_dim=256)

### MSE and Relative Error
mse_64_interp = mse_err(test_nonlinear_64[:sample_batch_size, :, :], samples_64_interp)
max_64_interp = max_err(test_nonlinear_64[:sample_batch_size, :, :], samples_64_interp)
fro_64_interp = fro_err(test_nonlinear_64[:sample_batch_size, :, :], samples_64_interp)
spec_64_interp = spectral_err(test_nonlinear_64[:sample_batch_size, :, :], samples_64_interp)

mse_128_interp = mse_err(test_nonlinear_128[:sample_batch_size, :, :], samples_128_interp)
max_128_interp = max_err(test_nonlinear_128[:sample_batch_size, :, :], samples_128_interp)
fro_128_interp = fro_err(test_nonlinear_128[:sample_batch_size, :, :], samples_128_interp)
spec_128_interp = spectral_err(test_nonlinear_128[:sample_batch_size, :, :], samples_128_interp)

mse_256_interp = mse_err(test_nonlinear_256[:sample_batch_size, :, :], samples_256_interp)
max_256_interp = max_err(test_nonlinear_256[:sample_batch_size, :, :], samples_256_interp)
fro_256_interp = fro_err(test_nonlinear_256[:sample_batch_size, :, :], samples_256_interp)
spec_256_interp = spectral_err(test_nonlinear_256[:sample_batch_size, :, :], samples_256_interp)

print(f"MSE 64 Interp: {mse_64_interp:.8f}, Max Error 64 Interp: {max_64_interp:.4f}, "
      f"Frobenius Error 64 Interp: {fro_64_interp:.4f}, Spectral Error 64 Interp: {spec_64_interp:.4f}")
print(f"MSE 128 Interp: {mse_128_interp:.8f}, Max Error 128 Interp: {max_128_interp:.4f}, "
        f"Frobenius Error 128 Interp: {fro_128_interp:.4f}, Spectral Error 128 Interp: {spec_128_interp:.4f}")
print(f"MSE 256 Interp: {mse_256_interp:.8f}, Max Error 256 Interp: {max_256_interp:.4f}, "
        f"Frobenius Error 256 Interp: {fro_256_interp:.4f}, Spectral Error 256 Interp: {spec_256_interp:.4f}")



### Plot and save
set_seed(13)

data1 = test_nonlinear_64[:sample_batch_size, :, :].cpu()
data2 = samples_64_interp.cpu()
data3 = np.abs(data1 - data2)

# Initialize the plot with 4 rows and 4 columns
fig, axs = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
plt.rcParams.update({'font.size': fs})

ticks_1, tick_labels_1 = create_ticks_labels(data1.shape[1])
ticks_2, tick_labels_2 = create_ticks_labels(data2.shape[1])
ticks_3, tick_labels_3 = create_ticks_labels(data3.shape[1])

# Randomly sample indices equal to the number of columns (4) for clarity
indices = [torch.randint(0, data1.shape[0], (1,)).item() for _ in range(4)]

# Define color scale parameters
max_val = 0.9
min_val = -0.9
err_max = 0.15
err_min = 0
cbar_ticks = np.linspace(min_val, max_val, 6)
cbar_ticks_err = np.linspace(err_min, err_max, 6)
cbar_ticks_contour = np.linspace(err_min, err_max, 6)

# Plot heatmaps and contour plots
for i, idx in enumerate(indices):
    j = i % 4  # Column index

    # --- Row 1: Truth Heatmap ---
    truth = data1[idx, ...].cpu().numpy()
    sns.heatmap(
        truth,
        ax=axs[0, j],
        cmap='rocket',
        cbar=(j == 3),  # Show colorbar only on the last column
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )
    axs[0, j].set_title(r"\text{Truth }" + str(j + 1))
    axs[0, j].set_xticks(ticks_1)
    axs[0, j].set_yticks(ticks_1)
    axs[0, j].set_xticklabels(tick_labels_1, rotation=0)
    axs[0, j].set_yticklabels(tick_labels_1, rotation=0)
    axs[0, j].invert_yaxis()

    # --- Row 2: Generated Heatmap ---
    generated = data2[idx, ...].cpu().numpy()
    sns.heatmap(
        generated,
        ax=axs[1, j],
        cmap='rocket',
        cbar=(j == 3),
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )
    axs[1, j].set_title(r"\text{Generated }" + str(j + 1))
    axs[1, j].set_xticks(ticks_2)
    axs[1, j].set_yticks(ticks_2)
    axs[1, j].set_xticklabels(tick_labels_2, rotation=0)
    axs[1, j].set_yticklabels(tick_labels_2, rotation=0)
    axs[1, j].invert_yaxis()

    # --- Row 3: Error Heatmap ---
    error = data3[idx, ...].cpu().numpy()
    ax_contour = axs[2, j]
    # Define the grid coordinates
    S = error.shape[0]
    x = np.arange(S)
    y = np.arange(S)
    X, Y = np.meshgrid(x, y)

    # Create filled contour plot using matplotlib
    contour = ax_contour.contourf(
        X, Y, error,
        levels=cbar_ticks_contour,  # Six levels to match cbar_ticks_err
        cmap='rocket',
        vmin=err_min,
        vmax=err_max
    )

    # Add colorbar only on the last column
    if j == 3:
        cbar_contour = fig.colorbar(
            contour,
            ax=ax_contour,
            format='%.2f',
        )

    ax_contour.set_title(r"\text{Error Contour }" + str(j + 1))
    ax_contour.set_xticks(ticks_3)
    ax_contour.set_yticks(ticks_3)
    ax_contour.set_xticklabels(tick_labels_3, rotation=0)
    ax_contour.set_yticklabels(tick_labels_3, rotation=0)

# Adjust tick parameters for all axes
for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=fs)

# Adjust layout and save the plot
plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.5)
plt.savefig(
    'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Plots\\H_test_64_interp.png',
    dpi=300,
    bbox_inches='tight'
)


### Plot and save
set_seed(13)

data1 = test_nonlinear_128[:sample_batch_size, :, :].cpu()
data2 = samples_128_interp.cpu()
data3 = samples_256_interp.cpu()

# Initialize the plot with 4 rows and 4 columns
fig, axs = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
plt.rcParams.update({'font.size': fs})

ticks_1, tick_labels_1 = create_ticks_labels(data1.shape[1])
ticks_2, tick_labels_2 = create_ticks_labels(data2.shape[1])
ticks_3, tick_labels_3 = create_ticks_labels(data3.shape[1])

# Randomly sample indices equal to the number of columns (4) for clarity
indices = [torch.randint(0, data1.shape[0], (1,)).item() for _ in range(4)]

# Define color scale parameters
max_val = 0.9
min_val = -0.9
err_max = 0.05
err_min = 0
cbar_ticks = np.linspace(min_val, max_val, 6)
cbar_ticks_err = np.linspace(err_min, err_max, 6)
cbar_ticks_contour = np.linspace(err_min, err_max, 6)

# Plot heatmaps and contour plots
for i, idx in enumerate(indices):
    j = i % 4  # Column index

    # --- Row 1: Truth Heatmap ---
    truth = data1[idx, ...].cpu().numpy()
    sns.heatmap(
        truth,
        ax=axs[0, j],
        cmap='rocket',
        cbar=(j == 3),  # Show colorbar only on the last column
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )
    axs[0, j].set_title(r"\text{Truth }" + str(j + 1))
    axs[0, j].set_xticks(ticks_1)
    axs[0, j].set_yticks(ticks_1)
    axs[0, j].set_xticklabels(tick_labels_1, rotation=0)
    axs[0, j].set_yticklabels(tick_labels_1, rotation=0)
    axs[0, j].invert_yaxis()

    # --- Row 2: Generated Heatmap ---
    generated = data2[idx, ...].cpu().numpy()
    sns.heatmap(
        generated,
        ax=axs[1, j],
        cmap='rocket',
        cbar=(j == 3),
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )
    axs[1, j].set_title(r"\text{Generated }" + str(j + 1))
    axs[1, j].set_xticks(ticks_2)
    axs[1, j].set_yticks(ticks_2)
    axs[1, j].set_xticklabels(tick_labels_2, rotation=0)
    axs[1, j].set_yticklabels(tick_labels_2, rotation=0)
    axs[1, j].invert_yaxis()

    # --- Row 3: Error Heatmap ---
    error = data3[idx, ...].cpu().numpy()
    sns.heatmap(
        error,
        ax=axs[2, j],
        cmap='rocket',
        cbar=(j == 3),
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )
    axs[2, j].set_title(r"\text{Generated }" + str(j + 1))
    axs[2, j].set_xticks(ticks_3)
    axs[2, j].set_yticks(ticks_3)
    axs[2, j].set_xticklabels(tick_labels_3, rotation=0)
    axs[2, j].set_yticklabels(tick_labels_3, rotation=0)
    axs[2, j].invert_yaxis()

# Adjust tick parameters for all axes
for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=fs)

# Adjust layout and save the plot
plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.5)
plt.savefig(
    'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Plots\\H_test_interp.png',
    dpi=300,
    bbox_inches='tight'
)

















################################################################
######################## Convolution Model #####################
################################################################
model_conv = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
model_conv.load_state_dict(torch.load('C:\\UWMadisonResearch\\SBM_FNO_Closure\\Training_Sampling'
                  '\\Trained_Models\\SparseNonlinearModel_Conv.pth', map_location=device))

sampler = partial(sampler,
                    marginal_prob_std = marginal_prob_std_fn,
                    diffusion_coeff = diffusion_coeff_fn,
                    batch_size = sample_batch_size,
                    num_steps = sample_steps,
                    time_noises = time_noises,
                    device = device,
                    sparse = True)

set_seed(13)
samples_64_conv = sampler(test_vorticity_64[:sample_batch_size, :, :],
                          test_nonlinear_64_sparse_normalized[:sample_batch_size, :, :], model_conv, spatial_dim=64)
samples_128_conv = sampler(test_vorticity_128[:sample_batch_size, :, :],
                           test_nonlinear_128_sparse_normalized[:sample_batch_size, :, :], model_conv, spatial_dim=128)
samples_256_conv = sampler(test_vorticity_256[:sample_batch_size, :, :],
                           test_nonlinear_256_sparse_normalized[:sample_batch_size, :, :], model_conv, spatial_dim=256)



### MSE and Relative Error
mse_64_conv = mse_err(test_nonlinear_64[:sample_batch_size, :, :], samples_64_conv)
max_64_conv = max_err(test_nonlinear_64[:sample_batch_size, :, :], samples_64_conv)
fro_64_conv = fro_err(test_nonlinear_64[:sample_batch_size, :, :], samples_64_conv)
spec_64_conv = spectral_err(test_nonlinear_64[:sample_batch_size, :, :], samples_64_conv)

mse_128_conv = mse_err(test_nonlinear_128[:sample_batch_size, :, :], samples_128_conv)
max_128_conv = max_err(test_nonlinear_128[:sample_batch_size, :, :], samples_128_conv)
fro_128_conv = fro_err(test_nonlinear_128[:sample_batch_size, :, :], samples_128_conv)
spec_128_conv = spectral_err(test_nonlinear_128[:sample_batch_size, :, :], samples_128_conv)

mse_256_conv = mse_err(test_nonlinear_256[:sample_batch_size, :, :], samples_256_conv)
max_256_conv = max_err(test_nonlinear_256[:sample_batch_size, :, :], samples_256_conv)
fro_256_conv = fro_err(test_nonlinear_256[:sample_batch_size, :, :], samples_256_conv)
spec_256_conv = spectral_err(test_nonlinear_256[:sample_batch_size, :, :], samples_256_conv)

print(f"MSE 64 Conv: {mse_64_conv:.8f}, Max Error 64 Conv: {max_64_conv:.4f}, "
        f"Frobenius Error 64 Conv: {fro_64_conv:.4f}, Spectral Error 64 Conv: {spec_64_conv:.4f}")
print(f"MSE 128 Conv: {mse_128_conv:.8f}, Max Error 128 Conv: {max_128_conv:.4f}, "
        f"Frobenius Error 128 Conv: {fro_128_conv:.4f}, Spectral Error 128 Conv: {spec_128_conv:.4f}")
print(f"MSE 256 Conv: {mse_256_conv:.8f}, Max Error 256 Conv: {max_256_conv:.4f}, "
        f"Frobenius Error 256 Conv: {fro_256_conv:.4f}, Spectral Error 256 Conv: {spec_256_conv:.4f}")




### Plot and save
set_seed(13)

data1 = test_nonlinear_64[:sample_batch_size, :, :].cpu()
data2 = samples_64_conv.cpu()
data3 = np.abs(data1 - data2)

# Initialize the plot with 4 rows and 4 columns
fig, axs = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
plt.rcParams.update({'font.size': fs})

ticks_1, tick_labels_1 = create_ticks_labels(data1.shape[1])
ticks_2, tick_labels_2 = create_ticks_labels(data2.shape[1])
ticks_3, tick_labels_3 = create_ticks_labels(data3.shape[1])

# Randomly sample indices equal to the number of columns (4) for clarity
indices = [torch.randint(0, data1.shape[0], (1,)).item() for _ in range(4)]

# Define color scale parameters
max_val = 0.9
min_val = -0.9
err_max = 0.15
err_min = 0
cbar_ticks = np.linspace(min_val, max_val, 6)
cbar_ticks_err = np.linspace(err_min, err_max, 6)
cbar_ticks_contour = np.linspace(err_min, err_max, 6)

# Plot heatmaps and contour plots
for i, idx in enumerate(indices):
    j = i % 4  # Column index

    # --- Row 1: Truth Heatmap ---
    truth = data1[idx, ...].cpu().numpy()
    sns.heatmap(
        truth,
        ax=axs[0, j],
        cmap='rocket',
        cbar=(j == 3),  # Show colorbar only on the last column
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )
    axs[0, j].set_title(r"\text{Truth }" + str(j + 1))
    axs[0, j].set_xticks(ticks_1)
    axs[0, j].set_yticks(ticks_1)
    axs[0, j].set_xticklabels(tick_labels_1, rotation=0)
    axs[0, j].set_yticklabels(tick_labels_1, rotation=0)
    axs[0, j].invert_yaxis()

    # --- Row 2: Generated Heatmap ---
    generated = data2[idx, ...].cpu().numpy()
    sns.heatmap(
        generated,
        ax=axs[1, j],
        cmap='rocket',
        cbar=(j == 3),
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )
    axs[1, j].set_title(r"\text{Generated }" + str(j + 1))
    axs[1, j].set_xticks(ticks_2)
    axs[1, j].set_yticks(ticks_2)
    axs[1, j].set_xticklabels(tick_labels_2, rotation=0)
    axs[1, j].set_yticklabels(tick_labels_2, rotation=0)
    axs[1, j].invert_yaxis()

    # --- Row 3: Error Heatmap ---
    error = data3[idx, ...].cpu().numpy()
    ax_contour = axs[2, j]
    # Define the grid coordinates
    S = error.shape[0]
    x = np.arange(S)
    y = np.arange(S)
    X, Y = np.meshgrid(x, y)

    # Create filled contour plot using matplotlib
    contour = ax_contour.contourf(
        X, Y, error,
        levels=cbar_ticks_contour,  # Six levels to match cbar_ticks_err
        cmap='rocket',
        vmin=err_min,
        vmax=err_max
    )

    # Add colorbar only on the last column
    if j == 3:
        cbar_contour = fig.colorbar(
            contour,
            ax=ax_contour,
            format='%.2f',
        )

    ax_contour.set_title(r"\text{Error Contour }" + str(j + 1))
    ax_contour.set_xticks(ticks_3)
    ax_contour.set_yticks(ticks_3)
    ax_contour.set_xticklabels(tick_labels_3, rotation=0)
    ax_contour.set_yticklabels(tick_labels_3, rotation=0)

# Adjust tick parameters for all axes
for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=fs)

# Adjust layout and save the plot
plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.5)
plt.savefig(
    'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Plots\\H_test_64_conv.png',
    dpi=300,
    bbox_inches='tight'
)








### Plot and save
set_seed(13)

data1 = test_nonlinear_128[:sample_batch_size, :, :].cpu()
data2 = samples_128_conv.cpu()
data3 = samples_256_conv.cpu()

# Initialize the plot with 4 rows and 4 columns
fig, axs = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
plt.rcParams.update({'font.size': fs})

ticks_1, tick_labels_1 = create_ticks_labels(data1.shape[1])
ticks_2, tick_labels_2 = create_ticks_labels(data2.shape[1])
ticks_3, tick_labels_3 = create_ticks_labels(data3.shape[1])

# Randomly sample indices equal to the number of columns (4) for clarity
indices = [torch.randint(0, data1.shape[0], (1,)).item() for _ in range(4)]

# Define color scale parameters
max_val = 0.9
min_val = -0.9
err_max = 0.05
err_min = 0
cbar_ticks = np.linspace(min_val, max_val, 6)
cbar_ticks_err = np.linspace(err_min, err_max, 6)
cbar_ticks_contour = np.linspace(err_min, err_max, 6)

# Plot heatmaps and contour plots
for i, idx in enumerate(indices):
    j = i % 4  # Column index

    # --- Row 1: Truth Heatmap ---
    truth = data1[idx, ...].cpu().numpy()
    sns.heatmap(
        truth,
        ax=axs[0, j],
        cmap='rocket',
        cbar=(j == 3),  # Show colorbar only on the last column
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )
    axs[0, j].set_title(r"\text{Truth }" + str(j + 1))
    axs[0, j].set_xticks(ticks_1)
    axs[0, j].set_yticks(ticks_1)
    axs[0, j].set_xticklabels(tick_labels_1, rotation=0)
    axs[0, j].set_yticklabels(tick_labels_1, rotation=0)
    axs[0, j].invert_yaxis()

    # --- Row 2: Generated Heatmap ---
    generated = data2[idx, ...].cpu().numpy()
    sns.heatmap(
        generated,
        ax=axs[1, j],
        cmap='rocket',
        cbar=(j == 3),
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )
    axs[1, j].set_title(r"\text{Generated }" + str(j + 1))
    axs[1, j].set_xticks(ticks_2)
    axs[1, j].set_yticks(ticks_2)
    axs[1, j].set_xticklabels(tick_labels_2, rotation=0)
    axs[1, j].set_yticklabels(tick_labels_2, rotation=0)
    axs[1, j].invert_yaxis()

    # --- Row 3: Error Heatmap ---
    error = data3[idx, ...].cpu().numpy()
    sns.heatmap(
        error,
        ax=axs[2, j],
        cmap='rocket',
        cbar=(j == 3),
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )
    axs[2, j].set_title(r"\text{Generated }" + str(j + 1))
    axs[2, j].set_xticks(ticks_3)
    axs[2, j].set_yticks(ticks_3)
    axs[2, j].set_xticklabels(tick_labels_3, rotation=0)
    axs[2, j].set_yticklabels(tick_labels_3, rotation=0)
    axs[2, j].invert_yaxis()

# Adjust tick parameters for all axes
for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=fs)

# Adjust layout and save the plot
plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.5)
plt.savefig(
    'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Plots\\H_test_conv.png',
    dpi=300,
    bbox_inches='tight'
)




smooth = False
_, k64_conv, E64_conv = energy_spectrum(samples_64_conv.cpu(), 1, 1, smooth=smooth)
_, k128_conv, E128_conv = energy_spectrum(samples_128_conv.cpu(), 1, 1, smooth=smooth)
_, k256_conv, E256_conv = energy_spectrum(samples_256_conv.cpu(), 1, 1, smooth=smooth)

_, k64_interp, E64_interp = energy_spectrum(samples_64_interp.cpu(), 1, 1, smooth=smooth)
_, k128_interp, E128_interp = energy_spectrum(samples_128_interp.cpu(), 1, 1, smooth=smooth)
_, k256_interp, E256_interp = energy_spectrum(samples_256_interp.cpu(), 1, 1, smooth=smooth)

_, k64_truth, E64_truth = energy_spectrum(test_nonlinear_64[:sample_batch_size].cpu(), 1, 1, smooth=smooth)
_, k128_truth, E128_truth = energy_spectrum(test_nonlinear_128[:sample_batch_size].cpu(), 1, 1, smooth=smooth)
_, k256_truth, E256_truth = energy_spectrum(test_nonlinear_256[:sample_batch_size].cpu(), 1, 1, smooth=smooth)

resolutions = [64, 128, 256]
conv_kn = [k64_conv, k128_conv, k256_conv]
conv_E = [E64_conv, E128_conv, E256_conv]

interp_kn = [k64_interp, k128_interp, k256_interp]
interp_E = [E64_interp, E128_interp, E256_interp]

truth_kn = [k64_truth, k128_truth, k256_truth]
truth_E = [E64_truth, E128_truth, E256_truth]

fig, axs = plt.subplots(1, 3, figsize=(42, 12), sharey=False, gridspec_kw={'width_ratios': [1, 1, 1]})
fs = 64
plt.rcParams.update({'font.size': fs})
plt.rcParams.update({'legend.fontsize': fs})

for i, res in enumerate(resolutions):
    # Upper row plots
    col = i
    axs[col].loglog(truth_kn[i][:-1], truth_E[i][:-1], label=r'\text{Truth}', linestyle='-.', linewidth=6)
    axs[col].loglog(interp_kn[i][:-1], interp_E[i][:-1], label=r'\text{Interpolation}', linestyle=':', linewidth=6)
    axs[col].loglog(conv_kn[i][:-1], conv_E[i][:-1], label=r'\text{Convolution}', linestyle='--', linewidth=6)
    axs[col].set_xscale('log')
    axs[col].set_yscale('log')
    axs[col].set_title(f'Energy Spectrum of $H$ ({res}x{res})', fontsize = fs)
    axs[col].set_xlabel(f'Wave number ($k$)', fontsize = fs)

axs[0].set_ylabel(f'Energy ($E$)', fontsize=fs)

for i, ax in enumerate(axs.flat):
    ax.tick_params(axis='both', which='major', length=14, width=2, labelsize=fs)
    ax.tick_params(axis='both', which='minor', length=7, width=2)
    ax.tick_params(axis='x', which='major', pad=12)
    ax.set_ylim(10**-21, 10**0)
    ax.set_xlim(1, 2 * 1e3)
    ticks = [10 ** -21, 10 ** -14, 10 ** -7, 10 ** 0]

    # Define corresponding labels (the first label is for the lowest tick, etc.)
    labels = ['$10^{-21}$', '$10^{-14}$', '$10^{-7}$', '$10^{0}$']

    # Apply the tick positions and labels
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)


    # ax.set_xscale('log')
    # ax.set_yscale('log')


    for spine in ax.spines.values():
        spine.set_linewidth(2)
    # For subplots other than the left-most one, remove y-axis ticks and labels.
    if i != 0:
        ax.tick_params(left=False, labelleft=False)

# Create a shared legend
handles, labels = axs[0].get_legend_handles_labels()
lege = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=fs, bbox_to_anchor=(0.5, 1),
                  fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(2)
# Adjust the layout to make space for the legend
plt.subplots_adjust(top=0.85)
plt.tight_layout(rect=[0, 0, 1, 0.88])
plt.savefig(
    'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Plots\\TKECompare_H_temp.png',
    dpi=300,
    bbox_inches='tight'
)
