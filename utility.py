import numpy as np
import torch
import torch.nn.functional as F
import random
import os

################################
##### Data Preprosessing #######
################################
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

################################
########### Sampling ###########
################################
def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])
def get_sigmas_karras(n, time_min, time_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = time_min ** (1 / rho)
    max_inv_rho = time_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def sampler(vorticity_condition,
           sparse_data,
           score_model,
           spatial_dim,
           marginal_prob_std,
           diffusion_coeff,
           batch_size,
           num_steps,
           time_noises,
           device,
            sparse=True):
    t = torch.ones(batch_size, device=device) * time_noises[0]
    init_x = torch.randn(batch_size, spatial_dim, spatial_dim, device=device) * marginal_prob_std(t)[:, None, None]
    x = init_x

    with (torch.no_grad()):
        for i in range(num_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_noises[i]
            step_size = time_noises[i] - time_noises[i + 1]
            g = diffusion_coeff(batch_time_step)
            if sparse:
                grad = score_model(batch_time_step, x, vorticity_condition, sparse_data)
            else:
                grad = score_model(batch_time_step, x, vorticity_condition)
            mean_x = x + (g ** 2)[:, None, None] * grad * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None] * torch.randn_like(x)

    return mean_x


################################
####### Energy Spectrum ########
################################
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
    knorm = (k0x + k0y) / 2.0

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

################################
########### Metrics ############
################################
def mse_err(data1, data2):
    return torch.mean((data1 - data2) ** 2)

def max_err(data1, data2):
    error_max = torch.amax(torch.abs(data1 - data2), dim=(1, 2))
    return torch.mean(error_max / torch.amax(torch.abs(data1), dim=(1, 2)))

def fro_err(data1, data2):
    error_fro = torch.linalg.matrix_norm(data1 - data2, 'fro', dim=(1, 2))
    return torch.mean(error_fro / torch.linalg.matrix_norm(data1, 'fro', dim=(1, 2)))

def spectral_err(data1, data2):
    error_spec = torch.linalg.matrix_norm(data1 - data2, 2, dim=(1, 2))
    return torch.mean(error_spec / torch.linalg.matrix_norm(data1, 2, dim=(1, 2)))

# Sparse information for interpolation conditioning
def interpolate2d(data, sparse_res):
    down_factor = data.shape[-1] // sparse_res
    data_sparse = data[:, ::down_factor, ::down_factor]
    interp_res = data.shape[-1] + 1 - down_factor
    data_sparse_interp = F.interpolate(data_sparse.unsqueeze(1),
                                       size=(interp_res, interp_res), mode='bicubic')
    data_sparse_interp = F.pad(data_sparse_interp, (0, down_factor-1, 0, down_factor-1), mode='replicate')
    return data_sparse_interp.squeeze(1)