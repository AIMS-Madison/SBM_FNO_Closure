import numpy as np
import pyfftw
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
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

def spectrum_2d(signal, n_observations, normalize=True):
    """This function computes the spectrum of a 2D signal using the Fast Fourier Transform (FFT).

    Paramaters
    ----------
    signal : a tensor of shape (T * n_observations * n_observations)
        A 2D discretized signal represented as a 1D tensor with shape
        (T * n_observations * n_observations), where T is the number of time
        steps and n_observations is the spatial size of the signal.

        T can be any number of channels that we reshape into and
        n_observations * n_observations is the spatial resolution.
    n_observations: an integer
        Number of discretized points. Basically the resolution of the signal.

    Returns
    --------
    spectrum: a tensor
        A 1D tensor of shape (s,) representing the computed spectrum.
    """
    T = signal.shape[0]
    signal = signal.view(T, n_observations, n_observations)

    if normalize:
        signal = torch.fft.fft2(signal)
    else:
        signal = torch.fft.rfft2(
            signal, s=(n_observations, n_observations), norm="backward"
        )

    # 2d wavenumbers following PyTorch fft convention
    k_max = n_observations // 2
    wavenumers = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1),
            torch.arange(start=-k_max, end=0, step=1),
        ),
        0,
    ).repeat(n_observations, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers

    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k

    # Remove symmetric components from wavenumbers
    index = -1.0 * torch.ones((n_observations, n_observations))
    k_max1 = k_max + 1
    index[0:k_max1, 0:k_max1] = sum_k[0:k_max1, 0:k_max1]

    spectrum = torch.zeros((T, n_observations))
    for j in range(1, n_observations + 1):
        ind = torch.where(index == j)
        spectrum[:, j - 1] = (signal[:, ind[0], ind[1]].sum(dim=1)).abs() ** 2

    spectrum = spectrum.mean(dim=0)

    k_bins = torch.range(1, n_observations) * 2 * np.pi
    return k_bins, spectrum

def energy_spectrum_t(nx,ny,w):
    epsilon = 1.0e-6

    dx = 2.0*np.pi/np.float64(nx)
    dy = 2.0*np.pi/np.float64(ny)

    kx = np.empty(nx)
    ky = np.empty(ny)

    kx[0:int(nx/2)] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(0,int(nx/2)))
    kx[int(nx/2):nx] = 2*np.pi/(np.float64(nx)*dx)*np.float64(np.arange(-int(nx/2),0))

    ky[0:ny] = kx[0:ny]

    kx[0] = epsilon
    ky[0] = epsilon

    kx, ky = np.meshgrid(kx, ky, indexing='ij')

    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    wf = fft_object(w)

    es =  np.empty((nx,ny))

    kk = np.sqrt(kx[:,:]**2 + ky[:,:]**2)
    es[:,:] = np.pi*((np.abs(wf[:,:])/(nx*ny))**2)/kk

    n = int(np.sqrt(nx*nx + ny*ny)/2.0)-1

    en = np.zeros(n+1)

    for k in range(1,n+1):
        en[k] = 0.0
        ic = 0
        ii,jj = np.where((kk[1:,1:]>(k-0.5)) & (kk[1:,1:]<(k+0.5)))
        ic = ii.size
        ii = ii+1
        jj = jj+1
        en[k] = np.sum(es[ii,jj])
        en[k] = en[k]/ic
    kn = np.linspace(1, n, n) * 2 * np.pi
    return kn, en[1:]


################################
########### Plotting ###########
################################

def plot_heatmaps_sample(data1, data2, data3, sample_batch_size=4, seed=12):
    set_seed(seed)

    # Initialize the plot
    fig, axs = plt.subplots(3, sample_batch_size, figsize=(16, 12), constrained_layout=True)
    plt.rcParams.update({'font.size': 14})

    # Ticks setting
    ticks_1 = np.arange(0, data1.shape[1], 10 * data1.shape[1] / 64)
    tick_labels_1 = [str(int(tick)) for tick in ticks_1]

    ticks_2 = np.arange(0, data2.shape[1], 10 * data2.shape[1] / 64)
    tick_labels_2 = [str(int(tick)) for tick in ticks_2]

    ticks_3 = np.arange(0, data3.shape[1], 10 * data3.shape[1] / 64)
    tick_labels_3 = [str(int(tick)) for tick in ticks_3]

    # Randomly sample indices
    indices = [torch.randint(0, data1.shape[0], (1,)).item() for _ in range(sample_batch_size)]

    # Variables to store global min/max for consistent coloring
    min_val, max_val = np.inf, -np.inf

    # Compute global min/max values
    for idx in indices:
        generated_64 = data1[idx, ...].cpu().numpy()
        generated_128 = data2[idx, ...].cpu().numpy()
        generated_256 = data3[idx, ...].cpu().numpy()
        min_val = min(min_val, generated_64.min(), generated_128.min(), generated_256.min())
        max_val = max(max_val, generated_64.max(), generated_128.max(), generated_256.max())

    # Plot heatmaps
    for i, idx in enumerate(indices):
        j = i % sample_batch_size

        # Truth plot
        generated_64 = data1[idx, ...].cpu().numpy()
        sns.heatmap(generated_64, ax=axs[0, j], cmap='rocket', cbar=(i % sample_batch_size == sample_batch_size - 1),
                    vmin=min_val, vmax=max_val)
        axs[0, j].set_title(r"$Generated$ " + str(j + 1))
        axs[0, j].set_xticks(ticks_1)
        axs[0, j].set_yticks(ticks_1)
        axs[0, j].set_xticklabels(tick_labels_1, rotation=0)
        axs[0, j].set_yticklabels(tick_labels_1, rotation=0)
        axs[0, j].invert_yaxis()

        # Generated plot
        generated_128 = data2[idx, ...].cpu().numpy()
        sns.heatmap(generated_128, ax=axs[1, j], cmap='rocket', cbar=(i % sample_batch_size == sample_batch_size - 1),
                    vmin=min_val, vmax=max_val)
        axs[1, j].set_title(r"$Generated$ " + str(j + 1))
        axs[1, j].set_xticks(ticks_2)
        axs[1, j].set_yticks(ticks_2)
        axs[1, j].set_xticklabels(tick_labels_2, rotation=0)
        axs[1, j].set_yticklabels(tick_labels_2)
        axs[1, j].invert_yaxis()

        # Error plot
        generated_256 = data3[idx, ...].cpu().numpy()
        sns.heatmap(generated_256, ax=axs[2, j], cmap='rocket', cbar=(i % sample_batch_size == sample_batch_size - 1),
                    vmin=min_val, vmax=max_val)
        axs[2, j].set_title(f"$Generated$ " + str(j + 1))
        axs[2, j].set_xticks(ticks_3)
        axs[2, j].set_yticks(ticks_3)
        axs[2, j].set_xticklabels(tick_labels_3, rotation=0)
        axs[2, j].set_yticklabels(tick_labels_3)
        axs[2, j].invert_yaxis()

    # Adjust tick parameters for all axes
    for ax in axs.flat:
        ax.tick_params(axis='both', which='major', labelsize=16)

    # Adjust layout and display the plot
    plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.5)
    plt.show()


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