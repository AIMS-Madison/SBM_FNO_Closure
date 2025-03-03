import sys
sys.path.append('C:\\SBM_FNO_Closure\\Conditional_Score_FNO')
import torch
import math
import time
from tqdm import tqdm
from Data_Generation.random_forcing import get_twod_bj, get_twod_dW

# a: domain where we are solving
# w0: initial vorticity
# f: deterministic forcing term
# visc: viscosity (1/Re)
# T: final time
# delta_t: internal time-step for solve (descrease if blow-up)
# record_steps: number of in-time snapshots to record
def navier_stokes_2d(a, w0, f, visc, T, delta_t, record_steps, thres = 30000, stochastic_forcing=None):
    # Grid size - must be power of 2
    N1, N2 = w0.size()[-2], w0.size()[-1]

    #Maximum frequency
    k_max = math.floor(N1/2.0)

    # Number of steps to final time
    steps = math.ceil(T / delta_t)

    # Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)
    # Forcing to Fourier space
    f_h = torch.fft.rfft2(f)
    # If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # If stochastic forcing
    if stochastic_forcing is not None:
        # initialise noise
        bj = get_twod_bj(delta_t, [N1, N2], a, stochastic_forcing['alpha'], w_h.device)


    # Record solution every this number of steps
    record_time = math.floor((steps-thres) / record_steps)

    #Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                     torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N1,1)
    #Wavenumbers in x-direction
    k_x = k_y.transpose(0,1)

    #Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    # Physical wavenumbers
    kx_2d = 2.0*torch.pi*k_x / a[0]
    ky_2d = 2.0*torch.pi*k_y / a[1]


    # Negative Laplacian in Fourier space
    lap = kx_2d ** 2 + ky_2d ** 2
    lap[0, 0] = 1.0

    #Dealiasing mask
    dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0/3.0)*k_max,
                                                torch.abs(k_x) <= (2.0/3.0)*k_max).float(), 0)

    # Saving solution and time
    sol = torch.zeros(*w0.size(), record_steps+1, device=w0.device)
    diffusion = torch.zeros(*w0.size(), record_steps+1, device=w0.device)
    nonlinear = torch.zeros(*w0.size(), record_steps+1, device=w0.device)
    sol_t = torch.zeros(record_steps+1, device=w0.device)

    # Record counter
    c = 0
    # Physical time
    t = 0.0

    for j in tqdm(range(steps+1)):
        # Stream function in Fourier space: solve Poisson equation
        psi_h = w_h / lap

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

        if stochastic_forcing:
            dW, dW2 = get_twod_dW(bj, stochastic_forcing['kappa'], w_h.shape[0], w_h.device)
            gudWh = stochastic_forcing['sigma'] * torch.fft.rfft2(dW)
        else:
            gudWh = torch.zeros_like(f_h)

        F_Stochastic_h = -F_h + gudWh
        diffusion_h = -visc * lap * w_h + 2 * gudWh

        if torch.isnan(w_h).any():
            print('Break')
            sol = torch.full(sol.size(), float('nan'))
            break

        # Update real time (used only for recording)
        if j >= thres:
            if j == thres:
                w = torch.fft.irfft2(w_h, s=(N1, N2))
                diffusion_term = torch.fft.irfft2(diffusion_h, s=(N1, N2))
                nonlinear_term = torch.fft.irfft2(F_Stochastic_h,  s=(N1, N2))
                sol[..., 0] = w
                diffusion[..., 0] = diffusion_term
                nonlinear[..., 0] = nonlinear_term
                sol_t[0] = 0

                c += 1

            if j != thres and (j) % record_time == 0:
                w = torch.fft.irfft2(w_h, s=(N1, N2))
                diffusion_term = torch.fft.irfft2(diffusion_h, s=(N1, N2))
                nonlinear_term = torch.fft.irfft2(F_Stochastic_h,  s=(N1, N2))
                sol[..., c] = w
                sol_t[c] = t
                diffusion[..., c] = diffusion_term
                nonlinear[..., c] = nonlinear_term
                c += 1

        t += delta_t

        w_h = ((w_h -delta_t * F_h + delta_t * f_h + delta_t * gudWh
                        - 0.5 * delta_t * visc * lap * w_h)
                      / (1.0 + 0.5 * delta_t * visc * lap))

    return sol, sol_t, diffusion, nonlinear


def navier_stokes_2d_closure(a, w0, f, visc, sparse_condition, sampler,
                             missing_term='diffusion', closure=False,
                             delta_t=1e-4, record_steps=1, eval_steps=10):
    """
    Unified 2D Navier–Stokes solver with closure.

    Parameters:
      a               : Tuple/list of domain lengths (used to scale the wavenumbers).
      w0              : Initial vorticity (tensor; last two dimensions are spatial).
      f               : Forcing field (tensor).
      visc            : Viscosity coefficient.
      sparse_condition: Tensor with conditions for the closure sampler.
      sampler         : Function that takes (w, condition) and returns a closure field.
      missing_term    : Choose 'diffusion' or 'nonlinear'.
      closure         : If True, incorporate the closure via sampler; otherwise use basic update.
      delta_t         : Time step.
      record_steps    : Number of time steps to record.
      eval_steps      : Frequency (in steps) at which to evaluate the sampler.

    Returns:
      sol             : Recorded physical vorticity (tensor of shape w0.size() + (record_steps,)).
      sol_t           : 1D tensor with time stamps.
      execution_time  : Total execution time in seconds.
    """
    # Grid size (assumes power-of-2)
    N1, N2 = w0.size()[-2], w0.size()[-1]
    k_max = math.floor(N1 / 2.0)

    # Transform initial vorticity and forcing to Fourier space
    w_h = torch.fft.rfft2(w0)
    f_h = torch.fft.rfft2(f)
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Set up wavenumber arrays in the y-direction then x-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, device=w0.device),
                     torch.arange(start=-k_max, end=0, device=w0.device)), 0).repeat(N1, 1)
    k_x = k_y.transpose(0, 1)
    # Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    # Physical wavenumbers and negative Laplacian in Fourier space
    kx_2d = 2.0 * torch.pi * k_x / a[0]
    ky_2d = 2.0 * torch.pi * k_y / a[1]
    lap = kx_2d ** 2 + ky_2d ** 2
    lap[0, 0] = 1.0  # avoid division by zero

    # Dealiasing mask (only needed for the diffusion branch)
    if missing_term == 'diffusion':
        dealias = torch.unsqueeze(torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max,
                                                    torch.abs(k_x) <= (2.0 / 3.0) * k_max).float(), 0)

    # Prepare solution storage
    sol = torch.zeros(*w0.size(), record_steps, device=w0.device)
    sol_t = torch.zeros(record_steps, device=w0.device)
    t = 0.0

    # Initialize closure sample variable if needed
    if closure:
        if missing_term == 'diffusion':
            diffusion_sample = None
        elif missing_term == 'nonlinear':
            nonlinear_sample = None

    start_time = time.time()
    for i in tqdm(range(record_steps)):
        if missing_term == 'diffusion':
            # Compute streamfunction in Fourier space
            psi_h = w_h / lap

            # Velocity field in x-direction = psi_y
            q = ky_2d * 1j * psi_h
            q = torch.fft.irfft2(q, s=(N1, N2))

            # Velocity field in y-direction = -psi_x
            v = -kx_2d * 1j * psi_h
            v = torch.fft.irfft2(v, s=(N1, N2))

            # Compute spatial derivatives of vorticity
            w_x = kx_2d * 1j * w_h
            w_x = torch.fft.irfft2(w_x, s=(N1, N2))
            w_y = ky_2d * 1j * w_h
            w_y = torch.fft.irfft2(w_y, s=(N1, N2))

            # Non-linear term: u·grad(w), computed in physical space then back to Fourier space
            F_h = torch.fft.rfft2(q * w_x + v * w_y)
            F_h = dealias * F_h

            # Get physical vorticity (for the closure sampler)
            w = torch.fft.irfft2(w_h, s=(N1, N2))

            if closure:
                if i % eval_steps == 0:
                    diffusion_sample = sampler(w, sparse_condition[:, :, :, i])
                else:
                    diffusion_sample = diffusion_sample + torch.randn_like(diffusion_sample) * 0.00005
                diffusion_h = torch.fft.rfft2(diffusion_sample)
                w_h = (w_h - delta_t * F_h + delta_t * f_h + 0.5 * delta_t * diffusion_h) / \
                      (1.0 + 0.5 * delta_t * visc * lap)
            else:
                w_h = (w_h - delta_t * F_h + delta_t * f_h) / (1.0 + 0.5 * delta_t * visc * lap)

        elif missing_term == 'nonlinear':
            # Obtain physical vorticity field
            w = torch.fft.irfft2(w_h, s=(N1, N2))
            if closure:
                if i % eval_steps == 0:
                    nonlinear_sample = sampler(w, sparse_condition[:, :, :, i])
                else:
                    nonlinear_sample = nonlinear_sample + torch.randn_like(nonlinear_sample) * 0.00005
                nonlinear_h = torch.fft.rfft2(nonlinear_sample)
                w_h = (w_h + delta_t * f_h + delta_t * nonlinear_h - 0.5 * delta_t * visc * lap * w_h) / \
                      (1.0 + 0.5 * delta_t * visc * lap)
            else:
                w_h = (w_h + delta_t * f_h - 0.5 * delta_t * visc * lap * w_h) / \
                      (1.0 + 0.5 * delta_t * visc * lap)
        else:
            raise ValueError("Unknown missing term. Please choose 'diffusion' or 'nonlinear'.")

        # Record the physical space solution and update time
        sol[..., i] = torch.fft.irfft2(w_h, s=(N1, N2))
        sol_t[i] = t
        t += delta_t

    execution_time = time.time() - start_time
    return sol, sol_t, execution_time
