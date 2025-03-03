import sys
sys.path.append('C:\\UWMadisonResearch\\SBM_FNO_Closure\\DiffusionTerm_Generation')

import h5py
import torch
import matplotlib.pyplot as plt
plt.rcParams["animation.html"] = "jshtml"
from torch.optim import Adam
from functools import partial
from tqdm import trange
from utility import set_seed

from Model_Designs import (marginal_prob_std, diffusion_coeff, FNO2d, loss_fn)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')

# Load the data
train_file = 'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Data_Generation\\train_diffusion_nonlinear.h5'
with h5py.File(train_file, 'r') as file:
    train_diffusion_64 = torch.tensor(file['train_diffusion_64'][:], device=device)
    train_vorticity_64 = torch.tensor(file['train_vorticity_64'][:], device=device)
    train_diffusion_64_sparse_interp = torch.tensor(file['train_diffusion_64_sparse_interp'][:], device=device)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_diffusion_64,
                                                                          train_vorticity_64,
                                                                          train_diffusion_64_sparse_interp),
                                                                          batch_size=80, shuffle=True)

set_seed(42)
################################
######## Model Training ########
################################
sigma = 26
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 8
width = 20
epochs = 1000
learning_rate = 0.001
scheduler_step = 200
scheduler_gamma = 0.5

model = FNO2d(marginal_prob_std_fn, modes, modes, width).cuda()
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

tqdm_epoch = trange(epochs)

loss_history = []

for epoch in tqdm_epoch:
    model.train()
    avg_loss = 0.
    num_items = 0
    for x, w, x_sparse in train_loader:
        x, w, x_sparse = x.cuda(), w.cuda(), x_sparse.cuda()
        optimizer.zero_grad()
        loss, score = loss_fn(model, x, w, x_sparse, marginal_prob_std_fn)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
    scheduler.step()
    avg_loss_epoch = avg_loss / num_items
    loss_history.append(avg_loss_epoch)
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
torch.save(model.state_dict(), 'SparseDiffusionModel_Interp_v3.pth')







import matplotlib.gridspec as gridspec

# Time values in seconds for the x-axis
time_values = [30, 35, 40, 45, 50]

# MSE and RMSE data for simulations

sim_vort_mse_II = [0, 1.5259e-04, 2.8789e-04, 7.3273e-04, 1.5795e-03]
sim_vort_rmse_II = [0, 1.3415e-02, 2.2945e-02, 3.0611e-02, 4.3552e-02]
sim_vort_mse_III = [0, 2.2781e-04, 5.2549e-04, 1.5526e-03, 3.8183e-03]
sim_vort_rmse_III = [0, 1.6223e-02, 2.4895e-02, 4.3086e-02, 6.9365e-02]
sim_vort_mse_IV = [0, 1.5088e-04, 2.8789e-04, 4.1774e-04, 7.9800e-04]
sim_vort_rmse_IV = [0, 1.3356e-02, 1.8562e-02, 2.2601e-02, 3.2098e-02]


# Create a figure with a custom gridspec layout
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Qt5Agg")
plt.rcParams["agg.path.chunksize"] = 10000
plt.rc("text", usetex=True)
mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

# Create a figure with two subplots in a 42x12 inch figure.
fig, axs = plt.subplots(1, 2, figsize=(42, 12))
# Reserve space: left/right margins, a top margin (82% of height for axes) and a bottom margin (15%)
# wspace=0.333 controls the space between the two subplots.
plt.subplots_adjust(left=0.111, right=0.889, top=0.748, bottom=0.15, wspace=0.333)

fs = 60

# MSE Plot
ax0 = axs[0]
ax0.plot(time_values, sim_vort_mse_II, marker='o', linestyle=":", markersize=10, linewidth=6, label=f"Simulation II")
ax0.plot(time_values, sim_vort_mse_III, marker='o', linestyle="--", markersize=10, linewidth=6, label=f"Simulation III")
ax0.plot(time_values, sim_vort_mse_IV, marker='o', linestyle="-.", markersize=10, linewidth=6, label=f"Simulation IV")
ax0.set_title(r"$D_{\text{MSE}}$ \text{Comparison}", fontsize=fs, pad=16)
ax0.set_xlabel(r"$t$", fontsize=fs)
ax0.set_ylabel(r"$D_{\text{MSE}}$", fontsize=fs)
ax0.set_xticks([30, 35, 40, 45, 50])
ax0.set_yticks([0.000, 0.001, 0.002])
ax0.tick_params(axis='both', which='major', labelsize=fs, width=2, length=14)
for spine in ax0.spines.values():
    spine.set_linewidth(2)

# RMSE Plot
ax1 = axs[1]
ax1.plot(time_values, sim_vort_rmse_II, marker='o', linestyle=":", markersize=10, linewidth=6, label=f"Simulation II")
ax1.plot(time_values, sim_vort_rmse_III, marker='o', linestyle="--", markersize=10, linewidth=6, label=f"Simulation III")
ax1.plot(time_values, sim_vort_rmse_IV, marker='o', linestyle="-.", markersize=10, linewidth=6, label=f"Simulation IV")
ax1.set_title(r"$D_{\text{Fro}}$ \text{Comparison}", fontsize=fs, pad=16)
ax1.set_xlabel(r"$t$", fontsize=fs)
ax1.set_ylabel(r"$D_{\text{Fro}}$", fontsize=fs)
ax1.set_xticks([30, 35, 40, 45, 50])
ax1.set_yticks([0.00, 0.02, 0.04, 0.06])
ax1.tick_params(axis='both', which='major', labelsize=fs, width=2, length=14)
for spine in ax1.spines.values():
    spine.set_linewidth(2)

# Create a shared legend at the top center, outside the axes
handles, labels = ax0.get_legend_handles_labels()
lege = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=fs,
                  bbox_to_anchor=(0.5, 1), fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(2)

# Save the figure as a PDF ensuring nothing overlaps
plt.savefig('C:\\UWMadisonResearch\\SBM_FNO_Closure\\Plots\\MSE_RE_Comparison_G.png', dpi=300)
plt.show()






