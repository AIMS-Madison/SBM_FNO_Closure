import sys
sys.path.append('C:\\UWMadisonResearch\\SBM_FNO_Closure')
import h5py
import torch
import torch.nn.functional as F
from utility import set_seed


##############################
#######  Data Loading ########
##############################
# Load raw data
device = torch.device('cuda')
filename = 'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Data_Generation\\2040_256_raw_v2.h5'
with h5py.File(filename, 'r') as file:
    sol_t = torch.tensor(file['t'][()], device='cuda')
    sol = torch.tensor(file['sol'][()], device='cuda')
    diffusion = torch.tensor(file['diffusion'][()], device='cuda')
    nonlinear = torch.tensor(file['nonlinear'][()], device='cuda')

# Only taking segments between 20s and 40s
sol_sliced_train = sol[:95, :, :, 0:201]
sol_sliced_test = sol[95:100, :, :, 0:201]
diffusion_sliced_train = diffusion[:95, :, :, 0:201]
diffusion_sliced_test = diffusion[95:100, :, :, 0:201]
nonlinear_sliced_train = nonlinear[:95, :, :, 0:201]
nonlinear_sliced_test = nonlinear[95:100, :, :, 0:201]

sol_reshaped_train = sol_sliced_train.permute(0,3,1,2).reshape(-1, 256, 256)
diffusion_reshaped_train = diffusion_sliced_train.permute(0,3,1,2).reshape(-1, 256, 256)
nonlinear_reshaped_train = nonlinear_sliced_train.permute(0,3,1,2).reshape(-1, 256, 256)

sol_reshaped_test = sol_sliced_test.permute(0,3,1,2).reshape(-1, 256, 256)
diffusion_reshaped_test = diffusion_sliced_test.permute(0,3,1,2).reshape(-1, 256, 256)
nonlinear_reshaped_test = nonlinear_sliced_test.permute(0,3,1,2).reshape(-1, 256, 256)


set_seed(42)
indices = torch.randperm(nonlinear_reshaped_train.shape[0])
sol_reshaped_train = sol_reshaped_train[indices]
diffusion_reshaped_train = diffusion_reshaped_train[indices]
nonlinear_reshaped_train = nonlinear_reshaped_train[indices]

set_seed(42)
indiced_test = torch.randperm(nonlinear_reshaped_test.shape[0])
sol_reshaped_test = sol_reshaped_test[indiced_test]
diffusion_reshaped_test = diffusion_reshaped_test[indiced_test]
nonlinear_reshaped_test = nonlinear_reshaped_test[indiced_test]

# Train/Test
Ntrain = 19000
Ntest = 1000

train_diffusion_256 = diffusion_reshaped_train[:Ntrain, :, :]
train_vorticity_256= sol_reshaped_train[:Ntrain, :, :]
train_nonlinear_256 = nonlinear_reshaped_train[:Ntrain, :, :]

test_diffusion_256 = diffusion_reshaped_test[:Ntest, :, :]
test_vorticity_256 = sol_reshaped_test[:Ntest, :, :]
test_nonlinear_256 = nonlinear_reshaped_test[:Ntest, :, :]

# Downsampling
train_diffusion_64 = train_diffusion_256[:, ::4, ::4]
train_vorticity_64 = train_vorticity_256[:, ::4, ::4]
train_nonlinear_64 = train_nonlinear_256[:, ::4, ::4]

test_diffusion_64 = test_diffusion_256[:, ::4, ::4]
test_vorticity_64 = test_vorticity_256[:, ::4, ::4]
test_nonlinear_64 = test_nonlinear_256[:, ::4, ::4]

test_diffusion_128 = test_diffusion_256[:, ::2, ::2]
test_vorticity_128 = test_vorticity_256[:, ::2, ::2]
test_nonlinear_128 = test_nonlinear_256[:, ::2, ::2]

# Sparse information for interpolation conditioning
def interpolate2d(data, sparse_res):
    down_factor = data.shape[-1] // sparse_res
    data_sparse = data[:, ::down_factor, ::down_factor]
    interp_res = data.shape[-1] + 1 - down_factor
    data_sparse_interp = F.interpolate(data_sparse.unsqueeze(1),
                                       size=(interp_res, interp_res), mode='bicubic')
    data_sparse_interp = F.pad(data_sparse_interp, (0, down_factor-1, 0, down_factor-1), mode='replicate')
    return data_sparse_interp.squeeze(1)

train_diffusion_64_sparse_interp = interpolate2d(train_diffusion_64, 16)
train_nonlinear_64_sparse_interp = interpolate2d(train_nonlinear_64, 16)
test_diffusion_64_sparse_interp = interpolate2d(test_diffusion_64, 16)
test_nonlinear_64_sparse_interp = interpolate2d(test_nonlinear_64, 16)

test_diffusion_128_sparse_interp = interpolate2d(test_diffusion_128, 16)
test_nonlinear_128_sparse_interp = interpolate2d(test_nonlinear_128, 16)
test_diffusion_256_sparse_interp = interpolate2d(test_diffusion_256, 16)
test_nonlinear_256_sparse_interp = interpolate2d(test_nonlinear_256, 16)



# def smoothConv2d(data, sparse_res, kernel_size):
#     B, Nx, Ny = data.shape
#     device = data.device
#     ratio = Nx // sparse_res
#
#     mask = torch.zeros_like(data)
#     mask[:, ::ratio, ::ratio] = 1
#     data_sparse = data * mask
#
#     kernel = torch.ones(1, 1, kernel_size, kernel_size) / kernel_size ** 2
#     kernel = kernel.to(device)
#
#     data_smoothed = F.conv2d(data_sparse.unsqueeze(1), kernel, padding='same').squeeze(1)
#
#     data_normalized = torch.empty_like(data_smoothed)
#     for i in range(data_smoothed.shape[0]):
#         batch_sparse = data_sparse[i][data_sparse[i] != 0]
#         batch_smoothed = data_smoothed[i][data_smoothed[i] != 0]
#         sparse_min, sparse_max = torch.min(batch_sparse), torch.max(batch_sparse)
#         smoothed_min, smoothed_max = torch.min(batch_smoothed), torch.max(batch_smoothed)
#         batch_normalized = (data_smoothed[i] - smoothed_min) / (smoothed_max - smoothed_min)
#         batch_normalized = batch_normalized * (sparse_max - sparse_min) + sparse_min
#         data_normalized[i] = batch_normalized
#
#     return data_normalized
#
# train_diffusion_64_sparse_normalized = smoothConv2d(train_diffusion_64, 16, 7)
# train_nonlinear_64_sparse_normalized = smoothConv2d(train_nonlinear_64, 16, 7)
# test_diffusion_64_sparse_normalized = smoothConv2d(test_diffusion_64, 16, 7)
# test_nonlinear_64_sparse_normalized = smoothConv2d(test_nonlinear_64, 16, 7)
# test_diffusion_128_sparse_normalized = smoothConv2d(test_diffusion_128, 16, 15)
# test_nonlinear_128_sparse_normalized = smoothConv2d(test_nonlinear_128, 16, 15)
# test_diffusion_256_sparse_normalized = smoothConv2d(test_diffusion_256, 16, 31)
# test_nonlinear_256_sparse_normalized = smoothConv2d(test_nonlinear_256, 16, 31)


filename = 'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Data_Generation\\train_diffusion_nonlinear_v2.h5'
with h5py.File(filename, 'w') as file:
    file.create_dataset('train_vorticity_64', data=train_vorticity_64.cpu().numpy())

    file.create_dataset('train_diffusion_64', data=train_diffusion_64.cpu().numpy())
    file.create_dataset('train_nonlinear_64', data=train_nonlinear_64.cpu().numpy())
    file.create_dataset('train_diffusion_64_sparse_interp', data=train_diffusion_64_sparse_interp.cpu().numpy())
    # file.create_dataset('train_diffusion_64_sparse_normalized', data=train_diffusion_64_sparse_normalized.cpu().numpy())
    file.create_dataset('train_nonlinear_64_sparse_interp', data=train_nonlinear_64_sparse_interp.cpu().numpy())
    # file.create_dataset('train_nonlinear_64_sparse_normalized', data=train_nonlinear_64_sparse_normalized.cpu().numpy())

filename = 'C:\\UWMadisonResearch\\SBM_FNO_Closure\\Data_Generation\\test_diffusion_nonlinear_v2.h5'
with h5py.File(filename, 'w') as file:
    file.create_dataset('test_diffusion_64', data=test_diffusion_64.cpu().numpy())
    file.create_dataset('test_vorticity_64', data=test_vorticity_64.cpu().numpy())
    file.create_dataset('test_nonlinear_64', data=test_nonlinear_64.cpu().numpy())
    # #
    file.create_dataset('test_diffusion_128', data=test_diffusion_128.cpu().numpy())
    file.create_dataset('test_vorticity_128', data=test_vorticity_128.cpu().numpy())
    file.create_dataset('test_nonlinear_128', data=test_nonlinear_128.cpu().numpy())
    # #
    file.create_dataset('test_diffusion_256', data=test_diffusion_256.cpu().numpy())
    file.create_dataset('test_vorticity_256', data=test_vorticity_256.cpu().numpy())
    file.create_dataset('test_nonlinear_256', data=test_nonlinear_256.cpu().numpy())
    #
    file.create_dataset('test_diffusion_64_sparse_interp', data=test_diffusion_64_sparse_interp.cpu().numpy())
    file.create_dataset('test_diffusion_128_sparse_interp', data=test_diffusion_128_sparse_interp.cpu().numpy())
    file.create_dataset('test_diffusion_256_sparse_interp', data=test_diffusion_256_sparse_interp.cpu().numpy())
    # file.create_dataset('test_diffusion_64_sparse_normalized', data=test_diffusion_64_sparse_normalized.cpu().numpy())
    # file.create_dataset('test_diffusion_128_sparse_normalized', data=test_diffusion_128_sparse_normalized.cpu().numpy())
    # file.create_dataset('test_diffusion_256_sparse_normalized', data=test_diffusion_256_sparse_normalized.cpu().numpy())

    file.create_dataset('test_nonlinear_64_sparse_interp', data=test_nonlinear_64_sparse_interp.cpu().numpy())
    file.create_dataset('test_nonlinear_128_sparse_interp', data=test_nonlinear_128_sparse_interp.cpu().numpy())
    file.create_dataset('test_nonlinear_256_sparse_interp', data=test_nonlinear_256_sparse_interp.cpu().numpy())
    # file.create_dataset('test_nonlinear_64_sparse_normalized', data=test_nonlinear_64_sparse_normalized.cpu().numpy())
    # file.create_dataset('test_nonlinear_128_sparse_normalized', data=test_nonlinear_128_sparse_normalized.cpu().numpy())
    # file.create_dataset('test_nonlinear_256_sparse_normalized', data=test_nonlinear_256_sparse_normalized.cpu().numpy())


